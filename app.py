"""
app.py — Dashboard Streamlit de goniometria digital em tempo real
=================================================================

Ponto de entrada principal do sistema. Este arquivo concentra APENAS
a lógica de interface e orquestração — todo o processamento de vídeo
vive em realtime_processor.py.

Fluxo de dados:
    av.VideoFrame (WebRTC)
    → VideoProcessor.recv()          [realtime_processor.py]
    → MediaPipe Hands
    → DigitalGoniometer.compute_all()
    → GoniometryFilterBank.smooth_all()
    → draw_goniometry_overlay()
    → buffers circulares em st.session_state  (thread-safe via Lock)
    → GoniometryCSVLogger

    @st.fragment (a cada PANEL_REFRESH_S)
    → _copy_state_snapshot()         leitura thread-safe
    → classify_hand_state()          [dashboard_utils]
    → compute_realtime_metrics()     [dashboard_utils]
    → build_tam_chart_data()         [dashboard_utils]
    → st.line_chart / st.metric / st.container

Separação de threads:
    - Thread WebRTC (recv): só escreve em session_state, sempre com Lock.
    - Thread Streamlit (fragment): só lê snapshots copiados, com Lock mínimo.
"""

import time
import threading
from collections import deque
from typing import Dict, Optional

import av  # noqa: F401 — importado para type hint de VideoFrame
import streamlit as st

try:
    import mediapipe  # noqa: F401 — verificação antecipada de disponibilidade
except ImportError:
    st.error("❌ MediaPipe não encontrado. Execute: pip install mediapipe")
    st.stop()

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from realtime_processor import VideoProcessor
from dashboard_utils import (
    FINGERS,
    FINGER_JOINTS,
    classify_hand_state,
    compute_realtime_metrics,
    build_tam_chart_data,
    freq_label,
    regularidade_label,
)


# =============================================================================
# CONFIGURAÇÕES GLOBAIS DO APP
# =============================================================================

# Tamanho do buffer circular: quantos frames manter em memória.
# A 30fps, 90 frames ≈ 3 segundos de janela deslizante.
BUFFER_SIZE: int = 90

# Parâmetros de atualização do painel Streamlit.
# 5 Hz (200 ms) é o mínimo clinicamente útil: abaixo disso, exercícios
# rápidos (>0.4 Hz) geram deltas aparentemente zero entre ciclos de refresh.
PANEL_REFRESH_S: float = 0.2

FINGER_NAMES_PT: Dict[str, str] = {
    "INDEX":  "Indicador",
    "MIDDLE": "Médio",
    "RING":   "Anelar",
    "PINKY":  "Mínimo",
    "THUMB":  "Polegar",
}

# Cores para o gráfico de linha (compatíveis com st.line_chart via Vega-Lite).
FINGER_COLORS: Dict[str, str] = {
    "Indicador": "#38bdf8",
    "Médio":     "#4ade80",
    "Anelar":    "#facc15",
    "Mínimo":   "#f87171",
    "Polegar":   "#c084fc",
}

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

from goniometry_csv import GoniometryCSVLogger  # noqa: E402


# =============================================================================
# ESTADO DA SESSÃO STREAMLIT
# =============================================================================

def _init_session_state() -> None:
    """
    Inicializa o estado da sessão do Streamlit para manter persistência
    dos logs e das métricas de delta entre execuções.
    """
    if "hand_detected" not in st.session_state:
        st.session_state.hand_detected = False

    if "frame_id" not in st.session_state:
        st.session_state.frame_id = 0

    if "csv_path" not in st.session_state:
        st.session_state.csv_path = ""

    if "csv_closed" not in st.session_state:
        st.session_state.csv_closed = False

    if "prev_metrics" not in st.session_state:
        st.session_state.prev_metrics = {}


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar(processor: Optional[VideoProcessor]) -> None:
    """
    Renderiza a barra lateral do app.

    Concentra:
    - status da detecção de mão;
    - número do frame atual;
    - exportação do CSV;
    - parâmetros globais do filtro.
    """
    with st.sidebar:
        st.title("⚙️ Controles")
        st.markdown("---")

        detected = st.session_state.get("hand_detected", False)
        frame_id = st.session_state.get("frame_id", 0)

        if detected:
            st.success("✅ Mão detectada")
        else:
            st.info("⏳ Aguardando mão…")

        st.caption(f"Frame processado: #{frame_id}")
        st.markdown("---")

        st.subheader("💾 Sessão CSV")
        csv_path = st.session_state.get("csv_path", "")
        csv_closed = st.session_state.get("csv_closed", False)

        if csv_path:
            st.caption(f"Arquivo: `{csv_path}`")

            if not csv_closed:
                if st.button("💾 Salvar e fechar CSV", use_container_width=True):
                    if processor:
                        processor.close_csv()
                    st.session_state.csv_closed = True

            try:
                with open(csv_path, "rb") as f:
                    csv_bytes = f.read()

                st.download_button(
                    label="⬇️ Baixar CSV",
                    data=csv_bytes,
                    file_name=csv_path,
                    mime="text/csv",
                    use_container_width=True,
                )
            except FileNotFoundError:
                st.warning("O arquivo CSV ainda não foi criado no disco.")

        st.markdown("---")

        st.subheader("🔧 Filtro EMA → Kalman")
        st.code(
            "EMA α  = 0.30\n"
            "Kalman Q = 0.01\n"
            "Kalman R = 0.10",
            language="text",
        )
        st.caption(
            "Todos os ângulos exibidos, armazenados em buffer e salvos em CSV "
            "passam pelo pipeline EMA → Kalman antes de qualquer uso."
        )

        st.markdown("---")
        st.caption(
            f"Janela: {BUFFER_SIZE} frames (~{BUFFER_SIZE / 30:.0f}s) · "
            f"Painel: {1 / PANEL_REFRESH_S:.0f} Hz"
        )
        st.caption("Goniometria Digital · Streamlit + WebRTC")


# =============================================================================
# SNAPSHOT THREAD-SAFE
# =============================================================================

def _copy_state_snapshot(processor: VideoProcessor) -> Optional[Dict]:
    """
    Faz uma cópia thread-safe dos dados usados pelo dashboard diretamente
    do processador de vídeo do WebRTC.

    Usa FINGER_JOINTS para copiar apenas os joints válidos por dedo,
    eliminando a inconsistência de ler buffers inexistentes do THUMB.
    """
    with processor.lock:
        snapshot = {
            "angle_buffers": {
                finger: {
                    joint: list(processor.angle_buffers[finger][joint])
                    for joint in FINGER_JOINTS[finger]
                }
                for finger in FINGERS
            },
            "time_buffers": {
                finger: list(processor.time_buffers[finger])
                for finger in FINGERS
            },
            "last_angles": dict(processor.last_angles),
            "hand_detected": bool(processor.hand_detected),
            "frame_id": int(processor.frame_id),
        }

    return snapshot


# =============================================================================
# BLOCO DE UM DEDO (renderização isolada)
# =============================================================================

def _render_finger_block(
    finger: str,
    finger_name: str,
    state: Dict,
    angle_buffers: Dict,
    time_buffers: Dict,
) -> None:
    """
    Renderiza o bloco clínico de um único dedo.

    Estrutura do bloco:
    ┌─────────────────────────────────────────────────────────┐
    │ 🟢/🔴  NOME DO DEDO · TAM: 245.3° [Excelente]          │
    ├──────────────────────┬──────────────────────────────────┤
    │ Amplitude | Vel.méd. │ Vel. pico | Freq. | Regularidade │
    ├──────────────────────┴──────────────────────────────────┤
    │ Gráfico TAM (linha temporal)                            │
    └─────────────────────────────────────────────────────────┘
    """
    tam_current = state.get("TAM", 0.0)
    assh_label = state.get("assh_label", "—")
    assh_color = state.get("assh_color", "#888888")
    is_closed = state.get("closed", False)

    # O polegar não tem buffer "TAM" — usamos "MCP" como proxy de amplitude.
    if finger == "THUMB":
        tam_buffer = angle_buffers[finger]["MCP"]
    else:
        tam_buffer = angle_buffers[finger]["TAM"]
    t_buffer = time_buffers[finger]

    metrics = compute_realtime_metrics(
        angle_buffer=tam_buffer,
        time_buffer=t_buffer,
        min_range_pct=0.30,
        min_dist_s=1.0,
    )

    # Delta em relação ao ciclo anterior.
    prev = st.session_state.prev_metrics.get(finger, {})
    delta_amp = metrics["amplitude"] - prev.get("amplitude", metrics["amplitude"])
    delta_vel = metrics["vel_media"] - prev.get("vel_media", metrics["vel_media"])

    # Cabeçalho do bloco.
    status_icon = "🔴" if is_closed else "🟢"
    st.markdown(
        f"#### {status_icon} {finger_name} &nbsp;&nbsp; "
        f"<span style='color:{assh_color}; font-size:0.85em;'>"
        f"TAM: {tam_current:.1f}° — {assh_label}"
        f"</span>",
        unsafe_allow_html=True,
    )

    # Linha 1: amplitude + velocidade média (com delta).
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Amplitude",
            f"{metrics['amplitude']:.1f}°",
            delta=f"{delta_amp:+.1f}°",
        )
    with c2:
        st.metric(
            "Vel. média",
            f"{metrics['vel_media']:.1f}°/s",
            delta=f"{delta_vel:+.1f}°/s",
        )
    with c3:
        st.metric(
            "Vel. pico",
            f"{metrics['vel_pico']:.1f}°/s",
        )
    with c4:
        st.metric(
            "Frequência",
            f"{metrics['freq_hz']:.2f} Hz",
        )

    # Linha 2: rótulos textuais de frequência e regularidade.
    col_freq, col_reg = st.columns(2)
    with col_freq:
        st.caption(f"🔄 {freq_label(metrics['freq_hz'])}")
    with col_reg:
        st.caption(f"📐 {regularidade_label(metrics['regularidade'], metrics['cv'])}")

    # Gráfico de linha temporal do TAM (substitui o st.progress).
    if len(tam_buffer) >= 3:
        chart_data = {finger_name: tam_buffer}
        st.line_chart(
            chart_data,
            height=100,
            use_container_width=True,
        )
    else:
        st.caption("_Aguardando dados suficientes para o gráfico…_")

    # Persiste métricas para o próximo ciclo (delta).
    st.session_state.prev_metrics[finger] = {
        "amplitude": metrics["amplitude"],
        "vel_media": metrics["vel_media"],
    }


# =============================================================================
# PAINEL PRINCIPAL AO VIVO
# =============================================================================

@st.fragment(run_every=PANEL_REFRESH_S)
def render_live_panel(processor: Optional[VideoProcessor]) -> None:
    """
    Atualiza o painel clínico em frequência fixa (PANEL_REFRESH_S).

    Separação de frequências:
    - recv() roda na taxa da câmera (~30 Hz): escreve nos buffers do processador.
    - Este fragmento roda a ~5 Hz (200 ms): lê snapshot do processador, calcula e renderiza.
    """
    if processor is None:
        st.info("⏳ Posicione a mão na câmera para iniciar as métricas em tempo real.")
        return

    snapshot = _copy_state_snapshot(processor)
    if snapshot is None:
        st.warning("⚠️ Estado da sessão não inicializado.")
        return

    hand_detected = snapshot["hand_detected"]
    last_angles = snapshot["last_angles"]
    angle_buffers = snapshot["angle_buffers"]
    time_buffers = snapshot["time_buffers"]

    if not hand_detected or not last_angles:
        st.info("⏳ Posicione a mão na câmera para iniciar as métricas em tempo real.")
        return

    # Estado global da mão.
    hand_state = classify_hand_state(last_angles)
    finger_states = hand_state["finger_states"]
    closed_count = hand_state["closed_count"]

    col_status, col_closed = st.columns([3, 1])
    with col_status:
        if hand_state["hand_open"]:
            st.markdown("### 🟢 MÃO ABERTA")
        else:
            st.markdown("### 🔴 MÃO FECHADA")
    with col_closed:
        st.metric("Dedos fechados", f"{closed_count}/5")

    st.markdown("---")

    # Gráfico temporal unificado do TAM (todos os 4 dedos).
    tam_chart_data = build_tam_chart_data(angle_buffers)
    min_series_len = min((len(v) for v in tam_chart_data.values()), default=0)

    if min_series_len >= 3:
        st.markdown("#### 📈 TAM — janela deslizante (últimos frames)")
        st.line_chart(
            tam_chart_data,
            height=160,
            use_container_width=True,
        )
        st.caption(
            "Cada linha representa o TAM suavizado (EMA → Kalman) de um dedo. "
            f"Janela: {BUFFER_SIZE} frames (~{BUFFER_SIZE / 30:.0f}s)."
        )
    else:
        st.caption("_Aguardando frames para exibir gráfico temporal…_")

    st.markdown("---")

    # Blocos individuais por dedo.
    for finger in FINGERS:
        finger_name = FINGER_NAMES_PT[finger]
        state = finger_states.get(finger, {})
        _render_finger_block(
            finger=finger,
            finger_name=finger_name,
            state=state,
            angle_buffers=angle_buffers,
            time_buffers=time_buffers,
        )
        st.markdown("---")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Monta a interface principal do Streamlit.

    Layout:
    ┌───────────────────────┬────────────────────────────────┐
    │  Webcam ao vivo       │  Painel clínico (tempo real)   │
    │  (overlay goniomét.)  │  (fragmento atualizado 5Hz)    │
    └───────────────────────┴────────────────────────────────┘
    """
    st.set_page_config(
        page_title="Goniometria Digital",
        page_icon="🖐️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_session_state()

    st.title("🖐️ Goniometria Digital em Tempo Real")
    st.caption(
        "MediaPipe Hands · EMA → Kalman · métricas clínicas em janela deslizante · "
        f"WebRTC + Streamlit"
    )

    col_cam, col_panel = st.columns([1.2, 1])

    with col_cam:
        st.subheader("📷 Webcam ao vivo")

        ctx = webrtc_streamer(
            key="goniometry-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640, "max": 640},
                    "height": {"ideal": 480, "max": 480},
                    "frameRate": {"ideal": 30, "max": 30},
                },
                "audio": False,
            },
            video_processor_factory=VideoProcessor,
            async_processing=True,
        )

        processor = ctx.video_processor if ctx and ctx.state.playing else None

        if processor:
            # Sincroniza o estado do processador para persistência local no session_state
            st.session_state.csv_path = processor.csv_path
            st.session_state.csv_closed = processor.csv_closed
            st.session_state.hand_detected = processor.hand_detected
            st.session_state.frame_id = processor.frame_id

        if ctx and ctx.state.playing:
            detected = st.session_state.get("hand_detected", False)
            if detected:
                st.success("✅ Mão detectada — overlay ativo")
            else:
                st.info("⏳ Aguardando mão na câmera…")
        else:
            st.caption("▶️ Clique em **START** para ativar a câmera.")

        with st.expander("ℹ️ Instruções rápidas"):
            st.markdown(
                """
- Clique em **START** para ativar a webcam.
- Posicione a mão de frente para a câmera.
- O vídeo recebe overlay goniométrico em tempo real (esqueleto + ângulos).
- O painel à direita exibe métricas clínicas atualizadas a cada 200ms (5 Hz).
- O gráfico mostra o TAM suavizado (EMA → Kalman) em janela deslizante.
- Use a sidebar para salvar e exportar a sessão em CSV.
                """
            )

    render_sidebar(processor)

    with col_panel:
        st.subheader("📊 Painel funcional — tempo real")
        render_live_panel(processor)


if __name__ == "__main__":
    main()