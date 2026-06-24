"""
dashboard_utils.py — Utilitários clínicos para dashboard em tempo real
=====================================================================

Funções puras para:
- classificar estado global da mão;
- calcular métricas em janela deslizante por dedo;
- classificar TAM por faixas funcionais;
- formatar frequência e regularidade para exibição.

Importante:
- este módulo não depende de Streamlit;
- todas as métricas são calculadas sobre buffers temporais;
- não calcula indicadores clínicos frame a frame isoladamente.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple
import math
import statistics


FINGERS = ["INDEX", "MIDDLE", "RING", "PINKY", "THUMB"]
JOINTS = ["MCP", "PIP", "DIP", "ABD", "TAM"]

# Joints válidos por dedo — fonte de verdade centralizada.
# O polegar tem anatomia diferente: apenas MCP e IP.
# Demais dedos: MCP, PIP, DIP, ABD, TAM.
FINGER_JOINTS: Dict[str, Tuple[str, ...]] = {
    "INDEX":  ("MCP", "PIP", "DIP", "ABD", "TAM"),
    "MIDDLE": ("MCP", "PIP", "DIP", "ABD", "TAM"),
    "RING":   ("MCP", "PIP", "DIP", "ABD", "TAM"),
    "PINKY":  ("MCP", "PIP", "DIP", "ABD", "TAM"),
    "THUMB":  ("MCP", "IP", "TAM"),
}


# =============================================================================
# CLASSIFICAÇÃO FUNCIONAL TAM / ASSH
# =============================================================================

def assh_classify(tam: float) -> Tuple[str, str]:
    """
    Classifica o TAM por faixas funcionais (dedos longos).

    Retorna:
        (label, color_hex)

    Faixas:
    - >= 260° : Excelente
    - 195–259°: Bom
    - 130–194°: Moderado
    - < 130° : Ruim
    """
    if tam >= 260.0:
        return "Excelente", "#22c55e"
    if tam >= 195.0:
        return "Bom", "#eab308"
    if tam >= 130.0:
        return "Moderado", "#f97316"
    return "Ruim", "#ef4444"


def assh_classify_thumb(tam: float) -> Tuple[str, str]:
    """
    Classifica o TAM do polegar por faixas funcionais adaptadas.

    O polegar tem TAM máximo anatômico ~120–130° (MCP + IP),
    portanto as faixas ASSH são proporcionalmente menores.

    Retorna:
        (label, color_hex)

    Faixas:
    - >= 110° : Excelente
    - 80–109° : Bom
    - 50–79°  : Moderado
    - < 50°   : Ruim
    """
    if tam >= 110.0:
        return "Excelente", "#22c55e"
    if tam >= 80.0:
        return "Bom", "#eab308"
    if tam >= 50.0:
        return "Moderado", "#f97316"
    return "Ruim", "#ef4444"


def tam_progress(tam: float, max_tam: float = 270.0) -> float:
    """
    Normaliza o TAM para barra de progresso entre 0 e 1.
    """
    if max_tam <= 0:
        return 0.0
    return max(0.0, min(1.0, tam / max_tam))


# =============================================================================
# ESTADO GLOBAL DA MÃO
# =============================================================================

def classify_hand_state(angles_smooth: Dict[str, Dict[str, float]]) -> Dict:
    """
    Classifica o estado da mão com base no TAM atual de cada dedo.

    Regra clínica implementada:
    - Dedos longos (INDEX, MIDDLE, RING, PINKY): fechado se TAM >= 130 graus.
    - Polegar (THUMB): fechado se MCP + IP >= 80 graus (proxy de amplitude total).
    - Mão fechada se >= 4 dos 5 dedos estiverem fechados.

    Retorna:
    {
        "finger_states": {
            "INDEX": {
                "MCP": float,
                "PIP": float,
                "DIP": float,
                "ABD": float,
                "TAM": float,
                "closed": bool,
                "assh_label": str,
                "assh_color": str,
            },
            "THUMB": {
                "MCP": float,
                "IP": float,
                "TAM": float,   # proxy: MCP + IP
                "closed": bool,
                "assh_label": str,
                "assh_color": str,
            },
            ...
        },
        "closed_count": int,
        "hand_open": bool,
    }
    """
    finger_states: Dict[str, Dict[str, float]] = {}
    closed_count = 0

    for finger in FINGERS:
        finger_data = angles_smooth.get(finger, {})

        if finger == "THUMB":
            mcp = float(finger_data.get("MCP", 0.0))
            ip  = float(finger_data.get("IP", 0.0))
            tam = float(finger_data.get("TAM", 0.0))
            # Threshold de fechado proporcional ao TAM máximo do polegar (~120°)
            closed = tam >= 85.0
            if closed:
                closed_count += 1
            assh_label, assh_color = assh_classify_thumb(tam)
            finger_states[finger] = {
                "MCP": mcp,
                "IP": ip,
                "TAM": tam,
                "closed": closed,
                "assh_label": assh_label,
                "assh_color": assh_color,
            }
        else:
            mcp = float(finger_data.get("MCP", 0.0))
            pip = float(finger_data.get("PIP", 0.0))
            dip = float(finger_data.get("DIP", 0.0))
            abd = float(finger_data.get("ABD", 0.0))
            tam = float(finger_data.get("TAM", 0.0))

            closed = tam >= 130.0
            if closed:
                closed_count += 1

            assh_label, assh_color = assh_classify(tam)

            finger_states[finger] = {
                "MCP": mcp,
                "PIP": pip,
                "DIP": dip,
                "ABD": abd,
                "TAM": tam,
                "closed": closed,
                "assh_label": assh_label,
                "assh_color": assh_color,
            }

    hand_open = closed_count < 4

    return {
        "finger_states": finger_states,
        "closed_count": closed_count,
        "hand_open": hand_open,
    }


# =============================================================================
# DETECÇÃO DE PICOS
# =============================================================================

def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return default
    return float(statistics.mean(values))


def _safe_std(values: Sequence[float], default: float = 0.0) -> float:
    if len(values) < 2:
        return default
    return float(statistics.pstdev(values))


def _estimate_fps(time_values: Sequence[float]) -> float:
    """
    Estima FPS médio a partir dos timestamps do buffer.
    """
    if len(time_values) < 2:
        return 0.0

    dts = []
    for i in range(1, len(time_values)):
        dt = time_values[i] - time_values[i - 1]
        if dt > 1e-6:
            dts.append(dt)

    if not dts:
        return 0.0

    mean_dt = _safe_mean(dts, default=0.0)
    if mean_dt <= 1e-6:
        return 0.0

    return 1.0 / mean_dt


def _detect_peaks(
    angle_values: Sequence[float],
    time_values: Sequence[float],
    min_range_pct: float = 0.30,
    min_dist_s: float = 1.0,
) -> List[int]:
    """
    Detecta picos locais (máximos) em uma janela deslizante.

    Critérios:
    - máximo local com tolerância a platôs: curr >= prev AND curr > next
      (o crítrio anterior curr > prev AND curr >= next falhava quando o
       sinal suavizado produzia platôs, marcando o pico na borda errada).
    - valor do pico acima de um limiar relativo ao range local:
        threshold = min(angle) + min_range_pct * (max(angle) - min(angle))
    - distância mínima entre picos em segundos convertida para samples.

    Retorna:
        lista de índices dos picos detectados
    """
    n = len(angle_values)
    if n < 3 or len(time_values) != n:
        return []

    min_val = min(angle_values)
    max_val = max(angle_values)
    amplitude = max_val - min_val

    if amplitude <= 1e-6:
        return []

    threshold = min_val + min_range_pct * amplitude

    fps_est = _estimate_fps(time_values)
    if fps_est <= 0:
        min_dist_samples = 1
    else:
        min_dist_samples = max(1, int(round(min_dist_s * fps_est)))

    candidate_peaks: List[int] = []
    for i in range(1, n - 1):
        prev_v = angle_values[i - 1]
        curr_v = angle_values[i]
        next_v = angle_values[i + 1]

        # Tolerância a platô: aceita curr == prev, mas exige curr > next.
        # Isso garante que o primeiro ponto de um platô seja aceito como pico.
        is_local_peak = (curr_v >= prev_v) and (curr_v > next_v)
        if is_local_peak and curr_v >= threshold:
            candidate_peaks.append(i)

    if not candidate_peaks:
        return []

    filtered_peaks: List[int] = [candidate_peaks[0]]
    for idx in candidate_peaks[1:]:
        last_idx = filtered_peaks[-1]
        if idx - last_idx < min_dist_samples:
            if angle_values[idx] > angle_values[last_idx]:
                filtered_peaks[-1] = idx
        else:
            filtered_peaks.append(idx)

    return filtered_peaks


def _detect_valleys(
    angle_values: Sequence[float],
    time_values: Sequence[float],
    min_range_pct: float = 0.30,
    min_dist_s: float = 1.0,
) -> List[int]:
    """
    Detecta vales locais (mínimos) em uma janela deslizante.

    Simétrico a _detect_peaks, mas inverte a direção do sinal.
    Necessário para contar ciclos completos (flexão + extensão).
    """
    n = len(angle_values)
    if n < 3 or len(time_values) != n:
        return []

    min_val = min(angle_values)
    max_val = max(angle_values)
    amplitude = max_val - min_val

    if amplitude <= 1e-6:
        return []

    # Limiar: vale deve estar abaixo de (max - min_range_pct * amplitude)
    threshold = max_val - min_range_pct * amplitude

    fps_est = _estimate_fps(time_values)
    if fps_est <= 0:
        min_dist_samples = 1
    else:
        min_dist_samples = max(1, int(round(min_dist_s * fps_est)))

    candidate_valleys: List[int] = []
    for i in range(1, n - 1):
        prev_v = angle_values[i - 1]
        curr_v = angle_values[i]
        next_v = angle_values[i + 1]

        is_local_valley = (curr_v <= prev_v) and (curr_v < next_v)
        if is_local_valley and curr_v <= threshold:
            candidate_valleys.append(i)

    if not candidate_valleys:
        return []

    filtered_valleys: List[int] = [candidate_valleys[0]]
    for idx in candidate_valleys[1:]:
        last_idx = filtered_valleys[-1]
        if idx - last_idx < min_dist_samples:
            if angle_values[idx] < angle_values[last_idx]:
                filtered_valleys[-1] = idx
        else:
            filtered_valleys.append(idx)

    return filtered_valleys


# =============================================================================
# MÉTRICAS EM TEMPO REAL
# =============================================================================

def compute_realtime_metrics(
    angle_buffer: Sequence[float] | Deque[float],
    time_buffer: Sequence[float] | Deque[float],
    min_range_pct: float = 0.30,
    min_dist_s: float = 1.0,
) -> Dict[str, float | int | str]:
    """
    Calcula métricas em tempo real para UM dedo com base no TAM recente.

    Parâmetros:
        angle_buffer: TAM dos últimos N frames
        time_buffer : timestamps correspondentes
        min_range_pct: percentual mínimo do range para aceitar um pico
        min_dist_s: distância mínima entre picos consecutivos

    Retorna:
    {
        "amplitude": float,
        "vel_media": float,
        "vel_pico": float,
        "freq_hz": float,
        "cv": float,
        "regularidade": str,
        "n_picos": int,
    }
    """
    angles = list(angle_buffer)
    times = list(time_buffer)

    if len(angles) < 2 or len(times) < 2 or len(angles) != len(times):
        return {
            "amplitude": 0.0,
            "vel_media": 0.0,
            "vel_pico": 0.0,
            "freq_hz": 0.0,
            "cv": 0.0,
            "regularidade": "Regular",
            "n_picos": 0,
        }

    amplitude = float(max(angles) - min(angles))

    velocities: List[float] = []
    for i in range(1, len(angles)):
        d_angle = abs(angles[i] - angles[i - 1])
        d_time = times[i] - times[i - 1]
        if d_time > 1e-6:
            velocities.append(d_angle / d_time)

    vel_media = _safe_mean(velocities, default=0.0)
    vel_pico = max(velocities) if velocities else 0.0

    peaks = _detect_peaks(
        angle_values=angles,
        time_values=times,
        min_range_pct=min_range_pct,
        min_dist_s=min_dist_s,
    )
    valleys = _detect_valleys(
        angle_values=angles,
        time_values=times,
        min_range_pct=min_range_pct,
        min_dist_s=min_dist_s,
    )
    n_picos = len(peaks)
    n_extremos = len(peaks) + len(valleys)

    # Frequência calculada por ciclos completos:
    # - um ciclo = 1 pico + 1 vale (flexão + extensão).
    # - dividimos n_extremos por 2 para obter ciclos completos.
    # - com 2+ picos: usa intervalo entre primeiro e último pico
    #   para evitar subestimativa quando picos estão concentrados
    #   no início da janela.
    peak_times = [times[idx] for idx in peaks]
    duration = times[-1] - times[0]
    if n_picos >= 2:
        pico_duration = peak_times[-1] - peak_times[0]
        freq_hz = (n_extremos / 2.0) / pico_duration if pico_duration > 1e-6 else 0.0
    elif n_extremos >= 2:
        freq_hz = (n_extremos / 2.0) / duration if duration > 1e-6 else 0.0
    else:
        freq_hz = 0.0
    intervals = [
        peak_times[i] - peak_times[i - 1]
        for i in range(1, len(peak_times))
        if (peak_times[i] - peak_times[i - 1]) > 1e-6
    ]

    if len(intervals) >= 2:
        mean_interval = _safe_mean(intervals, default=0.0)
        std_interval = _safe_std(intervals, default=0.0)
        cv = std_interval / mean_interval if mean_interval > 1e-6 else 0.0
        if cv <= 0.20:
            regularidade = "Regular"
        elif cv <= 0.40:
            regularidade = "Moderado"
        else:
            regularidade = "Irregular"
    else:
        cv = 0.0
        # Retorna "-" (indefinido) em vez de "Regular" para evitar falso positivo clínico.
        # "Regular" com n_picos < 2 significa apenas ausência de dados, não boa coordenação.
        regularidade = "-"

    return {
        "amplitude": float(amplitude),
        "vel_media": float(vel_media),
        "vel_pico": float(vel_pico),
        "freq_hz": float(freq_hz),
        "cv": float(cv),
        "regularidade": regularidade,
        "n_picos": int(n_picos),
    }


# =============================================================================
# FORMATADORES DE TEXTO
# =============================================================================

def freq_label(freq_hz: float) -> str:
    """
    Formata a frequência para exibição amigável.
    """
    if freq_hz <= 0:
        return "0.00 Hz"

    period_s = 1.0 / freq_hz if freq_hz > 1e-6 else 0.0

    if freq_hz < 0.15:
        emoji = "🐢"
        status = "lento"
    elif freq_hz > 0.40:
        emoji = "⚡"
        status = "ativo"
    else:
        emoji = "🔄"
        status = "moderado"

    return f"{emoji} {freq_hz:.2f} Hz ({status}, ≈ 1 ciclo/{period_s:.2f}s)"


def regularidade_label(regularidade: str, cv: float) -> str:
    """
    Formata a regularidade temporal para exibição amigável.
    """
    if regularidade == "Regular":
        emoji = "✅"
    elif regularidade == "Moderado":
        emoji = "🟡"
    else:
        emoji = "❌"

    return f"{emoji} CV {cv:.2f} → {regularidade}"


# =============================================================================
# MÉTRICAS AGREGADAS DE SESSÃO (a partir de buffers em memória)
# =============================================================================

def compute_session_metrics_from_buffers(
    angle_buffers: Dict[str, Dict[str, List[float]]],
    time_buffers: Dict[str, List[float]],
    min_range_pct: float = 0.30,
    min_dist_s: float = 1.0,
) -> Dict[str, Dict]:
    """
    Calcula métricas consolidadas para todos os dedos a partir dos buffers
    em memória do session_state.

    Reaproveita compute_realtime_metrics() para cada dedo, agrupando os
    resultados em um único dicionário indexado por nome de dedo.

    Parâmetros:
        angle_buffers : dicionário {finger: {joint: [valores]}}
                        (snapshot já copiado do session_state)
        time_buffers  : dicionário {finger: [timestamps]}
        min_range_pct : limiar de amplitude para aceitar picos
        min_dist_s    : distância mínima entre picos (em segundos)

    Retorna:
        {
            "INDEX":  {amplitude, vel_media, vel_pico, freq_hz, cv,
                       regularidade, n_picos},
            "MIDDLE": {...},
            "RING":   {...},
            "PINKY":  {...},
        }

    Compatibilidade:
        Esta função não altera o comportamento de compute_realtime_metrics().
        O fluxo antigo por CSV não é afetado.
    """
    results: Dict[str, Dict] = {}

    for finger in FINGERS:
        tam_buffer = angle_buffers.get(finger, {}).get("TAM", [])
        t_buffer = time_buffers.get(finger, [])

        results[finger] = compute_realtime_metrics(
            angle_buffer=tam_buffer,
            time_buffer=t_buffer,
            min_range_pct=min_range_pct,
            min_dist_s=min_dist_s,
        )

    return results


def build_tam_chart_data(
    angle_buffers: Dict[str, Dict[str, List[float]]],
) -> Dict[str, List[float]]:
    """
    Monta o dicionário de séries temporais do TAM por dedo, pronto para
    alimentar st.line_chart().

    Para dedos longos usa o buffer "TAM".
    Para o polegar usa o buffer "MCP" como proxy de amplitude.

    Retorna:
        {
            "Indicador":  [float, ...],
            "Médio":      [float, ...],
            "Anelar":     [float, ...],
            "Mínimo":    [float, ...],
            "Polegar":    [float, ...],
        }
    """
    name_map = {
        "INDEX":  "Indicador",
        "MIDDLE": "Médio",
        "RING":   "Anelar",
        "PINKY":  "Mínimo",
        "THUMB":  "Polegar",
    }

    # Chave do buffer a usar por dedo (polegar não tem TAM próprio).
    buffer_key = {
        "INDEX":  "TAM",
        "MIDDLE": "TAM",
        "RING":   "TAM",
        "PINKY":  "TAM",
        "THUMB":  "TAM",
    }

    series: Dict[str, List] = {}
    for finger, label in name_map.items():
        key = buffer_key[finger]
        values = angle_buffers.get(finger, {}).get(key, [])
        series[label] = list(values)

    # Alinha as séries para o comprimento máximo, preenchendo as mais curtas
    # com None (tratado como NaN pelo Vega-Lite = gap visível no gráfico).
    # Isso evita que um dedo com 1 frame a menos trunque todos os demais.
    max_len = max((len(v) for v in series.values()), default=0)
    if max_len > 0:
        for label in series:
            gap = max_len - len(series[label])
            if gap > 0:
                series[label] = [None] * gap + series[label]

    return series