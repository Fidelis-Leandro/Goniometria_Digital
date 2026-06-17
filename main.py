"""
main.py — Fallback OpenCV para uso sem Streamlit
=================================================

Este arquivo oferece uma alternativa ao app.py para situações onde
o Streamlit ou o WebRTC não estão disponíveis, como:
- ambientes sem navegador;
- depuração de câmera local;
- validação isolada do pipeline de goniometria.

Fluxo de dados:
cv2.VideoCapture
→ frame BGR
→ MediaPipe Hands
→ DigitalGoniometer.compute_all()
→ GoniometryFilterBank.smooth_all()
→ draw_goniometry_overlay()
→ cv2.imshow() (duas janelas: esqueleto + dados)
→ GoniometryCSVLogger.log()

Controles:
S : salvar frame atual como PNG
Q / ESC : encerrar
"""

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    print("ERRO: MediaPipe não encontrado. Execute: pip install mediapipe")
    sys.exit(1)

from goniometry import DigitalGoniometer
from smoothing import GoniometryFilterBank
from goniometry_overlay import draw_goniometry_overlay, compose_side_by_side
from goniometry_csv import GoniometryCSVLogger

# =============================================================================
# CONFIGURAÇÕES DO MODO FALLBACK
# =============================================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
TARGET_FPS = 30

EMA_ALPHA = 0.30
KALMAN_Q = 0.01
KALMAN_R = 0.10

MP_DETECT_CONF = 0.70
MP_TRACK_CONF = 0.50

PANEL_W = 640
PANEL_H = 520

WINDOW_MAIN = "Goniometria — Esqueleto"
WINDOW_DATA = "Goniometria — Dados Clínicos"

SAVE_DIR = Path("frames_salvos")


# =============================================================================
# INICIALIZAÇÃO DA CÂMERA
# =============================================================================

def _open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """
    Abre e configura a câmera com as dimensões desejadas.

    O backend CAP_DSHOW é preferido no Windows por ter menor latência.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Câmera {index} não pôde ser aberta. "
            "Verifique se está conectada e não está em uso por outro programa."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


# =============================================================================
# INICIALIZAÇÃO DO MEDIAPIPE
# =============================================================================

def _build_hands() -> mp.solutions.hands.Hands:
    """
    Cria o detector de mãos do MediaPipe configurado para tempo real.

    max_num_hands=1 é suficiente para goniometria clínica de uma mão por vez.
    """
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=MP_DETECT_CONF,
        min_tracking_confidence=MP_TRACK_CONF,
    )


# =============================================================================
# OVERLAYS DE STATUS
# =============================================================================

def _draw_fps(frame: np.ndarray, fps: float) -> None:
    """
    Sobrepõe o FPS atual no canto superior direito do frame.
    """
    text = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.50, 1)
    x = frame.shape[1] - tw - 10
    cv2.putText(frame, text, (x, 24), cv2.FONT_HERSHEY_DUPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)


def _draw_waiting(panel: np.ndarray) -> None:
    """
    Exibe mensagem de aguardo quando nenhuma mão é detectada.
    """
    h, w = panel.shape[:2]
    text = "Aguardando mao..."
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.60, 1)
    cv2.putText(
        panel,
        text,
        ((w - tw) // 2, h // 2 + 12),
        cv2.FONT_HERSHEY_DUPLEX,
        0.60,
        (100, 100, 100),
        1,
        cv2.LINE_AA,
    )


def _draw_controls(panel: np.ndarray) -> None:
    """
    Exibe os controles disponíveis no rodapé do painel de dados.
    """
    h, w = panel.shape[:2]
    lines = ["[S] Salvar frame PNG", "[Q] / [ESC] Encerrar"]
    for i, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (10, h - 26 + i * 14),
            cv2.FONT_HERSHEY_DUPLEX,
            0.32,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )


# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

def run() -> None:
    """
    Executa o loop de captura e processamento em tempo real.

    O loop:
    1. Captura um frame da câmera;
    2. Detecta landmarks com MediaPipe;
    3. Calcula ângulos com DigitalGoniometer;
    4. Suaviza com EMA → Kalman;
    5. Gera overlay goniométrico;
    6. Exibe em duas janelas (esqueleto + dados);
    7. Registra no CSV;
    8. Responde a eventos de teclado.
    """
    print("=" * 60)
    print("  Goniometria Digital — Modo Fallback OpenCV")
    print("=" * 60)
    print(f"  Câmera: índice {CAMERA_INDEX}")
    print(f"  Resolução: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {TARGET_FPS} FPS")
    print(f"  Filtro: EMA(α={EMA_ALPHA}) → Kalman(Q={KALMAN_Q}, R={KALMAN_R})")
    print()
    print("  Controles:")
    print("  [S]    Salvar frame PNG")
    print("  [Q/ESC] Encerrar")
    print("=" * 60)

    # Inicializa câmera.
    try:
        cap = _open_camera(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
    except RuntimeError as e:
        print(f"\nERRO: {e}")
        sys.exit(1)

    # Inicializa componentes do pipeline.
    hands = _build_hands()
    gonio = DigitalGoniometer()
    filter_bank = GoniometryFilterBank(
        ema_alpha=EMA_ALPHA,
        kalman_q=KALMAN_Q,
        kalman_r=KALMAN_R,
    )

    # Logger CSV com timestamp no nome.
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = f"session_fallback_{ts}.csv"
    logger = GoniometryCSVLogger(csv_path)
    print(f"\n  Sessão CSV: {csv_path}")
    print("  Iniciando captura...\n")

    # Janelas OpenCV.
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_DATA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_MAIN, PANEL_W, PANEL_H)
    cv2.resizeWindow(WINDOW_DATA, PANEL_W, PANEL_H)

    # Painéis padrão exibidos antes da primeira detecção.
    blank_main = np.full((PANEL_H, PANEL_W, 3), (26, 26, 26), dtype=np.uint8)
    blank_data = blank_main.copy()
    _draw_waiting(blank_main)
    _draw_controls(blank_data)

    # Estado do loop.
    frame_id = 0
    t_prev = time.perf_counter()
    fps_display = 0.0
    last_skel: Optional[np.ndarray] = blank_main.copy()
    last_data: Optional[np.ndarray] = blank_data.copy()

    # Painel vazio de boas-vindas.
    cv2.imshow(WINDOW_MAIN, blank_main)
    cv2.imshow(WINDOW_DATA, blank_data)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Aviso: frame vazio recebido da câmera.")
                continue

            frame_id += 1

            # Cálculo de FPS com janela de suavização simples.
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps_display = fps_display * 0.85 + (1.0 / dt) * 0.15

            # Espelhamento horizontal para experiência mais natural.
            frame = cv2.flip(frame, 1)

            # MediaPipe Hands requer entrada em RGB.
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            results = hands.process(img_rgb)
            img_rgb.flags.writeable = True

            hand_detected = results.multi_hand_landmarks is not None

            if hand_detected:
                landmarks = results.multi_hand_landmarks[0].landmark

                # Goniometria bruta e suavização EMA → Kalman.
                angles_raw = gonio.compute_all(landmarks)
                angles_smooth = filter_bank.smooth_all(angles_raw)

                # Mapa de estabilidade para colorir o overlay.
                stability_map = {
                    finger: {
                        joint: filter_bank.get_stability(finger, joint)
                        for joint in metrics.keys()
                    }
                    for finger, metrics in angles_smooth.items()
                }

                # Geração dos painéis visuais.
                skel_panel, data_panel = draw_goniometry_overlay(
                    frame,
                    landmarks,
                    angles_smooth,
                    panel_w=PANEL_W,
                    panel_h=PANEL_H,
                    frozen=False,
                    stability_map=stability_map,
                )

                # Logging CSV.
                logger.log(frame_id, angles_smooth)
                if frame_id % 30 == 0:
                    logger.flush()

                last_skel = skel_panel
                last_data = data_panel

            else:
                # Nenhuma mão: mostra o último painel com sobreposição de aviso.
                last_skel = last_skel.copy() if last_skel is not None else blank_main.copy()
                last_data = last_data.copy() if last_data is not None else blank_data.copy()
                _draw_waiting(last_skel)

            # Indicador de FPS.
            _draw_fps(last_skel, fps_display)

            # Exibição.
            cv2.imshow(WINDOW_MAIN, last_skel)
            cv2.imshow(WINDOW_DATA, last_data)

            # Tratamento de teclas.
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                print("\nEncerrando sessão...")
                break

            elif key in (ord("s"), ord("S")):
                SAVE_DIR.mkdir(exist_ok=True)
                save_ts = time.strftime("%Y%m%d_%H%M%S")
                combined = compose_side_by_side(last_skel, last_data)
                save_path = SAVE_DIR / f"frame_{save_ts}.png"
                cv2.imwrite(str(save_path), combined)
                print(f"  Frame salvo: {save_path}")

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")

    finally:
        # Fechamento limpo de recursos.
        logger.flush()
        logger.close()
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

        print(f"\n  Sessão encerrada. CSV salvo em: {csv_path}")
        print(f"  Total de frames processados: {frame_id}")


# =============================================================================
# PONTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    run()