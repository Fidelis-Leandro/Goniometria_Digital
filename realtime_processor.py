"""
realtime_processor.py — Processador de vídeo em tempo real para goniometria
============================================================================

Este módulo isola a classe VideoProcessor do app.py, mantendo toda a lógica
de processamento de frames longe do código de renderização Streamlit.

Responsabilidades:
- receber cada frame WebRTC via recv();
- executar MediaPipe Hands;
- calcular ângulos brutos com DigitalGoniometer;
- suavizar com EMA → Kalman (GoniometryFilterBank);
- desenhar overlay clínico goniométrico APENAS no painel de esqueleto;
- atualizar buffers circulares de forma thread-safe;
- registrar a sessão em CSV.

Nota de threading:
    O método recv() é chamado pela thread interna do aiortc/streamlit-webrtc,
    não pela thread principal do Streamlit.
    Por isso:
    - nunca chamamos widgets (st.write, st.metric, etc.) aqui;
    - nunca chamamos st.rerun() aqui;
    - toda escrita em buffers internos é protegida por threading.Lock.
    - leituras de configuração (parâmetros) são feitas em __init__, na thread
      principal, antes de o recv() iniciar.

Otimizações de latência:
    - Overlay desenhado apenas a cada OVERLAY_FRAME_INTERVAL frames.
    - Frame sem mão retornado imediatamente (sem overlay).
    - resize para MP_PROCESS_WIDTH×MP_PROCESS_HEIGHT antes do MediaPipe.
    - Estado do Kalman preservado entre frames sem detecção.
    - Lock mínimo: apenas para atualizar os campos de leitura do Streamlit.
"""

from __future__ import annotations

import time
import threading
from collections import deque
from typing import Optional

import av
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError(
        "MediaPipe não encontrado. Execute: pip install mediapipe"
    ) from e

from goniometry import DigitalGoniometer
from smoothing import GoniometryFilterBank
from goniometry_overlay import _build_skeleton
from goniometry_csv import GoniometryCSVLogger
from dashboard_utils import FINGERS, FINGER_JOINTS

# =============================================================================
# CONSTANTES DO PROCESSADOR
# =============================================================================

# Dimensões do painel de overlay gerado pelo goniômetro.
PANEL_W: int = 640
PANEL_H: int = 520

# Parâmetros do pipeline de suavização EMA → Kalman.
EMA_ALPHA: float = 0.30
KALMAN_Q: float = 0.01
KALMAN_R: float = 0.10

# Resolução reduzida para processamento do MediaPipe.
# Landmarks são normalizados (0.0–1.0), independem da resolução de entrada.
# 240×180 é suficiente para detecção confiável e reduz custo de inferência.
MP_PROCESS_WIDTH: int = 240
MP_PROCESS_HEIGHT: int = 180

# Confiança do MediaPipe (valores otimizados para performance).
# 0.60 detect: margem maior para detecção inicial sem perder qualidade.
# 0.40 track: mantém tracking fluido sem re-detectar a cada frame.
MP_DETECT_CONF: float = 0.60
MP_TRACK_CONF: float = 0.40

# Overlay pesado (_build_skeleton) só é gerado a cada N frames.
# Nos frames intermediários, o último overlay cacheado é reutilizado.
# Valor 6 = overlay em ~17% dos frames → ~83% do custo de renderização eliminado.
OVERLAY_FRAME_INTERVAL: int = 6

# Intervalo de gravação no CSV (a cada N frames com mão detectada).
CSV_LOG_INTERVAL: int = 3

# Frames consecutivos sem mão após os quais o filtro Kalman é resetado.
# A 30fps, 15 frames ≈ 500ms — evita resets desnecessarios por ocluções momentaneas.
NO_HAND_RESET_FRAMES: int = 15

# Intervalo do pipeline completo (MediaPipe + cálculo de ângulos).
# Em frames intermediários, retorna resultado cacheado imediatamente.
# Valor 2 = processa 50% dos frames → ~50% de economia de CPU.
MP_PROCESS_INTERVAL: int = 2

# =============================================================================
# PROCESSADOR DE VÍDEO
# =============================================================================

class VideoProcessor:
    """
    Processador de frames do streamlit-webrtc.

    Ciclo de vida:
    1. __init__() é chamado na thread principal do Streamlit quando o
       componente webrtc_streamer é montado.
    2. recv() é chamado repetidamente pela thread do aiortc para cada frame
       recebido do navegador.
    3. Quando o usuário clica em STOP, a thread do aiortc é encerrada e
       o objeto é descartado.

    Cada instância mantém:
    - um detector MediaPipe Hands próprio (não thread-safe entre instâncias);
    - um DigitalGoniometer;
    - um banco de filtros EMA + Kalman com estado persistente por série;
    - um contador local de frames;
    - o último frame de overlay renderizado (reutilizado nos frames intermediários).
    """

    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=MP_DETECT_CONF,
            min_tracking_confidence=MP_TRACK_CONF,
        )

        self.gonio = DigitalGoniometer()
        self.filter_bank = GoniometryFilterBank(
            ema_alpha=EMA_ALPHA,
            kalman_q=KALMAN_Q,
            kalman_r=KALMAN_R,
        )

        self._local_frame_id: int = 0

        # Thread-safe lock — protege apenas os campos lidos pelo Streamlit.
        self.lock = threading.Lock()

        # 60 frames ≈ 2 segundos de janela deslizante a 30fps.
        buffer_size = 10
        # Buffers inicializados apenas com os joints válidos por dedo.
        # THUMB tem apenas MCP e IP (anatomia diferente dos demais).
        self.angle_buffers = {
            finger: {
                joint: deque(maxlen=buffer_size)
                for joint in FINGER_JOINTS[finger]
            }
            for finger in FINGERS
        }

        self.time_buffers = {
            finger: deque(maxlen=buffer_size)
            for finger in FINGERS
        }

        self.last_angles: dict = {}
        self.hand_detected: bool = False
        self.frame_id: int = 0
        # Contador de frames consecutivos sem detecção de mão.
        # Ao atingir NO_HAND_RESET_FRAMES, o filtro é resetado para evitar
        # transientes falsos quando a mão retorna em posição diferente.
        self._no_hand_frames: int = 0

        # Último overlay renderizado — reutilizado nos frames sem novo overlay.
        self._last_overlay: Optional[np.ndarray] = None

        # Session CSV Logger (gerenciado inteiramente dentro do processador).
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"session_goniometry_{ts}.csv"
        self.csv_logger = GoniometryCSVLogger(self.csv_path)
        self.csv_closed = False

    # -------------------------------------------------------------------------
    # PROCESSAMENTO PRINCIPAL
    # -------------------------------------------------------------------------

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Processa um frame vindo do navegador via WebRTC.

        Pipeline otimizado:
        1. Converte o frame WebRTC para ndarray BGR.
        2. Espelha horizontalmente.
        3. Redimensiona para 320×240 e executa MediaPipe.
        4. Se nenhuma mão: atualiza estado e retorna frame imediatamente.
        5. Calcula ângulos + filtros.
        6. Atualiza buffers (thread-safe, lock mínimo).
        7. A cada OVERLAY_FRAME_INTERVAL frames: gera novo overlay.
           Nos demais frames: retorna o último overlay cacheado.
        8. Grava no CSV a cada CSV_LOG_INTERVAL frames.
        """
        self._local_frame_id += 1

        # ── Fast path: pula pipeline completo em frames alternados ────────
        # Retorna overlay cacheado sem executar MediaPipe nem cálculos.
        # Reduz carga de CPU pela metade sem impacto visual perceptível.
        if self._local_frame_id % MP_PROCESS_INTERVAL != 0:
            if self._last_overlay is not None:
                return av.VideoFrame.from_ndarray(self._last_overlay, format="bgr24")

        img_bgr = frame.to_ndarray(format="bgr24")

        # Espelhamento horizontal — experiência de "espelho" para o usuário.
        img_bgr = cv2.flip(img_bgr, 1)

        # ── MediaPipe em resolução reduzida ──────────────────────────────────
        img_small = cv2.resize(
            img_bgr, (MP_PROCESS_WIDTH, MP_PROCESS_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = self.mp_hands.process(img_rgb)
        img_rgb.flags.writeable = True

        hand_detected = results.multi_hand_landmarks is not None
        handedness_list = results.multi_handedness

        if not hand_detected:
            self._update_state_no_hand()
            # Retorna último overlay se disponível, ou frame bruto.
            if self._last_overlay is not None:
                return av.VideoFrame.from_ndarray(self._last_overlay, format="bgr24")
            return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

        # ── Detecta handedness (“Left” / “Right”) ────────────────────────────
        # O flip BGR não afeta os landmarks 3D do MediaPipe, então a mão
        # esquerda real continua reportada como “Left” mesmo após o espelho.
        # Passamos essa informação ao goniômetro para corrigir o sinal da normal.
        is_right_hand = True
        if handedness_list:
            label = handedness_list[0].classification[0].label
            is_right_hand = (label == "Right")

        # ── Pipeline goniométrico ────────────────────────────────────────────
        landmarks = results.multi_hand_landmarks[0].landmark
        angles_raw = self.gonio.compute_all(landmarks, is_right_hand=is_right_hand)
        angles_smooth = self.filter_bank.smooth_all(angles_raw)

        # ── Atualização de estado (thread-safe) ──────────────────────────────
        now = time.monotonic()
        self._update_state_with_angles(angles_smooth, now)

        # ── CSV: apenas a cada CSV_LOG_INTERVAL frames ───────────────────────
        if self._local_frame_id % CSV_LOG_INTERVAL == 0:
            self._log_to_csv(angles_smooth)

        # ── Overlay: apenas a cada OVERLAY_FRAME_INTERVAL frames ─────────────
        if self._local_frame_id % OVERLAY_FRAME_INTERVAL == 0:
            stability_map = {
                finger: {
                    joint: self.filter_bank.get_stability(finger, joint)
                    for joint in joints_dict.keys()
                }
                for finger, joints_dict in angles_smooth.items()
            }

            # Chama _build_skeleton diretamente em vez de draw_goniometry_overlay.
            # O painel de dados (_build_data_panel) era descartado (variável _),
            # mas custava ~50% do tempo total de overlay.
            skeleton_panel = _build_skeleton(
                img_bgr,
                landmarks,
                angles_smooth,
                pw=480,
                ph=390,
                frozen=False,
                stability_map=stability_map,
            )
            # Escala o painel de volta para PANEL_W × PANEL_H com interpolação rápida.
            if skeleton_panel.shape[:2] != (PANEL_H, PANEL_W):
                skeleton_panel = cv2.resize(
                    skeleton_panel, (PANEL_W, PANEL_H),
                    interpolation=cv2.INTER_LINEAR,
                )
            self._last_overlay = skeleton_panel

        # Retorna o overlay (novo ou cacheado do frame anterior).
        if self._last_overlay is not None:
            return av.VideoFrame.from_ndarray(self._last_overlay, format="bgr24")
        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

    # -------------------------------------------------------------------------
    # ATUALIZAÇÃO DE ESTADO (thread-safe)
    # -------------------------------------------------------------------------

    def _update_state_no_hand(self) -> None:
        """
        Atualiza o estado indicando que nenhuma mão foi detectada neste frame.

        Após NO_HAND_RESET_FRAMES frames consecutivos sem mão, os filtros
        Kalman são resetados para evitar transientes falsos quando a mão
        retorna em posição diferente da última armazenada no estado do filtro.
        """
        self._no_hand_frames += 1
        with self.lock:
            self.hand_detected = False
            self.frame_id = self._local_frame_id

        if self._no_hand_frames >= NO_HAND_RESET_FRAMES:
            self.filter_bank.reset_all()
            self._no_hand_frames = 0

    def _update_state_with_angles(
        self,
        angles_smooth: dict,
        timestamp: float,
    ) -> None:
        """
        Grava ângulos suavizados e timestamp nos buffers circulares locais.
        Lock mantido pelo menor tempo possível — apenas a escrita nos deques.
        """
        self._no_hand_frames = 0   # reset contador ao detectar mão
        with self.lock:
            for finger in FINGERS:
                if finger not in angles_smooth:
                    continue

                finger_metrics = angles_smooth[finger]
                # Itera apenas sobre os joints válidos do dedo.
                # Evita gravar 0.0 nos buffers do THUMB para joints inexistentes
                # (PIP, DIP, ABD, TAM), que distorcem métricas e o CSV.
                for joint in FINGER_JOINTS[finger]:
                    value = finger_metrics.get(joint, 0.0)
                    self.angle_buffers[finger][joint].append(value)

                self.time_buffers[finger].append(timestamp)

            self.last_angles = angles_smooth
            self.hand_detected = True
            self.frame_id = self._local_frame_id

    def _log_to_csv(self, angles_smooth: dict) -> None:
        """
        Grava o frame atual no CSV da sessão.
        Faz flush a cada 60 frames para reduzir I/O no loop principal.

        Usa o mesmo lock de close_csv() para evitar race condition:
        log() e close() não podem correr simultaneamente.
        """
        with self.lock:
            if self.csv_logger and not self.csv_closed:
                self.csv_logger.log(self._local_frame_id, angles_smooth)
                if self._local_frame_id % 60 == 0:
                    self.csv_logger.flush()

    def close_csv(self) -> None:
        """
        Salva e fecha o logger CSV com segurança.
        """
        with self.lock:
            if not self.csv_closed and self.csv_logger:
                self.csv_logger.flush()
                self.csv_logger.close()
                self.csv_closed = True
