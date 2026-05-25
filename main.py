"""
main.py — Orquestrador Unificado com Pipeline EMA→Kalman + Dashboard Automático
================================================================================
Ponto de entrada ÚNICO. Um comando inicia todo o sistema:

    python main.py

Ao pressionar Q ou ESC:
  1. Câmera e MediaPipe são liberados
  2. CSV é fechado com flush garantido
  3. last_session.json é gravado com o caminho do CSV
  4. Streamlit é iniciado automaticamente: http://localhost:8501

Controles:
  Q / ESC → encerrar + abre dashboard
  S       → congelar frame
  R       → retomar

Dependências:
  pip install mediapipe opencv-python numpy streamlit plotly pandas
"""

import sys
import json
import os
import queue
import threading
import time
import argparse
from datetime import datetime

import cv2
import numpy as np

try:
    import mediapipe as mp
    from goniometry        import DigitalGoniometer
    from smoothing         import GoniometryFilterBank
    from goniometry_overlay import draw_goniometry_overlay, compose_side_by_side
    from goniometry_csv    import GoniometryCSVLogger
except ImportError as e:
    print(f"\n[ERRO] Módulo não encontrado: {e}")
    print("Dependências: pip install mediapipe opencv-python numpy")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES PADRÃO
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CAM_INDEX   = 0
DEFAULT_PANEL_W     = 640
DEFAULT_PANEL_H     = 520
DEFAULT_CSV_FILE    = "session_goniometry.csv"
DEFAULT_EMA_ALPHA   = 0.30
DEFAULT_KALMAN_Q    = 0.01
DEFAULT_KALMAN_R    = 0.10
MP_DETECT_CONF      = 0.70
MP_TRACK_CONF       = 0.50


# ═══════════════════════════════════════════════════════════════════════════════
# THREAD-1: CAPTURA DE FRAMES
# ═══════════════════════════════════════════════════════════════════════════════

class CaptureThread(threading.Thread):
    """
    Thread dedicada à captura de frames da webcam.
    Estratégia Queue(maxsize=1): mantém sempre o frame mais recente,
    descartando frames antigos quando o processamento está atrasado.
    """

    def __init__(self, cam_index: int, frame_queue: queue.Queue,
                 stop_event: threading.Event):
        super().__init__(daemon=True, name="CaptureThread")
        self.cam_index       = cam_index
        self.queue           = frame_queue
        self.stop_event      = stop_event
        self._frames_dropped = 0

    def run(self) -> None:
        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            print(f"[CaptureThread] ERRO: câmera {self.cam_index} não disponível.")
            self.stop_event.set()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS,          30)
        print(f"[CaptureThread] Câmera {self.cam_index} iniciada.")

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            frame = cv2.flip(frame, 1)

            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
                self.queue.put_nowait(frame)
                self._frames_dropped += 1

        cap.release()
        print(f"[CaptureThread] Encerrada. Frames descartados: {self._frames_dropped}")


# ═══════════════════════════════════════════════════════════════════════════════
# SISTEMA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

class GoniometrySystem:
    """
    Sistema integrado de goniometria digital com pipeline EMA→Kalman obrigatório.

    Fluxo por frame:
      cap.read() [Thread-1] → Queue(1) → MediaPipe → DigitalGoniometer
      → GoniometryFilterBank (EMA→Kalman) → draw_overlay → imshow + CSV
    """

    def __init__(self, cam_index, panel_w, panel_h, csv_file,
                 ema_alpha, kalman_q, kalman_r, side_by_side):
        print("[Sistema] Inicializando pipeline EMA→Kalman...")

        self.frame_queue    = queue.Queue(maxsize=1)
        self.stop_event     = threading.Event()
        self.capture_thread = CaptureThread(cam_index, self.frame_queue, self.stop_event)

        mp_hands    = mp.solutions.hands
        self.hands  = mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=MP_DETECT_CONF,
            min_tracking_confidence=MP_TRACK_CONF,
        )
        print("  ✓ MediaPipe Hands")

        self.gonio       = DigitalGoniometer()
        self.filter_bank = GoniometryFilterBank(
            ema_alpha=ema_alpha, kalman_q=kalman_q, kalman_r=kalman_r)
        print(f"  ✓ Filtro EMA(α={ema_alpha})+Kalman(Q={kalman_q},R={kalman_r})")

        self.logger = GoniometryCSVLogger(csv_file)
        print(f"  ✓ CSV: {csv_file}")

        self.panel_w      = panel_w
        self.panel_h      = panel_h
        self.side_by_side = side_by_side
        self.running      = True
        self.frozen       = False
        self.frame_id     = 0

        self._cached_skeleton: np.ndarray = None
        self._cached_data:     np.ndarray = None
        self._last_angles:     dict       = {}

        self._fps_t0  = time.time()
        self._fps_cnt = 0
        self._fps_val = 0.0

        print("[Sistema] Pronto. Q=sair+dashboard | S=congelar | R=retomar\n")

    def _process_frame(self, frame: np.ndarray) -> tuple:
        """
        Pipeline completo:
          BGR→RGB → MediaPipe → angles_raw → EMA→Kalman → overlay → CSV
        Nenhum ângulo bruto é exibido ou salvo.
        """
        self.frame_id += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            ph = np.full((self.panel_h, self.panel_w, 3), (26, 26, 26), np.uint8)
            return ph, ph, {}

        landmarks    = results.multi_hand_landmarks[0].landmark
        angles_raw   = self.gonio.compute_all(landmarks)
        angles_smooth = self.filter_bank.smooth_all(angles_raw)

        stability_map = {
            finger: {joint: self.filter_bank.get_stability(finger, joint)
                     for joint in metrics}
            for finger, metrics in angles_smooth.items()
        }

        skeleton, data = draw_goniometry_overlay(
            frame, landmarks, angles_smooth,
            panel_w=self.panel_w, panel_h=self.panel_h,
            frozen=self.frozen, stability_map=stability_map,
        )

        self.logger.log(self.frame_id, angles_smooth)
        if self.frame_id % 30 == 0:
            self.logger.flush()

        self._cached_skeleton = skeleton.copy()
        self._cached_data     = data.copy()
        self._last_angles     = angles_smooth

        return skeleton, data, angles_smooth

    def _calc_fps(self) -> float:
        self._fps_cnt += 1
        if self._fps_cnt >= 30:
            self._fps_val = self._fps_cnt / (time.time() - self._fps_t0)
            self._fps_cnt = 0
            self._fps_t0  = time.time()
        return self._fps_val

    def _draw_fps(self, panel: np.ndarray) -> np.ndarray:
        fps = self._calc_fps()
        txt = f"FPS:{fps:.1f}  Frame:{self.frame_id}"
        col = (80, 220, 80) if fps >= 25 else (40, 180, 255)
        cv2.putText(panel, txt, (self.panel_w - 215, 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38, col, 1, cv2.LINE_AA)
        return panel

    def _handle_key(self, key: int) -> None:
        if key in (ord('q'), 27):
            print("[Sistema] Encerrando...")
            self.running = False
            self.stop_event.set()
        elif key == ord('s') and not self.frozen:
            self.frozen = True
            print(f"[Sistema] Frame {self.frame_id} congelado.")
        elif key == ord('r') and self.frozen:
            self.frozen = False
            self.filter_bank.reset_all(seed_angles=self._last_angles)
            print("[Sistema] Retomado.")

    def run(self) -> None:
        ph = np.full((self.panel_h, self.panel_w, 3), (26, 26, 26), np.uint8)
        cv2.putText(ph, "Posicione a mao na frente da camera",
                    (30, self.panel_h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, (100, 100, 100), 1)

        self.capture_thread.start()
        skel_panel = data_panel = ph.copy()

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                continue

            if self.frozen:
                if self._cached_skeleton is not None:
                    skel_panel = self._cached_skeleton
                    data_panel = self._cached_data
            else:
                skel_panel, data_panel, _ = self._process_frame(frame)

            skel_panel = self._draw_fps(skel_panel)

            if self.side_by_side:
                cv2.imshow("Goniometria Digital",
                           compose_side_by_side(skel_panel, data_panel))
            else:
                cv2.imshow("Hand Skeleton View",    skel_panel)
                cv2.imshow("Goniometry Data Panel", data_panel)

            self._handle_key(cv2.waitKey(1) & 0xFF)

    def shutdown(self) -> None:
        """Liberação ordenada de todos os recursos."""
        print("[Sistema] Liberando recursos...")
        self.stop_event.set()
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        self.hands.close()
        self.logger.close()   # flush + close garantidos
        cv2.destroyAllWindows()
        print(f"[Sistema] Frames processados: {self.frame_id}")


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD — ABERTURA AUTOMÁTICA APÓS ENCERRAMENTO
# ═══════════════════════════════════════════════════════════════════════════════

def _launch_dashboard(csv_path: str) -> None:
    """
    Abre o dashboard Streamlit automaticamente após o encerramento da câmera.

    Passos executados:
      1. Grava last_session.json com o caminho do CSV
         → app.py lê este arquivo para encontrar os dados
      2. Executa `python -m streamlit run app.py` via subprocess.Popen
         → não bloqueia o terminal enquanto o dashboard carrega
      3. Aguarda o processo (mantém o terminal aberto)
         → Ctrl+C encerra o dashboard

    Para ambientes sem navegador gráfico (headless):
      Altere "--server.headless" para "true" no Popen abaixo.
      O dashboard ainda funciona; apenas não abre o navegador automaticamente.
    """
    # Grava referência para o app.py encontrar o CSV correto
    config = {
        "last_csv":  str(csv_path),
        "timestamp": datetime.now().isoformat(),
    }
    try:
        with open("last_session.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"[Dashboard] Referência salva em last_session.json → {csv_path}")
    except OSError as e:
        print(f"[Dashboard] Aviso: não foi possível salvar last_session.json: {e}")

    print("\n" + "═" * 60)
    print("  SESSÃO ENCERRADA — Abrindo análise no navegador...")
    print(f"  CSV salvo:  {csv_path}")
    print("  URL:        http://localhost:8501")
    print("  Encerrar:   Ctrl+C neste terminal")
    print("═" * 60 + "\n")

    import subprocess
    try:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run", "app.py",
                "--server.headless",        "false",
                "--server.port",            "8501",
                "--server.address",         "localhost",
                "--browser.gatherUsageStats", "false",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[Dashboard] Streamlit iniciado (PID {proc.pid}).")
        print("[Dashboard] Aguardando... pressione Ctrl+C para encerrar o dashboard.\n")
        try:
            proc.wait()   # mantém o processo ativo
        except KeyboardInterrupt:
            proc.terminate()
            print("\n[Dashboard] Dashboard encerrado pelo usuário.")

    except FileNotFoundError:
        print("[Dashboard] ERRO: Streamlit não encontrado.")
        print("[Dashboard] Instale com: pip install streamlit plotly pandas")
        print(f"[Dashboard] Abra manualmente: streamlit run app.py -- --csv {csv_path}")

    except Exception as e:
        print(f"[Dashboard] Erro inesperado: {e}")
        print(f"[Dashboard] Abra manualmente: streamlit run app.py -- --csv {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ARGUMENTOS CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Goniometria Digital — Pipeline EMA→Kalman + Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python main.py                        # configuração padrão
  python main.py --cam 1                # câmera 1
  python main.py --size 800 600         # painéis maiores
  python main.py --kq 0.05 --kr 0.20   # Kalman mais responsivo
  python main.py --no-dashboard         # sem dashboard ao encerrar
  python main.py --no-log               # sem CSV
        """)
    p.add_argument("--cam",          type=int,   default=DEFAULT_CAM_INDEX)
    p.add_argument("--size",         type=int,   nargs=2,
                   metavar=("W","H"), default=[DEFAULT_PANEL_W, DEFAULT_PANEL_H])
    p.add_argument("--csv",          type=str,   default=DEFAULT_CSV_FILE)
    p.add_argument("--alpha",        type=float, default=DEFAULT_EMA_ALPHA,
                   help="Coeficiente EMA (default 0.30)")
    p.add_argument("--kq",           type=float, default=DEFAULT_KALMAN_Q,
                   help="Ruído de processo Kalman Q (default 0.01)")
    p.add_argument("--kr",           type=float, default=DEFAULT_KALMAN_R,
                   help="Ruído de medição Kalman R (default 0.10)")
    p.add_argument("--side-by-side",  action="store_true")
    p.add_argument("--no-log",        action="store_true",
                   help="Desabilitar logging CSV")
    p.add_argument("--no-dashboard",  action="store_true",
                   help="Não abrir dashboard ao encerrar")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    print("\n" + "═" * 60)
    print("  GONIOMETRIA DIGITAL — Pipeline EMA→Kalman")
    print("═" * 60)
    print(f"  Câmera:       {args.cam}")
    print(f"  Painéis:      {args.size[0]}×{args.size[1]} px")
    print(f"  EMA alpha:    {args.alpha}")
    print(f"  Kalman Q/R:   {args.kq} / {args.kr}")
    print(f"  CSV:          {'desativado' if args.no_log else args.csv}")
    print(f"  Dashboard:    {'desativado' if args.no_dashboard else 'abre ao encerrar'}")
    print("═" * 60 + "\n")

    csv_file = os.devnull if args.no_log else args.csv

    system = None
    try:
        system = GoniometrySystem(
            cam_index    = args.cam,
            panel_w      = args.size[0],
            panel_h      = args.size[1],
            csv_file     = csv_file,
            ema_alpha    = args.alpha,
            kalman_q     = args.kq,
            kalman_r     = args.kr,
            side_by_side = args.side_by_side,
        )
        system.run()

    except KeyboardInterrupt:
        print("\n[Sistema] Interrompido (Ctrl+C).")
    except RuntimeError as e:
        print(f"\n[ERRO] {e}")
        sys.exit(1)
    finally:
        # Garante liberação de recursos SEMPRE (inclusive em erros)
        if system:
            system.shutdown()

    # Abre o dashboard APÓS o shutdown completo (CSV já fechado com flush)
    if not args.no_log and not args.no_dashboard and os.path.isfile(args.csv):
        _launch_dashboard(args.csv)
    elif not args.no_dashboard and args.no_log:
        print("[Dashboard] Dashboard desativado (--no-log: sem CSV para analisar).")


if __name__ == "__main__":
    main()