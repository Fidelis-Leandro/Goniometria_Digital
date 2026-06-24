"""
workers/processing_worker.py — Thread de processamento goniométrico em tempo real
==================================================================================

Este módulo implementa o núcleo científico da aplicação PyQt6: recebe frames
brutos da câmera, executa o pipeline completo de análise e entrega os resultados
prontos para a interface gráfica, sem jamais bloquear a thread principal.

Problema que este módulo resolve:
    O pipeline MediaPipe + cálculo de ângulos + suavização por filtros é pesado:
    pode levar de 15ms a 50ms por frame, dependendo do hardware. Se rodasse na
    thread principal, a janela PyQt6 ficaria irresponsiva durante cada análise.

Solução:
    ProcessingWorker roda em sua própria QThread. Recebe frames do CameraWorker
    via Queue(maxsize=1) e entrega resultados à MainWindow via pyqtSignal —
    nunca tocando em nenhum widget diretamente.

Pipeline de dados por frame:
    frame_bgr (np.ndarray)
        → MediaPipe Hands                    [detecção de landmarks 3D]
        → DigitalGoniometer.compute_all()    [ângulos brutos por articulação]
        → GoniometryFilterBank.smooth_all()  [EMA → Kalman, remove tremidos]
        → _build_skeleton()                  [overlay visual BGR]
        → classify_hand_state()              [mão aberta/fechada, ASSH]
        → compute_realtime_metrics()         [velocidade, frequência, regularidade]
        → ProcessingResult                   [dataclass com tudo empacotado]
        → pyqtSignal result_ready            [entrega thread-safe à MainWindow]

Regras respeitadas:
    - NUNCA chamamos widgets aqui (violação causa crashes silenciosos no Qt).
    - Pipeline científico (goniometry.py, smoothing.py etc.) nunca é modificado.
    - Todos os parâmetros numéricos vêm de config.py.
"""

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

# Módulos científicos do projeto — importados mas NUNCA modificados.
from goniometry import DigitalGoniometer
from smoothing import GoniometryFilterBank
from goniometry_overlay import _build_skeleton
from goniometry_csv import GoniometryCSVLogger
from dashboard_utils import (
    FINGERS,
    FINGER_JOINTS,
    classify_hand_state,
    compute_realtime_metrics,
)

import config

try:
    import mediapipe as mp
except ImportError:
    raise ImportError(
        "MediaPipe não encontrado. Execute: pip install mediapipe\n"
        "Este pacote é obrigatório para a detecção de landmarks da mão."
    )


# =============================================================================
# DATACLASS DE RESULTADO
# =============================================================================

@dataclass
class ProcessingResult:
    """
    Estrutura de dados que empacota todos os resultados de UM frame processado.

    Esta dataclass é o "pacote de entrega" que o ProcessingWorker monta após
    executar o pipeline completo e envia para a MainWindow via pyqtSignal.
    A MainWindow então distribui cada campo para o widget correspondente.

    Por que usar dataclass?
        Dataclasses são mais legíveis que dicionários (acesso por atributo, não
        por string), têm tipagem explícita e são imutáveis quando necessário.
        Também autodocumentam os campos que o pipeline produz.

    Campos:
        frame_overlay: Array NumPy BGR com o frame da câmera + esqueleto da mão
                       desenhado pelo _build_skeleton(). Enviado ao VideoWidget.

        angles_smooth: Dicionário {dedo: {articulação: ângulo_suavizado}} retornado
                       por GoniometryFilterBank.smooth_all(). Contém MCP, PIP, DIP,
                       ABD e TAM para dedos longos; MCP, IP e TAM para o polegar.

        hand_state: Dicionário retornado por classify_hand_state(). Contém
                    finger_states (estado de cada dedo), closed_count e hand_open.

        metrics_per_finger: {nome_do_dedo: dict_de_métricas} onde cada dict é a
                            saída de compute_realtime_metrics() — amplitude,
                            velocidade média, velocidade de pico, frequência Hz,
                            coeficiente de variação e regularidade.

        hand_detected: True se o MediaPipe encontrou uma mão neste frame.
                       False se nenhuma mão estava visível (frame ignorado).

        frame_id: Contador sequencial de frames processados nesta sessão.
                  Usado pelo MetricsWidget e gravado no CSV.

        fps: Taxa de processamento em quadros por segundo, calculada por EMA.
             Reflete a velocidade REAL do pipeline, não a da câmera.

        tam_buffers_snapshot: Cópia thread-safe dos buffers circulares de TAM
                              por dedo. Usado pelos FingerCardWidgets para os
                              mini-gráficos individuais.
    """
    frame_overlay: np.ndarray
    angles_smooth: dict
    hand_state: dict
    metrics_per_finger: dict
    hand_detected: bool
    frame_id: int
    fps: float

    # Snapshot dos buffers TAM — list[] é seguro de copiar fora do lock
    # porque listas Python são copiadas por valor com list().
    tam_buffers_snapshot: Dict[str, List[float]] = field(default_factory=dict)


# =============================================================================
# WORKER DE PROCESSAMENTO
# =============================================================================

class ProcessingWorker(QThread):
    """
    Thread de processamento goniométrico — o núcleo científico da aplicação.

    Recebe frames brutos do CameraWorker, executa o pipeline completo de análise
    e emite os resultados encapsulados em ProcessingResult para a MainWindow.

    Arquitetura interna:
        A comunicação entre CameraWorker e ProcessingWorker usa Queue(maxsize=1).

        Por que Queue(maxsize=1) e não uma lista ou deque?
            Em tempo real, queremos SEMPRE processar o frame mais recente.
            Com maxsize=1:
            - Se o processador está ocupado quando chega um frame novo,
              o frame antigo NA FILA é descartado e o novo ocupa seu lugar.
            - Isso mantém a latência sempre no mínimo possível, evitando o
              problema de "frames em fila": processar frames que já tem 2-3
              segundos de atraso em relação ao movimento real.
            - Com uma lista ou deque sem limite, frames se acumulariam
              indefinidamente, fazendo a latência crescer até o sistema travar.

    Sessão CSV:
        O CSV é iniciado por start_session() e fechado por stop_session().
        O worker só grava no CSV se uma sessão estiver ativa — permitindo que
        a câmera fique ligada sem gravar (estado READY) antes do início formal.

    Sinais emitidos:
        result_ready(object): ProcessingResult completo para a MainWindow.
                              Tipo 'object' porque PyQt6 não suporta
                              pyqtSignal(ProcessingResult) diretamente.
        processing_error(str): Mensagem de erro não fatal para o LogWidget.
    """

    # Sinal que carrega o ProcessingResult completo.
    # Usamos 'object' como tipo porque pyqtSignal não suporta dataclasses
    # customizadas diretamente. A MainWindow recebe como 'object' e faz cast.
    result_ready: pyqtSignal = pyqtSignal(object)

    # Sinal de erro para erros não fatais (ex: frame corrompido isolado).
    # Erros fatais (ex: MediaPipe não instalado) usam raise ImportError.
    processing_error: pyqtSignal = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        """
        Inicializa o ProcessingWorker com todos os componentes do pipeline.

        Os objetos científicos (MediaPipe, DigitalGoniometer, etc.) são criados
        aqui no __init__ porque:
        1. A criação é leve (sem I/O de câmera).
        2. O __init__ roda na thread principal — boa prática para detectar erros
           de import (MediaPipe não instalado) antes de iniciar a thread.
        3. Os objetos SÃO usados em run() (thread worker) — isso é seguro porque
           apenas uma thread (o worker) os acessa após start().

        Parâmetros:
            parent: Widget Qt pai (opcional). Geralmente None para workers.
        """
        super().__init__(parent)

        # Evento de parada thread-safe — mesmo padrão do CameraWorker.
        self._stop_event: threading.Event = threading.Event()

        # Fila de frames entre CameraWorker → ProcessingWorker.
        # maxsize=1: nunca acumula frames antigos, sempre processa o mais recente.
        self._frame_queue: queue.Queue = queue.Queue(maxsize=config.QUEUE_SIZE)

        # --- Componentes do pipeline científico ---

        # Detector de mãos do MediaPipe.
        # static_image_mode=False: modo de vídeo — reutiliza rastreamento entre
        # frames (mais rápido que detectar do zero a cada frame).
        # max_num_hands=1: apenas uma mão por vez, suficiente para goniometria clínica.
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=config.MP_DETECT_CONF,
            min_tracking_confidence=config.MP_TRACK_CONF,
        )

        # Goniômetro digital — calcula os ângulos articulares brutos.
        self._gonio: DigitalGoniometer = DigitalGoniometer()

        # Banco de filtros EMA → Kalman — suaviza os ângulos brutos.
        # Uma instância de SeriesFilter por série (ex: "INDEX_MCP", "THUMB_IP").
        self._filter_bank: GoniometryFilterBank = GoniometryFilterBank(
            ema_alpha=config.EMA_ALPHA,
            kalman_q=config.KALMAN_Q,
            kalman_r=config.KALMAN_R,
        )

        # --- Buffers circulares de dados temporais ---
        # Um deque por dedo para armazenar o histórico de TAM.
        # deque(maxlen=N) descarta automaticamente o valor mais antigo quando cheio,
        # garantindo que a memória nunca cresça além de BUFFER_SIZE entradas.
        self._tam_buffers: Dict[str, Deque[float]] = {
            finger: deque(maxlen=config.BUFFER_SIZE)
            for finger in FINGERS
        }

        # Um deque por dedo para armazenar os timestamps dos frames.
        # Usado por compute_realtime_metrics() para calcular velocidade (°/s)
        # e frequência (Hz) — grandezas que dependem do tempo decorrido.
        self._time_buffers: Dict[str, Deque[float]] = {
            finger: deque(maxlen=config.BUFFER_SIZE)
            for finger in FINGERS
        }

        # Conta frames consecutivos sem detecção de mão.
        # Quando ultrapassa NO_HAND_RESET_FRAMES, os filtros são resetados.
        self._no_hand_frames: int = 0

        # Contador total de frames processados nesta sessão do worker.
        self._frame_id: int = 0

        # FPS do pipeline de processamento (EMA, igual ao CameraWorker).
        self._fps_ema: float = 0.0

        # Timestamp do último frame processado com sucesso.
        self._t_last_frame: float = 0.0

        # --- Logger CSV (inativo até start_session() ser chamado) ---
        self._csv_logger: Optional[GoniometryCSVLogger] = None
        self._csv_path: str = ""
        self._session_active: bool = False

        # Lock que protege _session_active e _csv_logger.
        # A MainWindow pode chamar start_session()/stop_session() de fora
        # da thread do worker, então precisamos de sincronização.
        self._session_lock: threading.Lock = threading.Lock()

    def reset_state(self) -> None:
        """
        Zera o estado interno do worker. Chamado ao iniciar uma Nova Sessão.
        Limpa a fila de frames, reseta os filtros e zera todos os buffers numéricos.
        """
        # Esvazia a fila pendente sem travar
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        # Reseta estado científico
        self._filter_bank.reset_all()
        
        # Limpa as medições históricas de todos os dedos
        for finger in FINGERS:
            self._tam_buffers[finger].clear()
            self._time_buffers[finger].clear()
        
        # Zera contadores de frame
        self._no_hand_frames = 0
        self._frame_id = 0
        self._fps_ema = 0.0

    # =========================================================================
    # INTERFACE COM O CameraWorker
    # =========================================================================

    def put_frame(self, frame: np.ndarray) -> None:
        """
        Recebe um frame do CameraWorker e o coloca na fila de processamento.

        Este método é chamado pela MainWindow ao conectar o sinal frame_ready
        do CameraWorker. Ele roda na thread do Qt (thread principal ou thread
        do CameraWorker, dependendo do tipo de conexão do sinal).

        Estratégia "descarta o antigo, mantém o novo":
            Queue.put_nowait() lança queue.Full se a fila estiver cheia.
            Nesse caso, removemos o frame antigo com get_nowait() e inserimos
            o novo. Isso garante que o processador SEMPRE receba o frame mais
            recente, mantendo a latência mínima independentemente da velocidade
            do processador.

        Parâmetros:
            frame: Array NumPy BGR espelhado, recebido diretamente do CameraWorker.
        """
        try:
            # Tenta inserir sem bloquear (não-bloqueante = nowait).
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            # A fila está cheia (já tem 1 frame aguardando).
            # Remove o frame antigo que ainda não foi processado...
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                # Condição de corrida extremamente improvável: entre o put_nowait
                # falhar e o get_nowait executar, o worker esvaziou a fila.
                # Nenhuma ação necessária — prosseguimos normalmente.
                pass

            # ...e insere o frame mais recente no lugar.
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                # Se ainda estiver cheia após a remoção, descartamos este frame.
                # Isso não deveria ocorrer na prática, mas é tratado defensivamente.
                pass

    # =========================================================================
    # LOOP PRINCIPAL (roda na thread separada)
    # =========================================================================

    def run(self) -> None:
        """
        Método principal do QThread — chamado automaticamente pelo Qt ao
        executar processing_worker.start(). Roda inteiramente na thread worker.

        Este método NÃO deve ser chamado diretamente. Use start().

        Fluxo:
            1. Aguarda um frame na fila com timeout de 100ms.
            2. Se não chegou frame, verifica se deve parar e volta ao passo 1.
            3. Executa o pipeline completo (MediaPipe → ângulos → filtros → métricas).
            4. Monta ProcessingResult e emite result_ready.
            5. Repete até stop() ser chamado.
        """
        while not self._stop_event.is_set():

            # Aguarda um frame com timeout de 100ms.
            # Timeout é necessário para que o loop possa verificar _stop_event
            # mesmo sem receber frames (ex: câmera pausada).
            try:
                frame_bgr = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                # Nenhum frame disponível no timeout — volta ao início do loop
                # para verificar _stop_event antes de aguardar novamente.
                continue

            # Frame recebido — executa o pipeline completo.
            try:
                self._process_frame(frame_bgr)
            except Exception as exc:
                # Captura erros não fatais (ex: frame corrompido isolado).
                # Não interrompemos a thread — apenas logamos e continuamos.
                self.processing_error.emit(
                    f"Erro ao processar frame #{self._frame_id}: {exc}"
                )

        # Loop encerrado — libera recursos do MediaPipe.
        self._cleanup()

    def _process_frame(self, frame_bgr: np.ndarray) -> None:
        """
        Executa o pipeline goniométrico completo em um único frame BGR.

        Esta função orquestra todos os módulos científicos em sequência.
        A ordem das etapas é determinística e não pode ser alterada — cada
        etapa depende da saída da anterior.

        Parâmetros:
            frame_bgr: Array NumPy de shape (altura, largura, 3), formato BGR,
                       com o frame já espelhado horizontalmente pelo CameraWorker.
        """
        self._frame_id += 1
        t_now: float = time.monotonic()

        # Calcula FPS do pipeline de processamento.
        self._update_fps(t_now)
        self._t_last_frame = t_now

        # --- Etapa 1: MediaPipe Hands ---
        # Converte BGR → RGB porque o MediaPipe espera imagens em RGB.
        # O OpenCV usa BGR por herança histórica do padrão DirectShow do Windows.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Marca como não-gravável para otimização: o MediaPipe pode referenciar
        # diretamente a memória do array sem copiar, economizando ~2-5ms por frame.
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # Verifica se o MediaPipe detectou ao menos uma mão no frame.
        hand_detected: bool = results.multi_hand_landmarks is not None

        if not hand_detected:
            # Nenhuma mão visível neste frame.
            self._no_hand_frames += 1
            self._handle_no_hand(frame_bgr)
            return

        # --- Etapa 2: Identificação da mão (handedness) ---
        # Por que isso importa para goniometria?
        #   O DigitalGoniometer.compute_all() usa a normal do plano da mão
        #   para determinar o sinal dos ângulos (flexão = positivo, extensão = negativo).
        #   Para a mão esquerda, a normal aponta na direção oposta — sem correção,
        #   todos os ângulos aparecem invertidos (flexão lida como extensão).
        #
        # Como o MediaPipe reporta handedness:
        #   results.multi_handedness[0].classification[0].label retorna "Left" ou "Right".
        #   ATENÇÃO: o frame está espelhado (flip horizontal). Após o flip, uma mão
        #   direita real aparece visualmente como esquerda no frame. Mas o MediaPipe
        #   usa os landmarks 3D (não a imagem espelhada) para classificar, então
        #   "Right" do MediaPipe = mão direita real do paciente.
        is_right_hand: bool = True
        if results.multi_handedness:
            label: str = results.multi_handedness[0].classification[0].label
            is_right_hand = (label == "Right")

        # --- Etapa 3: Cálculo dos ângulos brutos ---
        # compute_all() recebe os 21 landmarks 3D normalizados (coordenadas 0.0–1.0)
        # e retorna um dicionário {dedo: {articulação: ângulo_em_graus}}.
        # Exemplo: {"INDEX": {"MCP": 45.2, "PIP": 88.1, "DIP": 62.3, "ABD": 12.1, "TAM": 195.6}}
        landmarks = results.multi_hand_landmarks[0].landmark
        angles_raw: dict = self._gonio.compute_all(landmarks, is_right_hand=is_right_hand)

        # --- Etapa 4: Suavização EMA → Kalman ---
        # Por que a ordem EMA ANTES de Kalman importa?
        #   EMA remove ruído de ALTA frequência (jitter frame a frame do MediaPipe).
        #   Kalman remove ruído de BAIXA frequência (deriva lenta, tremor fino).
        #   Se invertêssemos a ordem (Kalman → EMA), o Kalman receberia o ruído
        #   de alta frequência diretamente, perdendo sua eficiência como estimador
        #   do "valor verdadeiro" do ângulo. A combinação EMA→Kalman produz ângulos
        #   suaves tanto em alta quanto em baixa frequência.
        angles_smooth: dict = self._filter_bank.smooth_all(angles_raw)

        # Frame válido com mão detectada — zera contador de frames sem mão.
        self._no_hand_frames = 0

        # --- Etapa 5: Atualização dos buffers temporais ---
        # Armazena TAM e timestamp de cada dedo para cálculo de métricas.
        self._update_buffers(angles_smooth, t_now)

        # --- Etapa 6: Geração do overlay visual ---
        # Por que _build_skeleton() roda aqui (no worker) e não na UI?
        #   O overlay envolve operações pesadas de desenho com OpenCV:
        #   linhas do esqueleto, círculos nos landmarks, arcos angulares
        #   e texto com os valores dos ângulos. Fazer isso na thread principal
        #   bloquearia a interface por ~5-15ms por frame.
        #   Aqui no worker, o overhead é "escondido" atrás do tempo de
        #   processamento do MediaPipe, sem impacto perceptível na UI.
        #
        # Mapa de estabilidade: informa ao overlay a cor de cada articulação
        # (verde=estável, amarelo=convergindo, azul=instável), derivado do
        # ganho de Kalman atual de cada filtro.
        stability_map: dict = {
            finger: {
                joint: self._filter_bank.get_stability(finger, joint)
                for joint in angles_smooth.get(finger, {}).keys()
            }
            for finger in FINGERS
            if finger in angles_smooth
        }

        # _build_skeleton() gera apenas o painel com o esqueleto da mão.
        # Chamamos diretamente em vez de draw_goniometry_overlay() porque
        # não precisamos do painel de dados clínicos (aquele é exibido no
        # app.py/Streamlit; aqui os dados vão para os widgets individuais).
        frame_overlay: np.ndarray = _build_skeleton(
            frame=frame_bgr,
            landmarks=landmarks,
            angles=angles_smooth,
            pw=config.CAMERA_WIDTH,
            ph=config.CAMERA_HEIGHT,
            frozen=False,
            stability_map=stability_map,
        )

        # --- Etapa 7: Classificação do estado da mão ---
        # Determina se a mão está aberta ou fechada, quantos dedos estão fechados
        # e a classificação ASSH (funcional) de cada dedo por sua amplitude de TAM.
        hand_state: dict = classify_hand_state(angles_smooth)

        # --- Etapa 8: Cálculo de métricas por dedo ---
        # Para cada dedo, calcula amplitude, velocidade, frequência e regularidade
        # com base no histórico de TAM e timestamps dos últimos BUFFER_SIZE frames.
        metrics_per_finger: dict = {}
        for finger in FINGERS:
            tam_buf = list(self._tam_buffers[finger])
            time_buf = list(self._time_buffers[finger])

            # Precisa de ao menos 2 pontos para calcular velocidade e frequência.
            if len(tam_buf) >= 2 and len(tam_buf) == len(time_buf):
                metrics_per_finger[finger] = compute_realtime_metrics(
                    angle_buffer=tam_buf,
                    time_buffer=time_buf,
                )
            else:
                # Dados insuficientes — retorna métricas zeradas para não
                # exibir NaN ou erros na interface.
                metrics_per_finger[finger] = {
                    "amplitude": 0.0,
                    "vel_media": 0.0,
                    "vel_pico": 0.0,
                    "freq_hz": 0.0,
                    "cv": 0.0,
                    "regularidade": "—",
                    "n_picos": 0,
                }

        # --- Etapa 9: Snapshot dos buffers TAM para os mini-gráficos ---
        # Convertemos de deque para list() para criar uma cópia independente.
        # Uma cópia é necessária porque o deque original continua sendo
        # modificado pelo worker enquanto a MainWindow distribui os dados.
        tam_snapshot: Dict[str, List[float]] = {
            finger: list(self._tam_buffers[finger])
            for finger in FINGERS
        }

        # --- Etapa 10: Montagem e emissão do resultado ---
        result = ProcessingResult(
            frame_overlay=frame_overlay,
            angles_smooth=angles_smooth,
            hand_state=hand_state,
            metrics_per_finger=metrics_per_finger,
            hand_detected=True,
            frame_id=self._frame_id,
            fps=self._fps_ema,
            tam_buffers_snapshot=tam_snapshot,
        )

        # Emite o resultado para a MainWindow via sinal thread-safe.
        # O Qt garante que o slot receptor (na thread principal) só será
        # invocado quando a thread principal estiver disponível para processá-lo.
        self.result_ready.emit(result)

        # --- Etapa 11: Gravação no CSV (apenas se sessão ativa) ---
        # Registra a cada CSV_LOG_INTERVAL frames para reduzir I/O de disco.
        # Com CSV_LOG_INTERVAL=3 e TARGET_FPS=30, temos ~10 linhas/segundo —
        # suficiente para análise clínica sem arquivo excessivamente grande.
        if self._frame_id % config.CSV_LOG_INTERVAL == 0:
            self._try_log_csv(angles_smooth)

    def _handle_no_hand(self, frame_bgr: np.ndarray) -> None:
        """
        Lida com o caso em que nenhuma mão foi detectada no frame atual.

        Dois comportamentos principais:
        1. Após NO_HAND_RESET_FRAMES frames sem mão, reseta os filtros de Kalman.
           Sem esse reset, quando a mão retornar, o filtro tentará "convergir"
           da posição antiga para a nova, causando ângulos errôneos nos primeiros
           frames (transiente falso). O reset garante que a primeira detecção
           após ausência longa seja tratada como "estado inicial".

        2. Emite um ProcessingResult com hand_detected=False e frame original.
           Isso permite que a MainWindow limpe a interface (ex: VideoWidget
           exibe o frame sem overlay, MetricsWidget apaga os valores).

        Parâmetros:
            frame_bgr: Frame BGR original, sem overlay, para exibição na UI.
        """
        # Reseta os filtros somente após ausência prolongada para evitar
        # resets desnecessários por oclusões momentâneas dos dedos.
        if self._no_hand_frames >= config.NO_HAND_RESET_FRAMES:
            self._filter_bank.reset_all()
            self._no_hand_frames = 0

        # Emite resultado indicando ausência de mão para a UI.
        result = ProcessingResult(
            frame_overlay=frame_bgr.copy(),
            angles_smooth={},
            hand_state={"finger_states": {}, "closed_count": 0, "hand_open": True},
            metrics_per_finger={},
            hand_detected=False,
            frame_id=self._frame_id,
            fps=self._fps_ema,
            tam_buffers_snapshot={f: [] for f in FINGERS},
        )
        self.result_ready.emit(result)

    # =========================================================================
    # BUFFERS TEMPORAIS
    # =========================================================================

    def _update_buffers(self, angles_smooth: dict, timestamp: float) -> None:
        """
        Atualiza os buffers circulares de TAM e timestamps de cada dedo.

        Os buffers são mantidos pelo worker e atualizados frame a frame.
        Eles acumulam o histórico de BUFFER_SIZE entradas mais recentes,
        que é usado por compute_realtime_metrics() para calcular métricas
        baseadas em janela deslizante (amplitude, velocidade, frequência).

        Parâmetros:
            angles_smooth: Dicionário com ângulos suavizados de todos os dedos.
            timestamp: Tempo atual em segundos (time.monotonic()) — mesma origem
                       para todos os dedos no mesmo frame.
        """
        for finger in FINGERS:
            finger_data = angles_smooth.get(finger, {})

            # Extrai o TAM (Total Active Motion) do dedo.
            # TAM é a métrica clínica mais importante: representa a amplitude
            # total de movimento ativo de todas as articulações do dedo somadas.
            tam_value: float = float(finger_data.get("TAM", 0.0))

            # Só adiciona ao buffer se o valor é válido (> 0).
            # TAM = 0.0 geralmente indica frame sem detecção ou articulação ausente,
            # não um ângulo real — incluir zeros distorceria as métricas de amplitude
            # e frequência calculadas a partir deste buffer.
            if tam_value > 0.0:
                self._tam_buffers[finger].append(tam_value)
                self._time_buffers[finger].append(timestamp)

    def get_tam_buffers(self) -> Dict[str, List[float]]:
        """
        Retorna uma cópia thread-safe dos buffers de TAM atuais.

        Usado pela MainWindow para alimentar os mini-gráficos dos FingerCardWidgets.
        Retorna cópias (list()) em vez dos deques originais para que o
        chamador não precise de sincronização adicional.

        Retorno:
            Dict mapeando nome do dedo (ex: "INDEX") para lista de floats
            com os valores de TAM dos últimos BUFFER_SIZE frames válidos.
        """
        return {
            finger: list(self._tam_buffers[finger])
            for finger in FINGERS
        }

    # =========================================================================
    # CÁLCULO DE FPS DO PIPELINE
    # =========================================================================

    def _update_fps(self, t_now: float) -> None:
        """
        Atualiza o FPS do pipeline de processamento usando EMA.

        FPS do pipeline ≠ FPS da câmera.
        A câmera pode capturar a 30 FPS, mas o processamento pode ser mais
        lento (ex: 20 FPS em CPU lenta) ou mais rápido (ex: 25 FPS se a câmera
        às vezes pula frames). Este método mede a velocidade REAL do processamento.

        Parâmetros:
            t_now: Timestamp atual em segundos (time.monotonic()).
                   Comparado com o timestamp do frame anterior para calcular dt.
        """
        if self._t_last_frame <= 0.0:
            # Primeiro frame — sem referência anterior para calcular intervalo.
            return

        dt: float = t_now - self._t_last_frame

        # Protege contra dt zero (dois frames processados no mesmo instante).
        if dt <= 0.0:
            return

        # FPS instantâneo deste frame.
        fps_instant: float = 1.0 / dt

        # EMA com α=0.15 — mesmo valor do CameraWorker para consistência.
        ema_alpha: float = 0.15

        if self._fps_ema == 0.0:
            self._fps_ema = fps_instant
        else:
            self._fps_ema = ema_alpha * fps_instant + (1.0 - ema_alpha) * self._fps_ema

    # =========================================================================
    # GERENCIAMENTO DE SESSÃO CSV
    # =========================================================================

    def start_session(self, csv_path: str) -> None:
        """
        Inicia uma nova sessão de gravação em CSV.

        Chamado pela MainWindow ao clicar "Iniciar Sessão", ANTES de start().
        Cria o GoniometryCSVLogger que gravará os ângulos de cada frame.

        O lock _session_lock protege _csv_logger e _session_active porque
        este método é chamado da thread principal enquanto o worker pode estar
        lendo _session_active no loop run(). Sem o lock, haveria race condition.

        Parâmetros:
            csv_path: Caminho completo do arquivo CSV a ser criado/aberto.
                      Exemplo: "session_goniometry_20260623_143512.csv"
        """
        with self._session_lock:
            # Fecha qualquer sessão anterior que possa estar aberta.
            if self._csv_logger is not None:
                self._csv_logger.close()

            self._csv_path = csv_path
            self._csv_logger = GoniometryCSVLogger(csv_path)
            self._session_active = True

    def stop_session(self) -> None:
        """
        Encerra a sessão de gravação CSV de forma segura.

        Chamado pela MainWindow ao clicar "Encerrar Sessão". Garante que todos
        os dados pendentes no buffer do logger são gravados em disco (flush)
        antes de fechar o arquivo.

        Por que flush() antes de close()?
            Python usa buffers de escrita por performance: os dados são mantidos
            em memória e gravados em lotes. Se o arquivo for fechado sem flush(),
            os dados no buffer podem ser perdidos (especialmente em caso de
            crash subsequente). flush() força a gravação imediata em disco.
        """
        with self._session_lock:
            if self._csv_logger is not None:
                self._csv_logger.flush()
                self._csv_logger.close()
                self._csv_logger = None
            self._session_active = False

    def _try_log_csv(self, angles_smooth: dict) -> None:
        """
        Tenta gravar o frame atual no CSV, se uma sessão estiver ativa.

        Executa de dentro do loop run() (thread worker). Usa o mesmo lock
        que start_session() e stop_session() para acesso thread-safe.

        A verificação de _session_active é feita dentro do lock para evitar
        a condição de corrida "check-then-act": sem lock, poderia acontecer
        de _session_active ser True ao verificar, mas _csv_logger ser fechado
        (None) por stop_session() antes de chegar no csv_logger.log().

        Parâmetros:
            angles_smooth: Dicionário de ângulos suavizados do frame atual.
        """
        with self._session_lock:
            if self._session_active and self._csv_logger is not None:
                self._csv_logger.log(self._frame_id, angles_smooth)

                # Flush a cada 60 frames para reduzir I/O sem risco de perda.
                # 60 frames × CSV_LOG_INTERVAL = a cada ~18 frames reais ≈ 0.6s.
                if self._frame_id % 60 == 0:
                    self._csv_logger.flush()

    # =========================================================================
    # CONTROLE DO CICLO DE VIDA
    # =========================================================================

    def stop(self) -> None:
        """
        Sinaliza ao loop de processamento que ele deve encerrar de forma limpa.

        Chamado pela MainWindow em closeEvent() ou ao clicar "Encerrar".
        Não força encerramento imediato — o loop finaliza o frame atual
        e então verifica _stop_event na próxima iteração.

        Retorno:
            None. O encerramento real é assíncrono — use wait() para aguardar.
        """
        self._stop_event.set()

    def _cleanup(self) -> None:
        """
        Libera todos os recursos do pipeline após o fim do loop.

        Chamado automaticamente ao final de run() quando o loop encerra.
        Garante que o MediaPipe libere memória GPU/CPU e que o CSV seja fechado.

        Por que fechar o MediaPipe explicitamente?
            mp.solutions.hands.Hands mantém recursos do TensorFlow Lite internamente.
            Sem close(), esses recursos podem persistir até o GC do Python coletar
            o objeto — o que pode nunca acontecer até o processo terminar, causando
            vazamento de memória em execuções longas.
        """
        # Fecha o detector do MediaPipe e libera recursos de modelo de ML.
        if self._hands is not None:
            self._hands.close()

        # Garante que o CSV seja fechado mesmo se stop_session() não foi chamado
        # explicitamente (ex: crash ou fechamento de janela abrupto).
        self.stop_session()
