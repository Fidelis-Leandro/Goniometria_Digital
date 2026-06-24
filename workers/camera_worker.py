"""
workers/camera_worker.py — Thread de captura de frames da webcam
================================================================

Este módulo isola completamente a captura de vídeo em uma thread separada
(CameraWorker), garantindo que a interface gráfica (MainWindow) nunca fique
bloqueada aguardando frames da câmera.

Problema que este módulo resolve:
    cap.read() é uma chamada BLOQUEANTE: o programa para e espera até que
    um frame chegue da câmera (~33ms a 30 FPS). Se essa espera acontecesse
    na thread principal, a janela PyQt6 congelaria a cada frame, tornando
    a interface irresponsiva.

Solução:
    CameraWorker roda em sua própria thread via QThread. Ele captura frames
    em um loop contínuo e os entrega à thread principal via pyqtSignal —
    o mecanismo seguro do Qt para comunicação entre threads.

Fluxo de dados:
    Webcam → cap.read() → flip horizontal → pyqtSignal(frame_bgr)
                                                     ↓
                                          ProcessingWorker.put_frame()

Regras respeitadas:
    - NUNCA chamamos widgets de dentro desta thread.
    - NUNCA usamos st.write, st.rerun ou qualquer API do Streamlit aqui.
    - Toda comunicação com a interface é via pyqtSignal (thread-safe por design).
"""

import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

# Importa todas as constantes do arquivo de configuração centralizado.
# NUNCA use números diretamente neste módulo — sempre via config.
import config


class CameraWorker(QThread):
    """
    Thread de captura de vídeo em tempo real via OpenCV.

    Herda de QThread (e não de threading.Thread) porque o Qt exige que
    toda comunicação com a interface gráfica aconteça através do seu próprio
    sistema de sinais (pyqtSignal). threads Python puras não têm acesso
    ao loop de eventos do Qt e causariam crashes ao tentar atualizar widgets.

    Ciclo de vida:
        1. Instanciado pela MainWindow no __init__() — ainda não inicia.
        2. camera_worker.start() é chamado ao clicar "Iniciar Sessão".
        3. O Qt chama run() automaticamente em uma thread separada.
        4. camera_worker.stop() é chamado ao clicar "Encerrar" ou fechar a janela.
        5. O loop encerra e cap.release() libera a câmera.

    Sinais emitidos (comunicação thread-safe com a MainWindow):
        frame_ready(np.ndarray) : frame BGR capturado, espelhado e pronto
                                  para ser processado pelo ProcessingWorker.
        fps_updated(float)      : taxa de quadros atual, calculada com EMA.
                                  Recebida pelo MetricsWidget para exibição.
        camera_error(str)       : mensagem de erro para o LogWidget quando
                                  a câmera não pode ser aberta ou trava.
    """

    # --- Definição dos sinais ---
    # pyqtSignal declara os tipos dos dados que cada sinal carrega.
    # O Qt usa isso para roteamento seguro entre threads.

    # Carrega um array NumPy BGR — o frame capturado e espelhado.
    frame_ready: pyqtSignal = pyqtSignal(np.ndarray)

    # Carrega um float — o FPS suavizado para exibição na interface.
    fps_updated: pyqtSignal = pyqtSignal(float)

    # Carrega uma string — mensagem de erro legível para o LogWidget.
    camera_error: pyqtSignal = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        """
        Inicializa o CameraWorker com estado inicial seguro.

        Apenas configura os atributos internos. A câmera NÃO é aberta aqui —
        isso acontece em run() quando a thread é iniciada. Essa separação
        é importante: o __init__ roda na thread principal, enquanto a câmera
        deve ser aberta e usada exclusivamente na thread do worker.

        Parâmetros:
            parent: Widget Qt pai (opcional). Usado pelo Qt para gerenciar
                    o ciclo de vida do objeto. Geralmente None para workers.
        """
        super().__init__(parent)

        # Evento de threading usado para sinalizar que o loop deve parar.
        # Usamos threading.Event (e não uma variável booleana simples) porque
        # ele é thread-safe: pode ser lido/escrito de qualquer thread sem
        # condições de corrida.
        self._stop_event: threading.Event = threading.Event()

        # Referência ao objeto da câmera. Inicialmente None porque a câmera
        # só é aberta quando run() é chamado pela thread worker.
        self._cap: Optional[cv2.VideoCapture] = None

        # Armazena o FPS suavizado pela EMA entre emissões.
        # Inicializa em 0.0 para indicar que nenhum frame foi capturado ainda.
        self._fps_ema: float = 0.0

        # Contador de frames capturados com sucesso nesta execução.
        # Usado para controlar a frequência de emissão do sinal fps_updated.
        self._frame_count: int = 0

    # =========================================================================
    # ABERTURA DA CÂMERA
    # =========================================================================

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        """
        Tenta abrir a câmera com a melhor configuração disponível no sistema.

        Estratégia de fallback:
            1. Tenta CAP_DSHOW (DirectShow — backend nativo do Windows).
               CAP_DSHOW reduz significativamente a latência no Windows porque
               elimina a camada de abstração do driver genérico. Sem ele, cada
               cap.read() pode ter latência extra de 50–150ms.
            2. Se CAP_DSHOW falhar (Linux/macOS ou driver incompatível),
               tenta o backend padrão do OpenCV (automático por SO).
            3. Se ambos falharem, retorna None para que run() possa emitir
               camera_error e encerrar o loop com segurança.

        Retorno:
            cv2.VideoCapture: objeto de câmera aberto e configurado, ou
            None se a câmera não pôde ser aberta.
        """
        # CAP_DSHOW é exclusivo do Windows — só tentamos neste sistema operacional.
        # No Linux/macOS, cv2.CAP_DSHOW não existe ou é ignorado.
        if sys.platform == "win32":
            cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)

            # Verifica se a abertura com CAP_DSHOW funcionou antes de configurar.
            if cap.isOpened():
                self._configure_camera(cap)
                return cap

            # Se CAP_DSHOW falhou, libera o recurso antes de tentar novamente.
            cap.release()

        # Fallback: backend padrão do OpenCV (V4L2 no Linux, AVFoundation no macOS).
        cap = cv2.VideoCapture(config.CAMERA_INDEX)

        if cap.isOpened():
            self._configure_camera(cap)
            return cap

        # Nenhum backend funcionou — a câmera não está disponível.
        return None

    def _configure_camera(self, cap: cv2.VideoCapture) -> None:
        """
        Aplica as configurações de resolução e FPS ao objeto de câmera.

        O OpenCV não garante que o driver honre as configurações solicitadas —
        ele tenta, mas a câmera pode retornar a resolução mais próxima que suporta.
        Por isso, usamos CAP_PROP_BUFFERSIZE = 1 para garantir que o buffer
        interno do driver tenha no máximo 1 frame enfileirado, mantendo a
        latência mínima independentemente da resolução real.

        Parâmetros:
            cap: objeto cv2.VideoCapture já aberto e válido.
        """
        # Solicita a resolução definida em config.py.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        # Solicita a taxa de FPS desejada ao driver.
        cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

        # Define o tamanho do buffer interno do driver como 1 frame.
        # Com buffers maiores (padrão = 4 no Windows), cap.read() retorna
        # frames ANTIGOS do buffer antes de capturar o frame atual.
        # Isso gera latência acumulada: a imagem exibida fica cada vez mais
        # atrasada em relação ao movimento real. Com BUFFERSIZE = 1, sempre
        # recebemos o frame mais recente.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # =========================================================================
    # LOOP PRINCIPAL (roda na thread separada)
    # =========================================================================

    def run(self) -> None:
        """
        Método principal do QThread — chamado automaticamente pelo Qt ao
        executar camera_worker.start(). Roda inteiramente na thread worker,
        nunca na thread principal.

        Este método NÃO deve ser chamado diretamente. Use start() para
        iniciar a thread de forma correta.

        Fluxo interno:
            1. Tenta abrir a câmera.
            2. Se falhar, emite camera_error e encerra.
            3. Loop de captura: cap.read() → flip → calcular FPS → emitir sinais.
            4. Ao parar (_stop_event ativado ou falhas consecutivas), libera a câmera.

        Retorno:
            None. Resultados são entregues via pyqtSignal (frame_ready, etc.).
        """
        # Tenta abrir a câmera antes de entrar no loop.
        self._cap = self._open_camera()

        if self._cap is None:
            # Emitir o sinal de erro é thread-safe: o Qt roteará a chamada
            # para a thread principal automaticamente, onde o LogWidget está.
            self.camera_error.emit(
                f"Não foi possível abrir a câmera (índice {config.CAMERA_INDEX}). "
                "Verifique se ela está conectada e não está em uso por outro programa."
            )
            return

        # Contador de falhas consecutivas de cap.read().
        # Se este contador atingir o limite, interpretamos como falha de hardware.
        consecutive_failures: int = 0

        # Limite de falhas antes de encerrar. Não usar número literal — vem de config.
        # Aqui calculamos dinamicamente: 10 falhas consecutivas a ~30fps = ~333ms
        # de câmera sem resposta, o que indica travamento real.
        max_consecutive_failures: int = 10

        # Timestamp do último frame capturado com sucesso — para cálculo de FPS.
        t_last: float = time.perf_counter()

        # Loop principal de captura — roda até que stop() seja chamado
        # ou que o número de falhas consecutivas seja atingido.
        while not self._stop_event.is_set():

            # cap.read() é BLOQUEANTE: espera até que um frame esteja disponível.
            # Retorna (True, frame) em caso de sucesso ou (False, None) em falha.
            ret, frame = self._cap.read()

            if not ret or frame is None:
                consecutive_failures += 1

                if consecutive_failures >= max_consecutive_failures:
                    # Câmera parou de responder por tempo suficiente para
                    # considerarmos uma falha de hardware real (desconexão, driver, etc.)
                    self.camera_error.emit(
                        f"Câmera perdida após {max_consecutive_failures} frames "
                        "inválidos consecutivos. Verifique a conexão USB."
                    )
                    break

                # Aguarda 10ms antes de tentar novamente.
                # Sem este sleep, o loop giraria em velocidade máxima consumindo
                # 100% de um núcleo de CPU apenas tentando ler frames inválidos.
                time.sleep(0.01)
                continue

            # Frame válido — zera o contador de falhas.
            consecutive_failures = 0
            self._frame_count += 1

            # Espelhamento horizontal do frame.
            # O MediaPipe funciona com a imagem original, mas do ponto de vista
            # do usuário, ver a própria mão espelhada (como um espelho físico)
            # é mais intuitivo para posicionar a mão na câmera.
            # cv2.flip(frame, 1): 1 = eixo vertical (espelho horizontal).
            frame = cv2.flip(frame, 1)

            # Calcula o FPS real e atualiza o valor suavizado pela EMA.
            self._update_fps(t_last)
            t_last = time.perf_counter()

            # Emite o frame capturado para quem estiver conectado ao sinal.
            # Em produção, o ProcessingWorker recebe via put_frame().
            # O sinal é thread-safe por design do Qt — não há risco de
            # condição de corrida ao emitir de dentro desta thread.
            self.frame_ready.emit(frame)

            # Emite o FPS suavizado a cada N frames para não inundar a UI.
            # Emitir a cada frame (30x/s) sobrecarregaria desnecessariamente
            # o MetricsWidget com atualizações tão rápidas que o olho humano
            # não perceberia diferença. A cada 30 frames ≈ 1 vez por segundo
            # é suficiente para manter o indicador visual atualizado.
            if self._frame_count % 30 == 0:
                self.fps_updated.emit(self._fps_ema)

        # --- Limpeza após o loop ---
        # O loop terminou (por stop() ou por falha). Libera os recursos da câmera.
        self._release_camera()

    # =========================================================================
    # CÁLCULO DE FPS
    # =========================================================================

    def _update_fps(self, t_last: float) -> None:
        """
        Atualiza o FPS suavizado usando Média Móvel Exponencial (EMA).

        Por que EMA em vez de média simples?
            A média simples (total_frames / tempo_total) tem dois problemas:
            1. Reage muito lentamente a mudanças de performance (necessita
               acumular muitos frames para refletir a velocidade atual).
            2. Nunca "esquece" frames antigos — se o sistema ficou lento por
               1 segundo no início, isso afeta a média por toda a sessão.

            A EMA com α=0.15 resolve ambos:
            - Reage rapidamente a mudanças (α controla a velocidade de resposta).
            - "Esquece" lentamente os valores antigos, mantendo o valor suavizado.
            - É computacionalmente trivial: apenas uma multiplicação e uma soma.

        Parâmetros:
            t_last: timestamp (em segundos) do frame anterior, obtido com
                    time.perf_counter(). Usado para calcular o delta de tempo.
        """
        # Calcula quanto tempo passou desde o último frame capturado com sucesso.
        t_now: float = time.perf_counter()
        dt: float = t_now - t_last

        # Protege contra divisão por zero: se dt for absurdamente pequeno
        # (dois frames no mesmo instante — impossível na prática, mas defensivo),
        # não atualizamos o FPS para evitar valores infinitos.
        if dt <= 0.0:
            return

        # FPS instantâneo deste frame: inverso do intervalo entre frames.
        fps_instant: float = 1.0 / dt

        # Fator de suavização da EMA — α=0.15 (15% do valor novo + 85% do histórico).
        # Valor empiricamente ajustado: suaviza picos de FPS causados por
        # variações de latência do driver de câmera sem introduzir lag visível
        # no indicador de FPS da interface.
        ema_alpha: float = 0.15

        if self._fps_ema == 0.0:
            # Na primeira leitura, inicializa com o valor instantâneo.
            # Se usássemos a fórmula EMA aqui, o valor inicial 0.0 arrastaria
            # o FPS para baixo nas primeiras dezenas de frames.
            self._fps_ema = fps_instant
        else:
            # Fórmula da EMA: novo = α × valor_atual + (1-α) × valor_anterior
            # O resultado fica "entre" o FPS atual e o histórico acumulado.
            self._fps_ema = ema_alpha * fps_instant + (1.0 - ema_alpha) * self._fps_ema

    # =========================================================================
    # CONTROLE DO CICLO DE VIDA
    # =========================================================================

    def stop(self) -> None:
        """
        Sinaliza ao loop de captura que ele deve encerrar de forma limpa.

        Este método é chamado pela thread principal (ex: ao clicar "Encerrar"
        ou ao fechar a janela). Ele NÃO força a thread a parar imediatamente —
        o loop verifica o evento a cada iteração e encerra na próxima oportunidade.

        Por que threading.Event em vez de uma variável booleana simples?
            Variáveis booleanas Python não são thread-safe: leituras e escritas
            de threads diferentes podem resultar em dados corrompidos (race condition).
            threading.Event usa primitivas de sincronização do sistema operacional
            que garantem acesso seguro de qualquer thread, sem locks manuais.

        Retorno:
            None. O encerramento real acontece no loop de run() de forma assíncrona.
        """
        # Marca o evento de parada — o loop em run() verificará isso na
        # próxima iteração e encerrará com segurança.
        self._stop_event.set()

    def _release_camera(self) -> None:
        """
        Libera os recursos da câmera de forma segura ao encerrar a thread.

        Por que liberar explicitamente?
            O Python tem coleta de lixo automática, mas ela não garante QUANDO
            um objeto será destruído. Se cap.release() não for chamado explicitamente,
            o driver de câmera pode continuar ocupado, impedindo que outros programas
            (ou uma nova instância do nosso) abram a câmera.

            No Windows, isso resulta no erro: "câmera já está em uso por outro
            processo" ao tentar reiniciar a aplicação sem fechar o processo anterior.

        Retorno:
            None.
        """
        if self._cap is not None and self._cap.isOpened():
            # Libera o handle da câmera no driver do sistema operacional.
            self._cap.release()

        # Redefine a referência para None para evitar uso acidental após liberação.
        self._cap = None

    def is_camera_open(self) -> bool:
        """
        Verifica se a câmera está atualmente aberta e disponível.

        Útil para verificações de estado na MainWindow antes de tentar
        iniciar uma nova sessão de captura.

        Retorno:
            bool: True se a câmera está aberta e operacional, False caso contrário.
        """
        return self._cap is not None and self._cap.isOpened()
