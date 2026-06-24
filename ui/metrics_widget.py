"""
ui/metrics_widget.py — Painel de métricas do sistema e estado da mão
=====================================================================

Este módulo implementa o MetricsWidget: um painel lateral que exibe em
tempo real as métricas de desempenho do sistema (FPS, CPU, RAM) e o estado
clínico da mão (aberta/fechada, contagem de dedos, identificação).

Responsabilidade:
    Receber um ProcessingResult pronto (calculado pelo ProcessingWorker) e
    atualizar os cartões visuais correspondentes. Não faz cálculos — apenas
    formata e exibe os dados que chegam.

Layout dos cartões (grade 2 linhas × 3 colunas):
    ┌──────────┬──────────┬──────────┐
    │   FPS    │   CPU    │   RAM    │
    ├──────────┼──────────┼──────────┤
    │  Frame#  │  Estado  │  Estado  │
    │          │  (mão)   │ (amplo)  │
    └──────────┴──────────┴──────────┘

    O cartão de Estado da Mão ocupa 2 colunas na segunda linha para ter
    espaço suficiente para o texto "🟢 MÃO ABERTA (X/5)" e "🔴 MÃO FECHADA".

Integração na MainWindow:
    self.metrics_widget = MetricsWidget()
    processing_worker.result_ready.connect(
        lambda result: self.metrics_widget.update_from_result(result)
    )
"""

from typing import Optional, Tuple

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# Importa os estilos centralizados do tema escuro.
from themes import (
    CARD_STYLE,
    CARD_HAND_CLOSED_STYLE,
    CARD_HAND_OPEN_STYLE,
    LABEL_HAND_STATE_STYLE,
    LABEL_TITLE_STYLE,
    LABEL_VALUE_STYLE,
)

# Importa a dataclass de resultado do worker para tipagem correta.
# Importação condicional evita import circular em caso de reorganização futura.
from workers.processing_worker import ProcessingResult

# Tenta importar psutil para coleta de métricas do sistema operacional.
# psutil é uma dependência OPCIONAL: se não estiver instalado, os cartões
# de CPU e RAM exibem "—" em vez de lançar uma exceção fatal.
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


class _MetricCard(QWidget):
    """
    Cartão visual reutilizável para exibir uma única métrica.

    Cada cartão tem:
    - Um QFrame como container com borda arredondada (visual de "card").
    - Um QLabel de título (ex: "FPS") em texto secundário pequeno.
    - Um QLabel de valor (ex: "58.3") em texto grande e em negrito.

    Este componente interno (_MetricCard, com underscore = privado ao módulo)
    é instanciado pelo MetricsWidget para cada uma das métricas. Centralizar
    a lógica de construção aqui evita repetição de código para os 5 cartões.
    """

    def __init__(
        self,
        title: str,
        initial_value: str = "—",
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Constrói um cartão de métrica com título e valor inicial.

        Parâmetros:
            title: Rótulo estático exibido no topo do cartão. Ex: "FPS", "CPU".
            initial_value: Valor exibido antes de qualquer dado real chegar.
                           Padrão "—" indica "sem dados disponíveis".
            parent: Widget pai Qt (opcional).
        """
        super().__init__(parent)

        # Layout vertical interno: título em cima, valor embaixo.
        layout = QVBoxLayout(self)
        # Margens internas pequenas para não desperdiçar espaço no painel lateral.
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        # Container com borda arredondada (estilo vem do themes.py).
        self._frame = QFrame()
        self._frame.setStyleSheet(CARD_STYLE)

        frame_layout = QVBoxLayout(self._frame)
        frame_layout.setContentsMargins(8, 6, 8, 6)
        frame_layout.setSpacing(2)
        frame_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Rótulo do título — texto pequeno em cinza secundário.
        self._label_title = QLabel(title)
        self._label_title.setStyleSheet(LABEL_TITLE_STYLE)
        self._label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Rótulo do valor — texto grande em negrito, destaque visual.
        self._label_value = QLabel(initial_value)
        self._label_value.setStyleSheet(LABEL_VALUE_STYLE)
        self._label_value.setAlignment(Qt.AlignmentFlag.AlignCenter)

        frame_layout.addWidget(self._label_title)
        frame_layout.addWidget(self._label_value)

        layout.addWidget(self._frame)

        # Permite que o cartão encolha verticalmente sem distorcer o layout.
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

    def set_value(self, value: str) -> None:
        """
        Atualiza o texto do label de valor do cartão.

        Chamado pelos métodos de atualização do MetricsWidget a cada novo resultado.

        Parâmetros:
            value: String formatada a exibir. Ex: "58.3", "23%", "4.1 GB".
        """
        self._label_value.setText(value)

    def set_frame_style(self, style: str) -> None:
        """
        Substitui o estilo visual do QFrame interno (cor de fundo, borda).

        Usado pelo cartão de Estado da Mão para alternar entre fundo verde
        (mão aberta) e fundo vermelho (mão fechada).

        Parâmetros:
            style: String de Qt StyleSheet para o QFrame.
                   Normalmente CARD_HAND_OPEN_STYLE ou CARD_HAND_CLOSED_STYLE.
        """
        self._frame.setStyleSheet(style)


class HandStateCard(QWidget):
    """
    Cartão especializado para exibir o estado clínico da mão.

    Diferente dos outros cartões (_MetricCard), este exibe:
    - Ícone colorido (🟢 ou 🔴).
    - Texto grande indicando ABERTA ou FECHADA.
    - Contagem de dedos fechados entre parênteses: "(X/5)".
    - Fundo que muda de cor (verde / vermelho) conforme o estado.

    A mudança de cor de fundo é o elemento mais importante: permite que
    o fisioterapeuta avalie o estado da mão com uma rápida olhada lateral,
    sem precisar ler o texto.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Inicializa o cartão de estado com visual padrão (sem dados).

        Parâmetros:
            parent: Widget pai Qt (opcional).
        """
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        # Container principal do cartão.
        self._frame = QFrame()
        self._frame.setStyleSheet(CARD_STYLE)

        frame_layout = QVBoxLayout(self._frame)
        frame_layout.setContentsMargins(10, 8, 10, 8)
        frame_layout.setSpacing(4)
        frame_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Rótulo de título estático.
        self._label_title = QLabel("ESTADO DA MÃO")
        self._label_title.setStyleSheet(LABEL_TITLE_STYLE)
        self._label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Rótulo de estado principal — texto grande, muda conforme a detecção.
        self._label_state = QLabel("⬜ AGUARDANDO")
        self._label_state.setStyleSheet(LABEL_HAND_STATE_STYLE)
        self._label_state.setAlignment(Qt.AlignmentFlag.AlignCenter)

        frame_layout.addWidget(self._label_title)
        frame_layout.addWidget(self._label_state)

        layout.addWidget(self._frame)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

    def update_state(self, hand_open: bool, closed_count: int, hand_detected: bool) -> None:
        """
        Atualiza o estado visual completo do cartão da mão.

        Muda simultaneamente:
        1. O texto e ícone (🟢/🔴).
        2. A cor de fundo do frame (verde/vermelho).

        Quando não há mão detectada, exibe estado neutro sem cor de alerta
        para não confundir o clínico durante posicionamento da câmera.

        Parâmetros:
            hand_open: True se a mão está considerada aberta (maioria dos dedos
                       com TAM acima do threshold), False se fechada.
            closed_count: Número de dedos considerados fechados (0–5).
            hand_detected: True se o MediaPipe encontrou uma mão neste frame.
        """
        if not hand_detected:
            # Sem mão detectada: estado neutro sem indicação de erro.
            self._frame.setStyleSheet(CARD_STYLE)
            self._label_state.setText("⬜ SEM DETECÇÃO")
            return

        # Calcula quantos dedos estão abertos (complemento dos fechados).
        # Exibimos dedos ABERTOS porque é mais intuitivo clinicamente:
        # "2/5 dedos abertos" informa o grau de abertura, não de fechamento.
        open_count: int = 5 - closed_count

        if hand_open:
            # Fundo verde escuro: mão considerada aberta — estado funcional positivo.
            self._frame.setStyleSheet(CARD_HAND_OPEN_STYLE)
            self._label_state.setText(f"🟢 MÃO ABERTA ({open_count}/5)")
        else:
            # Fundo vermelho escuro: mão considerada fechada — alerta clínico.
            self._frame.setStyleSheet(CARD_HAND_CLOSED_STYLE)
            self._label_state.setText(f"🔴 MÃO FECHADA ({open_count}/5)")


# =============================================================================
# WIDGET PRINCIPAL
# =============================================================================

class MetricsWidget(QGroupBox):
    """
    Painel lateral de métricas do sistema e estado clínico da mão.

    Organiza os cartões individuais em uma grade 2×3 e conecta as fontes
    de dados (ProcessingResult e psutil) a cada cartão correspondente.

    Hierarquia de widgets:
        MetricsWidget (QGroupBox)
        └── QGridLayout
            ├── _MetricCard("FPS")          [linha 0, coluna 0]
            ├── _MetricCard("CPU")          [linha 0, coluna 1]
            ├── _MetricCard("RAM")          [linha 0, coluna 2]
            ├── _MetricCard("Frame #")      [linha 1, coluna 0]
            └── HandStateCard               [linha 1, colunas 1–2, colspan=2]
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Inicializa o MetricsWidget com todos os cartões e o timer de sistema.

        O QTimer de sistema (_stats_timer) é iniciado aqui e dispara a cada
        1000ms para atualizar CPU e RAM independentemente dos frames processados.
        Isso garante que as métricas de sistema permaneçam atualizadas mesmo
        quando a câmera não está ativa (ex: estado IDLE).

        Parâmetros:
            parent: Widget pai Qt (opcional).
        """
        super().__init__("Métricas do Sistema", parent)

        # Grade 2 linhas × 3 colunas para os cartões de métricas.
        self._grid = QGridLayout(self)
        self._grid.setSpacing(8)
        self._grid.setContentsMargins(10, 16, 10, 10)

        # --- Linha 0: métricas de desempenho ---

        # FPS do pipeline de processamento (não da câmera).
        # Reflete a velocidade REAL do ProcessingWorker.
        self._card_fps = _MetricCard("FPS", "—")

        # Uso do processador em porcentagem do sistema inteiro.
        # psutil.cpu_percent() mede todos os núcleos.
        self._card_cpu = _MetricCard("CPU", "—")

        # Uso de memória RAM em Gigabytes.
        # Importante monitorar: MediaPipe + buffers podem usar bastante memória.
        self._card_ram = _MetricCard("RAM", "—")

        self._grid.addWidget(self._card_fps, 0, 0)
        self._grid.addWidget(self._card_cpu, 0, 1)
        self._grid.addWidget(self._card_ram, 0, 2)

        # --- Linha 1: frame counter + estado da mão ---

        # Contador de frames processados nesta sessão.
        # Útil para correlacionar eventos no log com frames no CSV.
        self._card_frame = _MetricCard("Frame #", "—")

        # Cartão especializado com mudança de cor para o estado da mão.
        # Ocupa 2 colunas (colspan=2) para ter espaço para o texto completo.
        self._card_hand = HandStateCard()

        self._grid.addWidget(self._card_frame, 1, 0)

        # colspan=2: o cartão de estado ocupa as colunas 1 e 2 da linha 1.
        # Isso dá mais espaço horizontal para o texto "MÃO ABERTA (5/5)".
        self._grid.addWidget(self._card_hand, 1, 1, 1, 2)

        # Garante que as 3 colunas da grade tenham o mesmo peso de espaço.
        # Sem isso, colunas com conteúdo menor ficariam mais estreitas.
        for col in range(3):
            self._grid.setColumnStretch(col, 1)

        # --- Timer de atualização das métricas do sistema ---
        # Dispara a cada 1000ms (1 segundo) — taxa adequada para CPU e RAM.
        # Atualizar mais rápido não traria informação adicional útil, pois
        # psutil.cpu_percent() já faz suavização interna.
        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(1000)
        self._stats_timer.timeout.connect(self._update_system_stats)

        # Inicia o timer imediatamente — exibe valores de CPU/RAM desde
        # o início, mesmo antes da câmera ser ligada.
        self._stats_timer.start()

        # Força a primeira leitura de CPU/RAM imediatamente ao criar o widget.
        self._update_system_stats()

    # =========================================================================
    # ATUALIZAÇÃO COM DADOS DO PROCESSAMENTO
    # =========================================================================

    def update_from_result(self, result: ProcessingResult) -> None:
        """
        Atualiza os cartões de FPS, Frame# e Estado da Mão com dados do worker.

        Chamado pela MainWindow a cada emissão do sinal result_ready do
        ProcessingWorker (~30 vezes/segundo). Deve ser rápido — apenas
        atualiza texto, sem nenhum cálculo ou acesso a disco.

        Parâmetros:
            result: ProcessingResult emitido pelo ProcessingWorker.
                    Contém fps, frame_id, hand_state e hand_detected.
        """
        # Formata o FPS com uma casa decimal para leitura estável.
        # Duas casas decimais causam "jitter" visual (58.33 → 58.21 → 58.45),
        # dificultando a leitura. Uma casa decimal é suficiente para monitoramento.
        self._card_fps.set_value(f"{result.fps:.1f}")

        # Frame# exibido sem formatação especial — é um inteiro sequencial simples.
        self._card_frame.set_value(str(result.frame_id))

        # Extrai o estado da mão do dicionário retornado por classify_hand_state().
        # Chaves esperadas: "hand_open" (bool) e "closed_count" (int, 0–5).
        hand_open: bool = result.hand_state.get("hand_open", True)
        closed_count: int = result.hand_state.get("closed_count", 0)

        # Propaga os dados para o cartão especializado de estado.
        self._card_hand.update_state(
            hand_open=hand_open,
            closed_count=closed_count,
            hand_detected=result.hand_detected,
        )

    # =========================================================================
    # ATUALIZAÇÃO DAS MÉTRICAS DO SISTEMA (CPU e RAM)
    # =========================================================================

    def _update_system_stats(self) -> None:
        """
        Coleta e exibe as métricas de desempenho do sistema operacional.

        Chamado pelo QTimer a cada 1000ms — não está vinculado ao processamento
        de frames. CPU e RAM são recursos do sistema, não da câmera.

        Degradação graciosa:
            Se psutil não estiver instalado, exibe "—" nos cartões
            sem lançar exceção. Isso permite que a aplicação funcione
            corretamente em ambientes onde psutil não está disponível,
            apenas sem o monitoramento de recursos.

        Por que interval=None em cpu_percent()?
            psutil.cpu_percent(interval=N) seria BLOQUEANTE por N segundos.
            Com interval=None, retorna o valor calculado desde a última chamada,
            sem bloquear. Como chamamos a cada 1s via QTimer, o intervalo
            efetivo é sempre ~1 segundo — ideal para monitoramento.
        """
        if not _PSUTIL_AVAILABLE:
            # psutil não instalado — exibe placeholder sem erro.
            self._card_cpu.set_value("—")
            self._card_ram.set_value("—")
            return

        try:
            # Percentual de uso do CPU (média de todos os núcleos).
            # interval=None: não-bloqueante, usa o intervalo desde a última chamada.
            cpu_percent: float = psutil.cpu_percent(interval=None)
            self._card_cpu.set_value(f"{cpu_percent:.0f}%")

            # Memória RAM em uso, convertida de bytes para Gigabytes.
            # 1024**3 = 1 GiB. Usamos 1 casa decimal para precisão adequada.
            ram_bytes: int = psutil.virtual_memory().used
            ram_gb: float = ram_bytes / (1024 ** 3)
            self._card_ram.set_value(f"{ram_gb:.1f} GB")

        except Exception as exc:
            # Captura qualquer erro inesperado do psutil (ex: permissão negada
            # em alguns sistemas Linux com restrições de acesso a /proc).
            # Não propagamos o erro para não interromper o loop do QTimer.
            self._card_cpu.set_value("!")
            self._card_ram.set_value("!")

    # =========================================================================
    # CONTROLE DO TIMER
    # =========================================================================

    def start_monitoring(self) -> None:
        """
        Inicia ou reinicia o timer de monitoramento de CPU e RAM.

        Chamado pela MainWindow ao iniciar uma sessão, caso o timer
        tenha sido parado anteriormente por stop_monitoring().
        """
        if not self._stats_timer.isActive():
            self._stats_timer.start()

    def stop_monitoring(self) -> None:
        """
        Para o timer de monitoramento de CPU e RAM.

        Pode ser chamado pela MainWindow ao encerrar a sessão para reduzir
        o overhead de CPU quando a aplicação está em estado STOPPED ou IDLE.
        O timer pode ser reiniciado com start_monitoring() a qualquer momento.
        """
        if self._stats_timer.isActive():
            self._stats_timer.stop()

    def reset_display(self) -> None:
        """
        Redefine todos os cartões para o estado inicial "sem dados" (—).

        Chamado pela MainWindow ao iniciar uma nova sessão para limpar
        os valores da sessão anterior, evitando que dados antigos sejam
        confundidos com dados da nova sessão durante o warmup inicial.
        """
        self._card_fps.set_value("—")
        self._card_frame.set_value("—")
        self._card_hand.update_state(
            hand_open=True,
            closed_count=0,
            hand_detected=False,
        )
