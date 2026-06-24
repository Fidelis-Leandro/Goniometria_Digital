"""
ui/main_window.py — Janela principal e orquestrador do sistema
==============================================================

Este módulo implementa a MainWindow: o maestro da aplicação PyQt6 de goniometria.
Ela instancia todos os componentes, conecta os sinais entre threads e controla
o ciclo de vida completo (abertura → sessão → encerramento → relatório).

Responsabilidades da MainWindow:
    1. Instanciar e dispor visualmente todos os widgets da interface.
    2. Instanciar os workers (CameraWorker, ProcessingWorker) SEM iniciá-los.
    3. Conectar sinais de workers → widgets de forma thread-safe.
    4. Gerenciar a máquina de estados (IDLE → READY → RUNNING → STOPPED).
    5. Controlar o ciclo de vida dos workers (start, stop, wait).
    6. Gerar relatório PDF sem travar a interface (thread separada).
    7. Fechar a aplicação de forma limpa ao fechar a janela.

Máquina de estados:
    IDLE    → Estado inicial. Câmera não iniciada. Formulário editável.
    READY   → Nome do paciente preenchido. Botão Iniciar habilitado.
    RUNNING → Câmera e processamento ativos. Encerrar habilitado.
    STOPPED → Sessão encerrada. PDF, CSV e Histórico habilitados.

Fluxo de dados (thread-safe via pyqtSignal):
    CameraWorker ──frame_ready──► ProcessingWorker (via put_frame, fila)
    ProcessingWorker ──result_ready──► MainWindow._on_result()
    MainWindow._on_result() ──distribui──► VideoWidget, MetricsWidget,
                                           PlotWidget, FingerCardsPanel
"""

import logging
import os
import shutil
from datetime import datetime
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

import config
from session_report import generate_pdf_report
from themes import (
    BUTTON_DANGER_STYLE,
    BUTTON_PRIMARY_STYLE,
    COLOR_ACCENT,
    COLOR_BG_DARK,
    COLOR_BG_MEDIUM,
)

# Importa os workers.
from workers.camera_worker import CameraWorker
from workers.processing_worker import ProcessingResult, ProcessingWorker

# Importa os widgets da interface.
from ui.finger_card_widget import FingerCardsPanel
from ui.log_widget import LogWidget
from ui.metrics_widget import MetricsWidget
from ui.plot_widget import GoniometryPlotWidget
from ui.session_header import SessionHeaderWidget
from ui.video_widget import VideoWidget

# Logger específico para este módulo — facilita rastrear eventos da janela.
logger = logging.getLogger(__name__)


# =============================================================================
# WORKER AUXILIAR PARA GERAÇÃO DO PDF
# =============================================================================

class _PdfGeneratorWorker(QThread):
    """
    Thread auxiliar para geração do relatório PDF sem bloquear a interface.

    Por que generate_pdf_report() precisa de thread separada?
        A função generate_pdf_report() em session_report.py é PESADA:
        - Lê e parseia o arquivo CSV inteiro (pode ter milhares de linhas).
        - Gera múltiplos gráficos com Matplotlib (100–400ms cada).
        - Monta o PDF com FPDF (inclui renderização de imagens).
        Em hardware lento, este processo pode levar 5–15 segundos.
        Se rodasse na thread principal, a janela ficaria completamente
        congelada por esse tempo — o sistema operacional mostraria
        "Aplicação não responde" e o usuário poderia fechar à força.

        Rodando em QThread:
        - A janela permanece responsiva durante todo o processo.
        - O usuário pode ver a barra de progresso ou o log atualizando.
        - O sinal finished_signal notifica a MainWindow quando o PDF
          está pronto, de forma thread-safe.

    Sinais:
        finished_signal(str): Caminho do PDF gerado quando concluído com sucesso.
        error_signal(str): Mensagem de erro se a geração falhar.
    """

    # Carrega o caminho do PDF gerado ao terminar com sucesso.
    finished_signal: pyqtSignal = pyqtSignal(str)

    # Carrega a mensagem de erro se a geração falhar.
    error_signal: pyqtSignal = pyqtSignal(str)

    def __init__(
        self,
        csv_path: str,
        patient_name: str,
        side: str,
        logo_path: Optional[str] = None,
        parent=None,
    ) -> None:
        """
        Configura os parâmetros da geração do PDF.

        Parâmetros:
            csv_path: Caminho do CSV gerado pela sessão encerrada.
            patient_name: Nome do paciente para o relatório.
            side: Mão avaliada ("Direita" ou "Esquerda").
            logo_path: Caminho do logo institucional (opcional).
            parent: Widget pai Qt (opcional).
        """
        super().__init__(parent)
        self._csv_path = csv_path
        self._patient_name = patient_name
        self._side = side
        self._logo_path = logo_path

    def run(self) -> None:
        """
        Executa a geração do PDF na thread separada.

        Chama generate_pdf_report() com os parâmetros configurados e
        emite finished_signal com o caminho do PDF ou error_signal com
        a descrição do erro. Nunca lança exceção para fora do QThread —
        erros são comunicados via sinal.
        """
        try:
            pdf_path: str = generate_pdf_report(
                csv_path=self._csv_path,
                patient_name=self._patient_name,
                side=self._side,
                logo_path=self._logo_path,
            )
            # Emite o caminho do PDF gerado para a MainWindow exibir no diálogo.
            self.finished_signal.emit(pdf_path)

        except Exception as exc:
            # Captura qualquer erro (CSV vazio, Matplotlib falhou, disco cheio etc.)
            # e notifica a MainWindow via sinal thread-safe.
            logger.error("Falha ao gerar PDF: %s", exc, exc_info=True)
            self.error_signal.emit(str(exc))


# =============================================================================
# JANELA PRINCIPAL
# =============================================================================

class MainWindow(QMainWindow):
    """
    Janela principal da aplicação de Goniometria Digital.

    Implementa o padrão "Controller" do MVC: concentra toda a lógica de
    coordenação em um único lugar, enquanto cada widget tem responsabilidade
    única. A MainWindow NÃO faz processamento científico — apenas conecta
    quem produz dados (workers) com quem exibe dados (widgets).

    Estado atual da aplicação:
        self._state: str — um dos valores: "IDLE", "READY", "RUNNING", "STOPPED"
        self._csv_path: str — caminho do CSV da sessão ativa (vazio se nenhuma)
        self._pdf_worker: QThread auxiliar para geração do PDF (ou None)
    """

    def __init__(self, parent=None) -> None:
        """
        Inicializa a janela principal: widgets, workers, sinais e estado inicial.

        Ordem de inicialização:
            1. Configuração da janela (título, tamanho mínimo).
            2. Criação de todos os widgets da interface.
            3. Instanciação dos workers (SEM iniciar as threads).
            4. Conexão de todos os sinais entre workers e widgets.
            5. Montagem do layout visual.
            6. Definição do estado inicial "IDLE".

        Por que instanciar workers no __init__ mas não iniciá-los?
            Os workers precisam existir para que seus sinais possam ser conectados.
            Mas iniciar as threads (start()) antes do usuário clicar "Iniciar Sessão"
            desperdiçaria recursos de CPU e câmera mesmo quando a aplicação está ociosa.

        Parâmetros:
            parent: Widget pai Qt (opcional). Geralmente None para janela principal.
        """
        super().__init__(parent)

        # --- Título e tamanho da janela ---
        self.setWindowTitle(config.APP_TITLE)
        self.setMinimumSize(config.WINDOW_MIN_WIDTH, config.WINDOW_MIN_HEIGHT)
        self.setStyleSheet(f"QMainWindow {{ background-color: {COLOR_BG_DARK}; }}")

        # --- Estado interno ---
        # String que controla quais botões estão habilitados e o comportamento
        # de cada método. Centralizar em uma variável evita lógica espalhada.
        self._state: str = "IDLE"

        # Caminho do CSV da sessão ativa. Definido em _iniciar_sessao().
        # Usado por _gerar_relatorio() e _exportar_csv().
        self._csv_path: str = ""

        # Worker de geração de PDF — mantemos referência para evitar garbage collection
        # antes de o PDF terminar de ser gerado.
        self._pdf_worker: Optional[_PdfGeneratorWorker] = None

        # === CRIAÇÃO DOS WIDGETS ===
        self._create_widgets()

        # === INSTANCIAÇÃO DOS WORKERS ===
        self._create_workers()

        # === CONEXÃO DOS SINAIS ===
        self._connect_signals()

        # === MONTAGEM DO LAYOUT ===
        self._build_layout()

        # === BARRA DE STATUS ===
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Pronto. Preencha os dados do paciente para iniciar.")

        # === ESTADO INICIAL ===
        # Começa em IDLE: nenhum botão habilitado exceto o formulário.
        self._set_state("IDLE")

        logger.info("MainWindow inicializada com sucesso.")

    # =========================================================================
    # CRIAÇÃO DOS COMPONENTES
    # =========================================================================

    def _create_widgets(self) -> None:
        """
        Instancia todos os widgets da interface.

        Mantido separado do __init__ para melhor organização e facilitar
        testes unitários dos widgets individualmente.
        """
        # Cabeçalho com formulário do paciente e cronômetro.
        self.session_header = SessionHeaderWidget()

        # Exibição do frame de vídeo com overlay goniométrico.
        self.video_widget = VideoWidget()

        # Painel lateral com cartões de métricas do sistema.
        self.metrics_widget = MetricsWidget()

        # Gráfico de TAM ao vivo para os 5 dedos.
        self.plot_widget = GoniometryPlotWidget()

        # Painel com 5 cartões clínicos individuais por dedo.
        self.finger_cards = FingerCardsPanel()

        # Log de eventos do sistema com timestamps.
        self.log_widget = LogWidget()

        # --- Botões de controle da sessão ---

        # Botão Nova Sessão — para reiniciar o sistema a qualquer momento
        self.btn_nova_sessao = QPushButton("🔄  Nova Sessão")
        self.btn_nova_sessao.setToolTip("Zera todos os dados da tela para permitir o registro de um novo paciente ou teste limpo.")

        # Botão principal de início — estilo verde destacado.
        self.btn_iniciar = QPushButton("▶  Iniciar Sessão")
        self.btn_iniciar.setStyleSheet(BUTTON_PRIMARY_STYLE)
        self.btn_iniciar.setToolTip("Inicia a captura de vídeo e o processamento goniométrico.")

        # Botão de encerramento — estilo vermelho para ação destrutiva/final.
        self.btn_encerrar = QPushButton("■  Encerrar Sessão")
        self.btn_encerrar.setStyleSheet(BUTTON_DANGER_STYLE)
        self.btn_encerrar.setToolTip("Encerra a captura e finaliza o arquivo CSV.")

        # Botões de pós-processamento — estilos padrão do tema.
        self.btn_pdf = QPushButton("📄  Gerar Relatório PDF")
        self.btn_pdf.setToolTip("Gera o relatório clínico PDF a partir do CSV da sessão encerrada.")

        self.btn_csv = QPushButton("💾  Exportar CSV")
        self.btn_csv.setToolTip("Copia o arquivo CSV da sessão para um local escolhido.")

        self.btn_historico = QPushButton("📁  Abrir Pasta de Sessões")
        self.btn_historico.setToolTip("Abre a pasta onde os arquivos de sessão são salvos.")

    def _create_workers(self) -> None:
        """
        Instancia os workers de câmera e processamento SEM iniciar as threads.

        Os workers são criados aqui para que seus sinais possam ser conectados
        em _connect_signals(). As threads só iniciam quando o usuário clicar
        em "Iniciar Sessão" — não antes.
        """
        self.camera_worker = CameraWorker(parent=self)
        self.processing_worker = ProcessingWorker(parent=self)

    def _connect_signals(self) -> None:
        """
        Conecta todos os sinais entre workers, widgets e métodos da MainWindow.

        Por que centralizar aqui?
            Manter todas as conexões em um único método cria uma "tabela de roteamento"
            legível do sistema. Ao depurar um bug de comunicação entre componentes,
            basta olhar aqui para ver quem fala com quem.

        Conexões estabelecidas:
            camera_worker.frame_ready   → processing_worker.put_frame()
            camera_worker.fps_updated   → video_widget.set_fps()
            camera_worker.camera_error  → _on_camera_error()
            processing_worker.result_ready   → _on_result()
            processing_worker.processing_error → log_widget.log()
            session_header textChanged  → _on_patient_name_changed()
            btn_* .clicked              → slots de ação
        """
        # --- Camera Worker → Processing Worker ---
        # AutoConnection (padrão): o Qt detecta automaticamente que os dois estão
        # em threads diferentes e usa uma conexão queued (thread-safe).
        # NÃO usamos DirectConnection aqui porque seria chamado diretamente na
        # thread do CameraWorker, e put_frame() acessa a Queue — operação segura,
        # mas DirectConnection é desnecessário quando AutoConnection funciona.
        self.camera_worker.frame_ready.connect(self.processing_worker.put_frame)

        # --- Camera Worker → VideoWidget (FPS) ---
        self.camera_worker.fps_updated.connect(self.video_widget.set_fps)

        # --- Camera Worker → tratamento de erros ---
        self.camera_worker.camera_error.connect(self._on_camera_error)

        # --- Processing Worker → MainWindow (resultado principal) ---
        self.processing_worker.result_ready.connect(self._on_result)

        # --- Processing Worker → LogWidget (erros não fatais) ---
        self.processing_worker.processing_error.connect(self.log_widget.log_error)

        # --- Formulário → verificação de prontidão ---
        # Cada vez que o texto do campo de nome muda, verificamos se o botão
        # Iniciar deve ser habilitado. textChanged dispara para cada caractere.
        self.session_header._input_patient.textChanged.connect(
            self._on_patient_name_changed
        )

        # --- Botões → ações ---
        self.btn_nova_sessao.clicked.connect(self._nova_sessao)
        self.btn_iniciar.clicked.connect(self._iniciar_sessao)
        self.btn_encerrar.clicked.connect(self._encerrar_sessao)
        self.btn_pdf.clicked.connect(self._gerar_relatorio)
        self.btn_csv.clicked.connect(self._exportar_csv)
        self.btn_historico.clicked.connect(self._abrir_historico)

    # =========================================================================
    # MONTAGEM DO LAYOUT
    # =========================================================================

    def _build_layout(self) -> None:
        """
        Monta a hierarquia de widgets e layouts na janela principal.

        Layout final (QVBoxLayout central):
            1. SessionHeaderWidget          — topo, altura fixa
            2. QHBoxLayout:
               VideoWidget (stretch=3) │ MetricsWidget (stretch=2)
            3. GoniometryPlotWidget         — altura fixa 220px
            4. FingerCardsPanel             — altura fixa
            5. LogWidget                    — altura máxima 120px
            6. QHBoxLayout de botões        — altura fixa

        Por que usar stretch factors no QHBoxLayout do vídeo?
            stretch=3 para o vídeo e stretch=2 para as métricas resulta em
            proporção 60%/40%, que na maioria dos monitores dá ~768px/512px.
            Isso mantém o vídeo grande o suficiente para ver o overlay sem
            reduzir as métricas a ponto de ficarem ilegíveis.
        """
        # =========================================================
        # 1. Configuração do ScrollArea Central
        # =========================================================
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(f"QScrollArea {{ border: none; background: {COLOR_BG_DARK}; }}")
        self.setCentralWidget(scroll_area)

        # Widget container que vai dentro da ScrollArea.
        container_widget = QWidget()
        scroll_area.setWidget(container_widget)

        # Layout vertical principal.
        main_layout = QVBoxLayout(container_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Aumentamos o tamanho mínimo da janela do vídeo e métricas
        # para garantir ótima visualização. O scroll cuidará do restante.
        self.video_widget.setMinimumHeight(450)
        self.metrics_widget.setMinimumHeight(450)

        # --- 1. Cabeçalho da sessão ---
        main_layout.addWidget(self.session_header)

        # --- 2. Linha do meio: Vídeo + Métricas ---
        mid_row = QHBoxLayout()
        mid_row.setSpacing(6)

        # Vídeo: ocupa ~60% da largura da linha do meio.
        mid_row.addWidget(self.video_widget, stretch=3)

        # Métricas: ocupa ~40% da largura da linha do meio.
        mid_row.addWidget(self.metrics_widget, stretch=2)

        main_layout.addLayout(mid_row)

        # --- 3. Gráfico TAM ao vivo ---
        # Aumentamos a altura do gráfico para facilitar leitura
        self.plot_widget.setMinimumHeight(280)
        main_layout.addWidget(self.plot_widget)

        # --- 4. Cartões individuais por dedo ---
        # Mantemos o scroll interno dos cartões (se a tela for muito estreita)
        # ou apenas altura fixa para eles.
        cards_scroll = QScrollArea()
        cards_scroll.setWidget(self.finger_cards)
        cards_scroll.setWidgetResizable(True)
        cards_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        cards_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        cards_scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {COLOR_BG_DARK}; }}"
        )
        cards_scroll.setFixedHeight(310)

        main_layout.addWidget(cards_scroll)

        # --- 5. Log de eventos ---
        self.log_widget.setMinimumHeight(150)
        main_layout.addWidget(self.log_widget)

        # --- 6. Barra de botões ---
        main_layout.addLayout(self._build_button_row())

    def _build_button_row(self) -> QHBoxLayout:
        """
        Constrói a linha horizontal com todos os botões de controle.

        Ordem dos botões:
            [Iniciar] [Encerrar] | [Gerar PDF] [Exportar CSV] [Abrir Pasta]

        O separador visual (stretch) entre os dois grupos distingue
        ações de sessão (esquerda) de ações de exportação (direita).

        Retorno:
            QHBoxLayout pronto para ser adicionado ao layout principal.
        """
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        # Grupo esquerdo: controle de sessão.
        btn_row.addWidget(self.btn_nova_sessao)
        btn_row.addWidget(self.btn_iniciar)
        btn_row.addWidget(self.btn_encerrar)

        # Separador elástico entre grupos de botões.
        btn_row.addStretch()

        # Grupo direito: exportação e navegação.
        btn_row.addWidget(self.btn_pdf)
        btn_row.addWidget(self.btn_csv)
        btn_row.addWidget(self.btn_historico)

        return btn_row

    # =========================================================================
    # MÁQUINA DE ESTADOS
    # =========================================================================

    def _set_state(self, state: str) -> None:
        """
        Centraliza a habilitação e desabilitação de botões conforme o estado.

        Por que centralizar em _set_state() em vez de habilitar/desabilitar
        botões diretamente em cada método de ação?

            Sem centralização, cada método (iniciar, encerrar, etc.) precisaria
            conhecer e manipular TODOS os botões. Se um novo botão for adicionado
            no futuro, seria necessário atualizar TODOS os métodos. Com _set_state(),
            basta adicionar o novo botão aqui e ele estará corretamente gerenciado
            em todos os estados sem alterar mais nada.

            É o padrão "State Machine" aplicado à UI: o estado é a fonte de verdade,
            e os botões são consequência do estado — não o contrário.

        Transições válidas:
            IDLE    → READY  (quando nome é preenchido)
            READY   → RUNNING (ao clicar Iniciar)
            RUNNING → STOPPED (ao clicar Encerrar)
            STOPPED → READY  (ao preencher nome para nova sessão)

        Parâmetros:
            state: String identificando o novo estado.
                   Valores válidos: "IDLE", "READY", "RUNNING", "STOPPED".
        """
        self._state = state

        # --- Botões de controle de sessão ---
        # Nova Sessão: disponível apenas após o encerramento de uma sessão.
        self.btn_nova_sessao.setEnabled(state == "STOPPED")

        # Iniciar: habilitado apenas quando há dados para iniciar (READY).
        self.btn_iniciar.setEnabled(state == "READY")

        # Encerrar: habilitado apenas durante a gravação ativa (RUNNING).
        self.btn_encerrar.setEnabled(state == "RUNNING")

        # --- Botões de exportação ---
        # Somente habilitados após o encerramento formal da sessão (STOPPED).
        # Em RUNNING, o CSV ainda está sendo gravado — exportar seria inconsistente.
        self.btn_pdf.setEnabled(state == "STOPPED")
        self.btn_csv.setEnabled(state == "STOPPED")
        self.btn_historico.setEnabled(state in ("IDLE", "READY", "STOPPED"))

        # --- Campos do formulário ---
        # Bloqueados durante RUNNING para evitar alteração acidental dos dados
        # que identificam a sessão enquanto ela está sendo gravada.
        self.session_header.set_fields_enabled(state != "RUNNING")

        # --- Mensagem na barra de status ---
        status_messages = {
            "IDLE":    "Preencha os dados do paciente para habilitar o início da sessão.",
            "READY":   "Pronto para iniciar. Clique em 'Iniciar Sessão'.",
            "RUNNING": "Sessão em andamento — capturando e processando dados...",
            "STOPPED": "Sessão encerrada. Você pode gerar o relatório PDF ou exportar o CSV.",
        }
        self._status_bar.showMessage(status_messages.get(state, ""))

        logger.debug("Estado alterado para: %s", state)

    # =========================================================================
    # SLOTS DE RESULTADO E ERRO
    # =========================================================================

    def _on_result(self, result: object) -> None:
        """
        Recebe e distribui o ProcessingResult para todos os widgets.

        Chamado ~30 vezes/segundo pelo sinal result_ready do ProcessingWorker.
        Deve ser rápido: apenas distribui dados, sem nenhum cálculo.

        Este método roda na thread principal (thread do Qt), garantido pelo
        sistema de sinais do Qt. Nunca acessamos widgets de dentro de uma
        QThread — sempre via este slot conectado ao sinal.

        Parâmetros:
            result: objeto ProcessingResult emitido pelo ProcessingWorker.
                    Tipado como 'object' porque pyqtSignal(ProcessingResult)
                    não é suportado diretamente — fazemos cast aqui.
        """
        # Cast para o tipo correto — seguro porque só o ProcessingWorker
        # emite result_ready e sempre com ProcessingResult.
        r: ProcessingResult = result  # type: ignore[assignment]

        # Atualiza o frame de vídeo com o overlay goniométrico.
        self.video_widget.update_frame(r.frame_overlay)

        # Atualiza os cartões de FPS, Frame# e estado da mão.
        self.metrics_widget.update_from_result(r)

        # Atualiza o gráfico de TAM ao vivo dos 5 dedos.
        self.plot_widget.update_data(r.angles_smooth, r.hand_detected)

        # Atualiza os 5 cartões individuais com métricas clínicas e mini-gráficos.
        # get_tam_buffers() retorna uma cópia thread-safe dos deques do worker.
        self.finger_cards.update_all(
            finger_states=r.hand_state.get("finger_states", {}),
            metrics_per_finger=r.metrics_per_finger,
            tam_buffers_per_finger=r.tam_buffers_snapshot,
        )

    def _on_camera_error(self, message: str) -> None:
        """
        Trata erros fatais da câmera emitidos pelo CameraWorker.

        Quando a câmera para de funcionar durante uma sessão ativa,
        encerra a sessão automaticamente para evitar a gravação de
        frames inválidos no CSV. Exibe o erro no log e na barra de status.

        Parâmetros:
            message: Descrição do erro enviada pelo CameraWorker via sinal.
        """
        self.log_widget.log_error(f"Câmera: {message}")
        self.video_widget.set_no_signal(message)
        self._status_bar.showMessage(f"ERRO DE CÂMERA: {message}")
        logger.error("Erro de câmera: %s", message)

        # Se a sessão estava em andamento, encerra automaticamente.
        # Continuar gravando sem frames seria criar um CSV corrompido.
        if self._state == "RUNNING":
            self._encerrar_sessao()

    def _on_patient_name_changed(self, text: str) -> None:
        """
        Reage à mudança do nome do paciente no formulário.

        Chamado pelo sinal textChanged do QLineEdit de nome a cada caractere.
        Verifica se o formulário está pronto (is_ready()) e transita entre
        os estados IDLE e READY conforme o preenchimento.

        Parâmetros:
            text: Conteúdo atual do campo de nome (string bruta, com espaços).
        """
        # Só altera o estado se não estiver em RUNNING ou STOPPED.
        # Não queremos que a digitação do nome altere o estado durante uma sessão.
        if self._state in ("IDLE", "READY"):
            if self.session_header.is_ready():
                self._set_state("READY")
            else:
                self._set_state("IDLE")

    # =========================================================================
    # AÇÕES DOS BOTÕES
    # =========================================================================

    def _nova_sessao(self) -> None:
        """
        Zera completamente o estado do sistema e prepara a interface para um novo paciente.

        Por que recriar os workers em vez de apenas resetar?
            QThread em Qt tem um ciclo de vida unidirecional: uma vez que run()
            retorna e a thread termina, o objeto QThread não pode ser reiniciado
            com start() novamente. Além disso, _cleanup() no ProcessingWorker
            fecha o MediaPipe (self._hands.close()), e _release_camera() no
            CameraWorker libera o cv2.VideoCapture. Esses recursos precisam ser
            recriados do zero para uma nova sessão funcionar.

            A solução segura é: destruir os workers antigos, criar novos e
            reconectar todos os sinais.
        """
        if self._state == "RUNNING":
            QMessageBox.warning(self, "Aviso", "Encerre a sessão atual primeiro antes de iniciar uma nova.")
            return

        # Confirmação do usuário para evitar perda acidental de dados na tela
        resp = QMessageBox.question(
            self,
            "Atenção: Nova Sessão",
            "Iniciar uma nova sessão apagará todos os gráficos e métricas atuais da tela.\n\n"
            "Certifique-se de já ter exportado o CSV ou gerado o Relatório PDF se precisar destes dados.\n\n"
            "Tem certeza de que deseja iniciar do zero?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No  # Default para Não por segurança
        )
        if resp != QMessageBox.StandardButton.Yes:
            return

        # === 1. Garantir que os workers antigos estão totalmente parados ===
        if self.camera_worker.isRunning():
            self.camera_worker.stop()
            self.camera_worker.wait(3000)

        if self.processing_worker.isRunning():
            self.processing_worker.stop()
            self.processing_worker.wait(3000)

        # === 2. Recriar workers do zero (novos objetos QThread) ===
        self._create_workers()

        # === 3. Reconectar todos os sinais com os novos workers ===
        self._connect_signals()

        # === 4. Limpar todos os widgets da interface ===
        self.video_widget.set_no_signal("Aguardando nova sessão...")
        self.plot_widget.clear_data()
        self.finger_cards.clear_all()
        self.metrics_widget.reset_display()
        self.metrics_widget.stop_monitoring()
        self.session_header.reset()

        # === 5. Limpar o log e exibir confirmação ===
        self.log_widget.clear_log()
        self.log_widget.log_success("Sistema reiniciado. Pronto para uma nova sessão.")

        # === 6. Zerar referências de sessão ===
        self._csv_path = ""
        self._pdf_worker = None

        # === 7. Reavalia o estado ===
        if self.session_header.is_ready():
            self._set_state("READY")
        else:
            self._set_state("IDLE")

    def _iniciar_sessao(self) -> None:
        """
        Inicia a sessão de captura e processamento goniométrico.

        Sequência de operações:
            1. Valida se o formulário está completo (is_ready()).
            2. Cria o diretório de logs se não existir.
            3. Gera o caminho do CSV com timestamp para unicidade.
            4. Inicia a sessão no ProcessingWorker (abre o CSV).
            5. Inicia as threads dos workers.
            6. Atualiza o cronômetro e o log.
            7. Transita para o estado RUNNING.

        Chamado pelo botão "Iniciar Sessão" (btn_iniciar).
        """
        # Validação de pré-condição — linha de defesa extra além do estado READY.
        if not self.session_header.is_ready():
            QMessageBox.warning(
                self,
                "Dados Incompletos",
                "Por favor, preencha o nome completo do paciente antes de iniciar.",
            )
            return

        # Coleta os dados do formulário para uso no CSV e no PDF.
        session_info = self.session_header.get_session_info()
        patient_name: str = session_info["patient_name"]
        hand: str = session_info["hand"]
        session_number: int = session_info["session_number"]

        # Cria a pasta de logs se não existir.
        # exist_ok=True: não lança erro se a pasta já existe.
        os.makedirs(config.LOG_DIR, exist_ok=True)

        # Gera o nome do arquivo CSV com timestamp para garantir unicidade.
        # Formato: "logs/sessao_João_Silva_20260623_143512_s1.csv"
        # O timestamp evita sobrescrever sessões anteriores do mesmo paciente.
        timestamp_str: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Remove caracteres inválidos em nomes de arquivo (espaços → underscores).
        safe_name: str = patient_name.replace(" ", "_").replace("/", "_")
        csv_filename: str = f"sessao_{safe_name}_{timestamp_str}_s{session_number}.csv"
        self._csv_path = os.path.join(config.LOG_DIR, csv_filename)

        # Inicia a sessão CSV no worker ANTES de iniciar as threads.
        # Isso garante que o logger está pronto quando os primeiros frames chegarem.
        self.processing_worker.start_session(self._csv_path)

        # Limpa os dados dos widgets da sessão anterior.
        self.plot_widget.clear_data()
        self.finger_cards.clear_all()
        self.metrics_widget.reset_display()

        # Inicia as threads dos workers.
        # start() do QThread chama run() na thread separada.
        # Os workers já existem desde o __init__ — apenas iniciamos a execução.
        if not self.camera_worker.isRunning():
            self.camera_worker.start()

        if not self.processing_worker.isRunning():
            self.processing_worker.start()

        # Inicia o cronômetro no cabeçalho.
        self.session_header.start_timer()

        # Registra o início no log e transita para RUNNING.
        self.log_widget.log_success(
            f"Sessão {session_number} iniciada — {patient_name} | Mão {hand} | {csv_filename}"
        )

        self._set_state("RUNNING")
        logger.info(
            "Sessão iniciada: paciente=%s, mão=%s, sessão=%d, csv=%s",
            patient_name, hand, session_number, self._csv_path,
        )

    def _encerrar_sessao(self) -> None:
        """
        Encerra a sessão de captura e finaliza o arquivo CSV.

        Sequência de operações:
            1. Sinaliza o ProcessingWorker para encerrar a sessão CSV.
            2. Sinaliza o CameraWorker para parar o loop de captura.
            3. Para o cronômetro.
            4. Registra no log.
            5. Transita para o estado STOPPED.

        Chamado pelo botão "Encerrar Sessão" (btn_encerrar) OU automaticamente
        por _on_camera_error() quando a câmera falha durante uma sessão ativa.

        Atenção: não usa wait() aqui para não bloquear a thread principal.
        O closeEvent() usa wait() com timeout quando a janela é fechada.
        """
        # Encerra a gravação CSV de forma segura (flush + close).
        self.processing_worker.stop_session()

        # Sinaliza o CameraWorker para parar o loop de captura.
        # O loop em run() verificará o evento na próxima iteração.
        self.camera_worker.stop()

        # Para o cronômetro — o display congela no tempo total da sessão.
        self.session_header.stop_timer()

        self.log_widget.log_success(
            f"Sessão encerrada. CSV salvo em: {self._csv_path}"
        )
        self._set_state("STOPPED")
        logger.info("Sessão encerrada. CSV: %s", self._csv_path)

    def _gerar_relatorio(self) -> None:
        """
        Gera o relatório PDF da sessão encerrada em uma thread separada.

        Por que thread separada? (Ver _PdfGeneratorWorker para explicação completa)
        Em resumo: generate_pdf_report() pode levar 5–15 segundos (Matplotlib +
        FPDF) e congelaria completamente a interface se rodasse na thread principal.

        Comportamento:
            - Desabilita o botão PDF durante a geração (evita cliques duplos).
            - Exibe uma mensagem na barra de status indicando o progresso.
            - Ao terminar, _on_pdf_finished() exibe o diálogo de sucesso.
            - Ao falhar, _on_pdf_error() exibe o diálogo de erro.

        Chamado pelo botão "Gerar Relatório PDF" (btn_pdf).
        """
        if not self._csv_path or not os.path.exists(self._csv_path):
            QMessageBox.warning(
                self,
                "CSV não encontrado",
                f"O arquivo CSV da sessão não foi encontrado:\n{self._csv_path}\n\n"
                "Verifique se a sessão foi encerrada corretamente.",
            )
            return

        session_info = self.session_header.get_session_info()

        # Desabilita o botão durante a geração para evitar cliques duplos.
        self.btn_pdf.setEnabled(False)
        self.btn_pdf.setText("⏳  Gerando PDF...")
        self._status_bar.showMessage("Gerando relatório PDF... Aguarde.")
        self.log_widget.log("Iniciando geração do relatório PDF...")

        # Cria e configura o worker de geração do PDF.
        self._pdf_worker = _PdfGeneratorWorker(
            csv_path=self._csv_path,
            patient_name=session_info["patient_name"],
            side=session_info["hand"],
            logo_path=os.path.abspath(os.path.join("assets", "logo_ufcspa.png")),
            parent=self,
        )

        # Conecta os sinais de conclusão e erro ao worker.
        self._pdf_worker.finished_signal.connect(self._on_pdf_finished)
        self._pdf_worker.error_signal.connect(self._on_pdf_error)

        # Inicia o worker de PDF na thread separada.
        self._pdf_worker.start()

    def _on_pdf_finished(self, pdf_path: str) -> None:
        """
        Trata a conclusão bem-sucedida da geração do PDF.

        Chamado pelo sinal finished_signal do _PdfGeneratorWorker quando
        o PDF foi gerado com sucesso. Restaura o botão e exibe o resultado.

        Parâmetros:
            pdf_path: Caminho absoluto do arquivo PDF gerado.
        """
        # Restaura o botão ao estado original.
        self.btn_pdf.setEnabled(True)
        self.btn_pdf.setText("📄  Gerar Relatório PDF")
        self._status_bar.showMessage(f"PDF gerado: {pdf_path}")
        self.log_widget.log_success(f"Relatório PDF gerado: {pdf_path}")

        # Exibe diálogo com o caminho do PDF e opção de abrir a pasta.
        msg = QMessageBox(self)
        msg.setWindowTitle("Relatório Gerado")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText("✅ Relatório PDF gerado com sucesso!")
        msg.setInformativeText(f"Arquivo salvo em:\n{pdf_path}")
        msg.exec()

    def _on_pdf_error(self, error_message: str) -> None:
        """
        Trata falha na geração do PDF.

        Chamado pelo sinal error_signal do _PdfGeneratorWorker quando
        a geração falhou (CSV inválido, disco cheio, Matplotlib ausente etc.).

        Parâmetros:
            error_message: Descrição do erro retornada pela exceção.
        """
        # Restaura o botão e reporta o erro.
        self.btn_pdf.setEnabled(True)
        self.btn_pdf.setText("📄  Gerar Relatório PDF")
        self._status_bar.showMessage("Falha ao gerar o relatório PDF.")
        self.log_widget.log_error(f"Falha ao gerar PDF: {error_message}")

        QMessageBox.critical(
            self,
            "Erro ao Gerar PDF",
            f"Não foi possível gerar o relatório PDF.\n\nDetalhe:\n{error_message}",
        )

    def _exportar_csv(self) -> None:
        """
        Exporta o CSV da sessão para um local escolhido pelo usuário.

        Abre um diálogo de arquivo para o usuário escolher onde salvar
        a cópia do CSV. Usa shutil.copy() para preservar os dados originais.
        O arquivo original em config.LOG_DIR não é movido nem apagado.

        Chamado pelo botão "Exportar CSV" (btn_csv).
        """
        if not self._csv_path or not os.path.exists(self._csv_path):
            QMessageBox.warning(
                self,
                "CSV não encontrado",
                "Não há arquivo CSV disponível para exportação.",
            )
            return

        # QFileDialog.getSaveFileName: diálogo nativo de "Salvar Como".
        # Sugere o nome do arquivo original como padrão.
        default_name = os.path.basename(self._csv_path)
        destination, selected_filter = QFileDialog.getSaveFileName(
            parent=self,
            caption="Exportar CSV da Sessão",
            directory=default_name,
            filter="Arquivos CSV (*.csv);;Todos os Arquivos (*)",
        )

        # Se o usuário cancelou o diálogo, destination é string vazia.
        if not destination:
            return

        try:
            # Copia o arquivo CSV original para o destino escolhido.
            # shutil.copy() copia conteúdo E permissões, mais robusto que open().
            shutil.copy(self._csv_path, destination)
            self.log_widget.log_success(f"CSV exportado para: {destination}")
            self._status_bar.showMessage(f"CSV exportado: {destination}")

            QMessageBox.information(
                self,
                "Exportação Concluída",
                f"✅ CSV exportado com sucesso para:\n{destination}",
            )

        except (OSError, shutil.Error) as exc:
            self.log_widget.log_error(f"Falha ao exportar CSV: {exc}")
            QMessageBox.critical(
                self,
                "Erro na Exportação",
                f"Não foi possível exportar o CSV.\n\nDetalhe:\n{exc}",
            )

    def _abrir_historico(self) -> None:
        """
        Abre a pasta onde os arquivos de sessão são salvos no explorador.

        Usa os.startfile() no Windows para abrir a pasta no Explorador de Arquivos.
        A pasta é criada se não existir antes de tentar abrir.

        Chamado pelo botão "Abrir Pasta de Sessões" (btn_historico).
        """
        # Garante que a pasta existe antes de tentar abrir.
        os.makedirs(config.LOG_DIR, exist_ok=True)
        abs_log_dir: str = os.path.abspath(config.LOG_DIR)

        try:
            # os.startfile() é exclusivo do Windows — abre com o programa padrão.
            # No Windows, abre o Explorador de Arquivos na pasta indicada.
            os.startfile(abs_log_dir)
        except AttributeError:
            # Fallback para Linux/macOS onde os.startfile() não existe.
            import subprocess
            subprocess.Popen(["xdg-open", abs_log_dir])
        except Exception as exc:
            self.log_widget.log_error(f"Não foi possível abrir a pasta: {exc}")

    # =========================================================================
    # CICLO DE VIDA DA JANELA
    # =========================================================================

    def closeEvent(self, event) -> None:
        """
        Intercepta o evento de fechamento da janela para encerrar os workers.

        Chamado pelo Qt quando o usuário clica no botão "X" da janela ou
        quando QApplication.quit() é chamado.

        Por que wait() com timeout aqui e não em _encerrar_sessao()?
            _encerrar_sessao() é chamado durante a sessão enquanto a janela
            ainda está visível — bloquear a thread principal com wait() ali
            congelaria a interface por alguns frames. Aqui, a janela já está
            sendo fechada de qualquer forma, então o bloqueio temporário é
            aceitável para garantir que os workers terminem limpos.

        Sequência:
            1. Para os workers (sinaliza encerramento).
            2. Aguarda até 3000ms por worker para terminar.
            3. Aceita o evento de fechamento (janela fecha).

        Parâmetros:
            event: QCloseEvent fornecido pelo Qt com o evento de fechamento.
        """
        logger.info("closeEvent: encerrando workers antes de fechar.")

        # Encerra a sessão ativa se houver uma.
        if self._state == "RUNNING":
            self.processing_worker.stop_session()

        # Para e aguarda o CameraWorker.
        if self.camera_worker.isRunning():
            self.camera_worker.stop()
            # wait(3000): aguarda até 3 segundos. Se o worker não terminar,
            # o Qt encerrará a thread de forma forçada ao fechar a aplicação.
            self.camera_worker.wait(3000)

        # Para e aguarda o ProcessingWorker.
        if self.processing_worker.isRunning():
            self.processing_worker.stop()
            self.processing_worker.wait(3000)

        # Para o worker de PDF se estiver gerando.
        if self._pdf_worker is not None and self._pdf_worker.isRunning():
            self._pdf_worker.wait(5000)

        logger.info("closeEvent: workers encerrados. Fechando janela.")

        # Aceita o evento — a janela fecha normalmente.
        event.accept()
