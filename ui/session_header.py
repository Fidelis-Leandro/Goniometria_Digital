"""
ui/session_header.py — Cabeçalho da sessão clínica com formulário e cronômetro
===============================================================================

Este módulo implementa o SessionHeaderWidget: a barra superior da interface que
coleta os dados identificadores da sessão clínica ANTES de iniciar a câmera.

Responsabilidade:
    1. Formulário de identificação: nome do paciente, mão avaliada, número da sessão.
    2. Controle do cronômetro: registra o horário de início e exibe o tempo decorrido.
    3. Validação de pré-condição: is_ready() garante que a sessão só inicie com
       dados mínimos preenchidos (nome do paciente obrigatório).

Por que coletar esses dados aqui e não no início do processamento?
    O CSV gerado pelo GoniometryCSVLogger e o PDF do session_report.py precisam
    do nome do paciente, da mão avaliada e do número da sessão nos metadados.
    Coletar antes de iniciar garante que esses campos sempre estejam disponíveis
    quando o ProcessingWorker começar a gravar — sem campos em branco no arquivo.

Fluxo de uso na MainWindow:
    1. Usuário preenche o formulário.
    2. MainWindow chama is_ready() — fica verificando via QLineEdit.textChanged.
    3. Quando pronto, botão Iniciar é habilitado.
    4. Ao clicar Iniciar: MainWindow chama get_session_info() e start_timer().
    5. Ao clicar Encerrar: MainWindow chama stop_timer().
"""

from datetime import datetime, timedelta
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from themes import (
    COLOR_ACCENT,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    LABEL_SECTION_TITLE_STYLE,
    SESSION_HEADER_STYLE,
)


class SessionHeaderWidget(QWidget):
    """
    Barra superior da interface de goniometria com formulário e cronômetro.

    Exibida permanentemente no topo da MainWindow, independente do estado da
    sessão (IDLE, READY, RUNNING, STOPPED). Os campos do formulário ficam
    editáveis em IDLE/READY e bloqueados em RUNNING/STOPPED para evitar
    alterações acidentais durante a gravação.

    Layout visual:
        ┌─────────────────────────────────────────────────────────────────────┐
        │ Paciente: [___________________] Mão: [▾] Sessão: [▲1▼] │Início: 14:35│
        │                                                          │Decorrido: 00:12:48│
        └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Inicializa o cabeçalho com todos os campos do formulário e o cronômetro.

        O QTimer interno (_timer) é criado aqui mas NÃO iniciado —
        é ativado apenas quando start_timer() é chamado pela MainWindow.
        Isso garante que o cronômetro não comece a rodar antes da sessão.

        Parâmetros:
            parent: Widget pai Qt (opcional). Geralmente a MainWindow.
        """
        super().__init__(parent)

        # Aplica estilo visual diferenciado para separar o cabeçalho do restante.
        self.setStyleSheet(SESSION_HEADER_STYLE)

        # Política de tamanho: expande horizontalmente, altura fixa.
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        # Layout horizontal principal — todos os campos ficam na mesma linha.
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 8, 12, 8)
        main_layout.setSpacing(16)

        # === CAMPO: Nome do Paciente ===
        self._build_patient_field(main_layout)

        # === CAMPO: Mão Avaliada ===
        self._build_hand_field(main_layout)

        # === CAMPO: Número da Sessão ===
        self._build_session_number_field(main_layout)

        # Separador vertical entre formulário e cronômetro.
        self._add_vertical_separator(main_layout)

        # === CRONÔMETRO: Horário de Início + Tempo Decorrido ===
        self._build_timer_display(main_layout)

        # Empurra tudo para a esquerda, deixando o cronômetro colado à direita.
        main_layout.addStretch()

        # === TIMER INTERNO ===
        # QTimer que dispara a cada 1000ms para atualizar o tempo decorrido.
        # interval=1000ms garante precisão de segundo sem sobrecarregar a CPU.
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

        # Timestamp do momento em que start_timer() foi chamado.
        # None indica que a sessão ainda não foi iniciada.
        self._start_time: Optional[datetime] = None

    # =========================================================================
    # CONSTRUTORES DAS SEÇÕES DO FORMULÁRIO
    # =========================================================================

    def _build_patient_field(self, layout: QHBoxLayout) -> None:
        """
        Cria o campo de nome do paciente com rótulo e QLineEdit.

        O nome é o único campo OBRIGATÓRIO — is_ready() retorna False
        enquanto este campo estiver vazio ou contiver apenas espaços.

        Parâmetros:
            layout: Layout pai onde o grupo de widgets será adicionado.
        """
        # Container vertical: rótulo em cima, campo de texto embaixo.
        container = QVBoxLayout()
        container.setSpacing(2)
        container.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel("Paciente")
        lbl.setStyleSheet(LABEL_SECTION_TITLE_STYLE)

        # QLineEdit com placeholder para guiar o usuário.
        # maxLength=100: evita nomes absurdamente longos que poderiam
        # causar problemas no nome do arquivo CSV gerado.
        self._input_patient = QLineEdit()
        self._input_patient.setPlaceholderText("Nome completo do paciente...")
        self._input_patient.setMaxLength(100)
        self._input_patient.setMinimumWidth(220)
        self._input_patient.setStyleSheet(
            f"QLineEdit {{ color: {COLOR_TEXT_PRIMARY}; padding: 4px 8px; "
            f"border: 1px solid #334155; border-radius: 5px; "
            f"background: #16213e; font-size: 13px; }}"
            f"QLineEdit:focus {{ border-color: {COLOR_ACCENT}; }}"
        )

        container.addWidget(lbl)
        container.addWidget(self._input_patient)
        layout.addLayout(container)

    def _build_hand_field(self, layout: QHBoxLayout) -> None:
        """
        Cria o seletor de mão avaliada com QComboBox.

        "Direita" e "Esquerda" correspondem ao parâmetro 'side' do
        generate_pdf_report() e ao parâmetro is_right_hand do DigitalGoniometer.

        Parâmetros:
            layout: Layout pai onde o grupo de widgets será adicionado.
        """
        container = QVBoxLayout()
        container.setSpacing(2)
        container.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel("Mão Avaliada")
        lbl.setStyleSheet(LABEL_SECTION_TITLE_STYLE)

        self._combo_hand = QComboBox()
        self._combo_hand.addItems(["Direita", "Esquerda"])
        self._combo_hand.setMinimumWidth(100)
        self._combo_hand.setStyleSheet(
            f"QComboBox {{ color: {COLOR_TEXT_PRIMARY}; padding: 4px 8px; "
            f"border: 1px solid #334155; border-radius: 5px; "
            f"background: #16213e; font-size: 13px; }}"
            f"QComboBox:focus {{ border-color: {COLOR_ACCENT}; }}"
            f"QComboBox QAbstractItemView {{ background: #16213e; "
            f"color: {COLOR_TEXT_PRIMARY}; selection-background-color: {COLOR_ACCENT}; }}"
        )

        container.addWidget(lbl)
        container.addWidget(self._combo_hand)
        layout.addLayout(container)

    def _build_session_number_field(self, layout: QHBoxLayout) -> None:
        """
        Cria o campo do número da sessão com QSpinBox.

        O número da sessão identifica cronologicamente as avaliações do mesmo
        paciente. Por padrão começa em 1 e o clínico incrementa manualmente
        a cada nova sessão do mesmo paciente.

        Parâmetros:
            layout: Layout pai onde o grupo de widgets será adicionado.
        """
        container = QVBoxLayout()
        container.setSpacing(2)
        container.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel("Sessão Nº")
        lbl.setStyleSheet(LABEL_SECTION_TITLE_STYLE)

        # Mínimo 1: não faz sentido ter sessão 0 ou negativa.
        # Máximo 999: razoável para qualquer histórico clínico real.
        self._spin_session = QSpinBox()
        self._spin_session.setMinimum(1)
        self._spin_session.setMaximum(999)
        self._spin_session.setValue(1)
        self._spin_session.setMinimumWidth(70)
        self._spin_session.setStyleSheet(
            f"QSpinBox {{ color: {COLOR_TEXT_PRIMARY}; padding: 4px 8px; "
            f"border: 1px solid #334155; border-radius: 5px; "
            f"background: #16213e; font-size: 13px; }}"
            f"QSpinBox:focus {{ border-color: {COLOR_ACCENT}; }}"
        )

        container.addWidget(lbl)
        container.addWidget(self._spin_session)
        layout.addLayout(container)

    def _build_timer_display(self, layout: QHBoxLayout) -> None:
        """
        Cria o display do cronômetro com horário de início e tempo decorrido.

        Os dois labels ficam em coluna vertical, alinhados à direita.
        O horário de início é preenchido por start_timer().
        O tempo decorrido é atualizado por _tick() a cada segundo.

        Parâmetros:
            layout: Layout pai onde o display será adicionado.
        """
        container = QVBoxLayout()
        container.setSpacing(4)
        container.setContentsMargins(0, 0, 0, 0)
        container.setAlignment(Qt.AlignmentFlag.AlignRight)

        # --- Linha "Início:" ---
        row_start = QHBoxLayout()
        row_start.setSpacing(6)

        lbl_inicio_title = QLabel("Início:")
        lbl_inicio_title.setStyleSheet(LABEL_SECTION_TITLE_STYLE)

        # Preenchido por start_timer() com o horário real de início.
        self._lbl_start_time = QLabel("—")
        self._lbl_start_time.setStyleSheet(
            f"QLabel {{ color: {COLOR_TEXT_PRIMARY}; font-size: 14px; "
            f"font-weight: bold; font-family: 'Consolas', monospace; }}"
        )

        row_start.addWidget(lbl_inicio_title)
        row_start.addWidget(self._lbl_start_time)

        # --- Linha "Decorrido:" ---
        row_elapsed = QHBoxLayout()
        row_elapsed.setSpacing(6)

        lbl_elapsed_title = QLabel("Decorrido:")
        lbl_elapsed_title.setStyleSheet(LABEL_SECTION_TITLE_STYLE)

        # Atualizado a cada segundo pelo QTimer via _tick().
        # Fonte monoespaçada: evita que o layout "pule" quando os dígitos mudam
        # (dígitos de largura variável causam deslocamento em fontes proporcionais).
        self._lbl_elapsed = QLabel("00:00:00")
        self._lbl_elapsed.setStyleSheet(
            f"QLabel {{ color: {COLOR_ACCENT}; font-size: 18px; "
            f"font-weight: bold; font-family: 'Consolas', monospace; }}"
        )

        row_elapsed.addWidget(lbl_elapsed_title)
        row_elapsed.addWidget(self._lbl_elapsed)

        container.addLayout(row_start)
        container.addLayout(row_elapsed)
        layout.addLayout(container)

    def _add_vertical_separator(self, layout: QHBoxLayout) -> None:
        """
        Adiciona um separador vertical entre o formulário e o cronômetro.

        QFrame com frameShape=VLine cria uma barra vertical fina usada como
        divisor visual para separar logicamente as duas seções do cabeçalho:
        dados de identificação (esquerda) e cronômetro (direita).

        Parâmetros:
            layout: Layout pai onde o separador será inserido.
        """
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("color: #334155;")
        separator.setFixedWidth(2)
        layout.addWidget(separator)

    # =========================================================================
    # INTERFACE PÚBLICA — usada pela MainWindow
    # =========================================================================

    def is_ready(self) -> bool:
        """
        Verifica se as condições mínimas para iniciar uma sessão estão atendidas.

        Condição obrigatória: nome do paciente preenchido (não vazio, não só espaços).
        A mão avaliada e o número da sessão sempre têm valores padrão válidos,
        então não precisam ser validados separadamente.

        Usado pela MainWindow para habilitar/desabilitar o botão "Iniciar Sessão"
        em resposta ao evento QLineEdit.textChanged.

        Retorno:
            bool: True se o nome do paciente está preenchido, False caso contrário.
        """
        # strip() remove espaços no início e fim — evita que "   " (só espaços)
        # seja aceito como nome válido, o que geraria arquivos CSV sem identificação real.
        return bool(self._input_patient.text().strip())

    def get_session_info(self) -> dict:
        """
        Retorna os dados do formulário como dicionário para uso no pipeline.

        Deve ser chamado pela MainWindow ao iniciar a sessão, APÓS is_ready()
        retornar True. Os dados retornados alimentam:
        - ProcessingWorker.start_session(csv_path): nome + sessão para o nome do arquivo.
        - generate_pdf_report(): patient_name e side para o relatório PDF.
        - GoniometryCSVLogger: metadados no cabeçalho do CSV.

        Retorno:
            dict com as chaves:
                patient_name (str): Nome completo do paciente sem espaços extras.
                hand (str): "Direita" ou "Esquerda" — conforme seleção do ComboBox.
                session_number (int): Número da sessão (1–999).
                start_time (datetime | None): Momento de início da sessão,
                    ou None se start_timer() ainda não foi chamado.
        """
        return {
            # strip() garante que o nome não contenha espaços desnecessários
            # que poderiam aparecer no nome do arquivo CSV ou no PDF.
            "patient_name": self._input_patient.text().strip(),
            "hand": self._combo_hand.currentText(),
            "session_number": self._spin_session.value(),
            "start_time": self._start_time,
        }

    def start_timer(self) -> None:
        """
        Registra o horário de início da sessão e ativa o cronômetro.

        Chamado pela MainWindow imediatamente após iniciar os workers.
        Faz três coisas:
        1. Registra datetime.now() como horário zero do cronômetro.
        2. Exibe o horário de início no label correspondente.
        3. Inicia o QTimer que chamará _tick() a cada 1000ms.

        Por que datetime.now() e não time.monotonic()?
            datetime.now() fornece a hora real do relógio (para exibir "14:35:12")
            E permite calcular o tempo decorrido por subtração de datetimes.
            time.monotonic() seria mais preciso para intervalos, mas não fornece
            a hora do dia — precisaríamos de duas variáveis separadas.
        """
        # Registra o momento exato de início com microsegundos.
        # Os microsegundos são truncados na exibição mas mantidos internamente
        # para que _tick() calcule o tempo decorrido com precisão de segundo.
        self._start_time = datetime.now()

        # Exibe o horário de início formatado como HH:MM:SS.
        self._lbl_start_time.setText(
            self._start_time.strftime("%H:%M:%S")
        )

        # Reseta o display de tempo decorrido para garantir que mostre 00:00:00
        # antes do primeiro _tick() ser chamado (após ~1 segundo).
        self._lbl_elapsed.setText("00:00:00")

        # Inicia o timer — a partir de agora _tick() será chamado a cada 1 segundo.
        # Se o timer já estiver ativo (chamada duplicada), start() o reinicia do zero.
        self._timer.start()

    def stop_timer(self) -> None:
        """
        Para o cronômetro e congela o display de tempo decorrido.

        Chamado pela MainWindow ao encerrar a sessão. O display congela no
        último valor exibido, permitindo que o clínico leia o tempo total
        da sessão mesmo após o processamento parar.

        O método não limpa o horário de início nem o tempo decorrido —
        esses dados permanecem visíveis para referência até uma nova sessão.
        """
        # Para o QTimer — _tick() não será mais chamado.
        if self._timer.isActive():
            self._timer.stop()

    def reset(self) -> None:
        """
        Redefine o formulário e o cronômetro para o estado inicial.

        Chamado pela MainWindow ao iniciar uma nova sessão após uma anterior
        já ter sido encerrada, ou ao clicar em um botão "Nova Sessão" futuro.
        Não incrementa o número da sessão — isso deve ser feito manualmente
        pelo clínico para manter controle sobre a numeração.

        Atenção: não limpa o nome do paciente — o clínico pode querer
        iniciar outra sessão para o mesmo paciente sem redigitar.
        """
        # Para o timer se estiver ativo.
        self.stop_timer()

        # Limpa as referências de tempo.
        self._start_time = None

        # Redefine os displays para o estado "sem dados".
        self._lbl_start_time.setText("—")
        self._lbl_elapsed.setText("00:00:00")

    def set_fields_enabled(self, enabled: bool) -> None:
        """
        Habilita ou desabilita os campos de entrada do formulário.

        Chamado pela MainWindow ao transitar entre estados:
        - RUNNING: desabilita campos (não alterar dados durante gravação).
        - STOPPED/IDLE: habilita campos (permite editar para próxima sessão).

        Parâmetros:
            enabled: True para habilitar edição, False para bloquear.
        """
        self._input_patient.setEnabled(enabled)
        self._combo_hand.setEnabled(enabled)
        self._spin_session.setEnabled(enabled)

    # =========================================================================
    # LÓGICA INTERNA DO CRONÔMETRO
    # =========================================================================

    def _tick(self) -> None:
        """
        Calcula e exibe o tempo decorrido desde o início da sessão.

        Chamado pelo QTimer interno a cada 1000ms (1 segundo).
        Não deve ser chamado diretamente — é conectado ao timer em start_timer().

        Por que calcular elapsed a cada tick em vez de incrementar um contador?
            Incrementar um contador inteiro (segundos += 1) acumula erro: se algum
            tick demorar mais que 1 segundo (ex: CPU sobrecarregada), o contador
            ficaria para trás em relação ao relógio real. Calcular elapsed como
            (agora - início) garante sempre o tempo real correto, independente
            de variações no intervalo do QTimer.
        """
        if self._start_time is None:
            # Proteção defensiva: timer rodando sem horário de início definido.
            return

        # Calcula o intervalo real decorrido desde o início da sessão.
        elapsed: timedelta = datetime.now() - self._start_time

        # Extrai os componentes de horas, minutos e segundos do timedelta.
        # total_seconds() retorna o total em segundos como float.
        # Dividimos e aplicamos módulo para obter H:M:S independentemente.
        total_seconds: int = int(elapsed.total_seconds())

        # Separação em horas, minutos e segundos restantes.
        # int(total_seconds / 3600) = horas completas.
        # (total_seconds % 3600) // 60 = minutos restantes após remover horas.
        # total_seconds % 60 = segundos restantes após remover horas e minutos.
        hours: int = total_seconds // 3600
        minutes: int = (total_seconds % 3600) // 60
        seconds: int = total_seconds % 60

        # Formata como "HH:MM:SS" com zero à esquerda em cada componente.
        # :02d = inteiro com mínimo 2 dígitos, preenchido com zero à esquerda.
        elapsed_str: str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        self._lbl_elapsed.setText(elapsed_str)
