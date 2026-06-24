"""
ui/finger_card_widget.py — Cartões individuais por dedo com mini-gráfico
=========================================================================

Este módulo implementa dois componentes visuais:

1. FingerCardWidget(QGroupBox):
   Exibe TODAS as métricas clínicas de UM único dedo em um cartão compacto.
   Cada instância é dedicada a um dedo específico (Polegar, Indicador, etc.).

2. FingerCardsPanel(QWidget):
   Container que organiza os 5 FingerCardWidgets lado a lado em linha.
   É o único componente que a MainWindow precisa instanciar — ele cuida
   dos 5 cartões internamente.

Métricas exibidas por cartão (vindas do pipeline científico):
    TAM (°)       : Total Active Motion — amplitude total de movimento ativo.
    ASSH          : Classificação funcional (Excelente / Bom / Moderado / Ruim).
    Amplitude (°) : Diferença entre TAM máximo e mínimo na janela de tempo.
    Vel. Média    : Velocidade angular média (°/s) — ritmo geral do movimento.
    Vel. Pico     : Velocidade angular máxima (°/s) — pico de esforço.
    Frequência    : Taxa de ciclos completos por segundo (Hz).
    Regularidade  : Avaliação qualitativa da consistência do movimento.
    Mini-gráfico  : Histórico de TAM nos últimos BUFFER_SIZE pontos (PyQtGraph).

Fontes dos dados:
    state   ← classify_hand_state()["finger_states"][finger]
    metrics ← compute_realtime_metrics(angle_buffer, time_buffer)

Por que um cartão por dedo?
    O fisioterapeuta muitas vezes precisa comparar rapidamente o desempenho
    de dedos adjacentes (ex: Indicador vs Médio pós-lesão). Exibir todos
    em linha permite essa comparação visual instantânea sem navegar por menus.
"""

from typing import Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# Importa estilos centralizados do módulo de tema.
from themes import (
    FINGER_CARD_STYLE,
    LABEL_CLINICAL_SECONDARY_STYLE,
    LABEL_CLINICAL_VALUE_STYLE,
    LABEL_SECTION_TITLE_STYLE,
)
import config

# Tenta importar PyQtGraph para os mini-gráficos.
# Se não estiver instalado, os mini-gráficos são substituídos por um label.
try:
    import pyqtgraph as pg
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False


class FingerCardWidget(QGroupBox):
    """
    Cartão clínico completo para um único dedo da mão.

    Exibe métricas de goniometria, métricas cinéticas e um mini-gráfico
    de TAM ao vivo em um espaço compacto, projetado para caber em linha
    com outros 4 cartões sem sobreposição.

    Layout interno (QVBoxLayout):
        ┌──────────────────────────────┐
        │ [nome do dedo em português]  │  ← título do QGroupBox
        │ TAM: 127.3°  [Moderado] 🟡  │  ← TAM + ASSH em linha
        │ ─────────────────────────── │
        │ Amplitude  : 45.2°          │
        │ Vel. Média : 38.1 °/s       │
        │ Vel. Pico  : 112.4 °/s      │
        │ Frequência : 0.33 Hz        │
        │ Regularidade: ✅ Regular    │
        │ ─────────────────────────── │
        │ [mini-gráfico TAM, 80px]    │
        └──────────────────────────────┘
    """

    def __init__(
        self,
        finger_key: str,
        name_pt: str,
        color_hex: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Inicializa o cartão para um dedo específico.

        Parâmetros:
            finger_key: Identificador interno do dedo no pipeline científico.
                        Valores: "INDEX", "MIDDLE", "RING", "PINKY", "THUMB".
            name_pt: Nome do dedo em português para o título do grupo.
                     Ex: "Indicador", "Médio", "Polegar".
            color_hex: Cor hex do dedo (de config.FINGER_COLORS).
                       Usada no mini-gráfico e em elementos de destaque.
            parent: Widget pai Qt (opcional).
        """
        super().__init__(name_pt, parent)

        # Guarda a chave do dedo para uso futuro (ex: depuração, logging).
        self._finger_key: str = finger_key
        self._color_hex: str = color_hex

        # Aplica o estilo visual do cartão (borda, fundo, título).
        self.setStyleSheet(FINGER_CARD_STYLE)

        # Largura mínima para que os 5 cartões não fiquem espremidos.
        # Com 5 cartões em linha em janela de 1280px: 1280/5 = 256px por cartão.
        self.setMinimumWidth(200)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        # Layout principal vertical do cartão.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 16, 8, 8)
        layout.setSpacing(4)

        # --- Linha TAM + Classificação ASSH ---
        self._build_tam_row(layout)

        # --- Separador visual ---
        self._add_separator(layout)

        # --- Métricas cinéticas em grade ---
        self._build_metrics_grid(layout)

        # --- Separador visual ---
        self._add_separator(layout)

        # --- Mini-gráfico TAM ao vivo ---
        self._build_mini_chart(layout, color_hex)

    # =========================================================================
    # CONSTRUTORES DAS SEÇÕES INTERNAS
    # =========================================================================

    def _build_tam_row(self, parent_layout: QVBoxLayout) -> None:
        """
        Constrói a linha principal com TAM e classificação ASSH.

        Coloca TAM e ASSH na mesma linha horizontal para economizar espaço
        vertical sem sacrificar legibilidade — TAM é o valor mais importante
        e deve ter destaque visual imediato.

        Parâmetros:
            parent_layout: Layout pai onde a linha será adicionada.
        """
        row = QHBoxLayout()
        row.setSpacing(6)

        # Rótulo "TAM" — indica a grandeza exibida.
        lbl_tam_title = QLabel("TAM:")
        lbl_tam_title.setStyleSheet(LABEL_SECTION_TITLE_STYLE)
        lbl_tam_title.setFixedWidth(36)

        # Valor numérico do TAM — maior destaque visual.
        self._lbl_tam_value = QLabel("—")
        self._lbl_tam_value.setStyleSheet(LABEL_CLINICAL_VALUE_STYLE)

        # Classificação ASSH com cor dinâmica (verde/amarelo/laranja/vermelho).
        # A cor é aplicada via setStyleSheet em update(), não aqui.
        self._lbl_assh = QLabel("—")
        self._lbl_assh.setStyleSheet(LABEL_CLINICAL_SECONDARY_STYLE)
        self._lbl_assh.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        row.addWidget(lbl_tam_title)
        row.addWidget(self._lbl_tam_value)
        row.addStretch()
        row.addWidget(self._lbl_assh)

        parent_layout.addLayout(row)

    def _build_metrics_grid(self, parent_layout: QVBoxLayout) -> None:
        """
        Constrói a grade com as métricas cinéticas (amplitude, velocidade, frequência).

        Usa uma grade de 2 colunas (rótulo | valor) para alinhar corretamente
        os dados sem desperdiçar espaço. Cada linha representa uma métrica diferente.

        Parâmetros:
            parent_layout: Layout pai onde a grade será adicionada.
        """
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)

        # Nomes das métricas (rótulos estáticos) e referências aos labels de valor.
        # A ordem segue a relevância clínica: amplitude primeiro, depois velocidade,
        # frequência e regularidade (da mais objetiva à mais interpretada).
        metrics_rows = [
            ("Amplitude",   "—"),
            ("Vel. Média",  "—"),
            ("Vel. Pico",   "—"),
            ("Frequência",  "—"),
            ("Regularidade", "—"),
        ]

        # Dicionário para acesso rápido em update() — mapeando nome interno → QLabel.
        self._metric_labels: Dict[str, QLabel] = {}

        for row_idx, (label_text, init_value) in enumerate(metrics_rows):
            # Rótulo estático à esquerda.
            lbl_title = QLabel(f"{label_text}:")
            lbl_title.setStyleSheet(LABEL_CLINICAL_SECONDARY_STYLE)
            lbl_title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            # Valor dinâmico à direita — atualizado a cada frame com dados reais.
            lbl_value = QLabel(init_value)
            lbl_value.setStyleSheet(LABEL_CLINICAL_SECONDARY_STYLE)
            lbl_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            grid.addWidget(lbl_title, row_idx, 0)
            grid.addWidget(lbl_value, row_idx, 1)

            # Armazena referência usando o nome sem ":" como chave.
            self._metric_labels[label_text] = lbl_value

        # Coluna 0 (rótulos): largura fixa. Coluna 1 (valores): estica.
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)

        parent_layout.addLayout(grid)

    def _build_mini_chart(self, parent_layout: QVBoxLayout, color_hex: str) -> None:
        """
        Constrói o mini-gráfico PyQtGraph de TAM ao vivo dentro do cartão.

        O mini-gráfico tem height=80px, sem eixos visíveis, apenas a curva.
        Seu propósito é dar contexto temporal ao valor numérico do TAM —
        o clínico pode ver se o valor está subindo, descendo ou oscilando.

        Por que sem eixos?
            Com 5 cartões em linha, cada um com apenas 200px de largura,
            eixos com labels tomariam ~30% do espaço do gráfico. A curva
            em si já comunica a tendência sem precisar de escalas numéricas.

        Parâmetros:
            parent_layout: Layout pai onde o mini-gráfico será adicionado.
            color_hex: Cor da curva, correspondente ao dedo (config.FINGER_COLORS).
        """
        self._mini_curve = None
        self._mini_plot = None

        if not _PG_AVAILABLE:
            # Fallback sem PyQtGraph: exibe mensagem no lugar do gráfico.
            lbl_no_pg = QLabel("(PyQtGraph não instalado)")
            lbl_no_pg.setStyleSheet(LABEL_CLINICAL_SECONDARY_STYLE)
            lbl_no_pg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_no_pg.setFixedHeight(80)
            parent_layout.addWidget(lbl_no_pg)
            return

        # Configura fundo do mini-gráfico com o mesmo tom escuro do cartão.
        self._mini_plot = pg.PlotWidget()
        self._mini_plot.setBackground("#1e293b")

        # Remove borda do widget — o QGroupBox já tem borda.
        self._mini_plot.setStyleSheet("border: none;")

        # Altura fixa de 80px — compacto mas suficiente para mostrar a tendência.
        self._mini_plot.setFixedHeight(80)

        # Remove TODOS os eixos para máxima compactação visual.
        # Os eixos consumiriam ~40% da altura em um gráfico tão pequeno.
        plot_item = self._mini_plot.getPlotItem()
        plot_item.hideAxis("left")
        plot_item.hideAxis("bottom")

        # Remove o menu de contexto do botão direito — desnecessário em mini-gráfico.
        plot_item.setMenuEnabled(False)

        # Desabilita interação do mouse — o mini-gráfico é somente visualização.
        self._mini_plot.setMouseEnabled(x=False, y=False)

        # Cria a única curva do mini-gráfico: TAM ao longo do tempo.
        # Largura 1.5px: visível em 80px de altura sem ser muito grossa.
        self._mini_curve = plot_item.plot(
            pen=pg.mkPen(color=color_hex, width=1.5),
        )

        parent_layout.addWidget(self._mini_plot)

    def _add_separator(self, parent_layout: QVBoxLayout) -> None:
        """
        Adiciona um separador horizontal sutil entre seções do cartão.

        QFrame com frameShape=HLine cria uma linha horizontal fina, usada
        como divisor visual entre TAM, métricas e o mini-gráfico.

        Parâmetros:
            parent_layout: Layout pai onde o separador será adicionado.
        """
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        # Sombra rebaixada: cria efeito de linha ligeiramente afundada.
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("color: #334155; max-height: 1px;")
        parent_layout.addWidget(separator)

    # =========================================================================
    # ATUALIZAÇÃO DOS DADOS
    # =========================================================================

    def update(
        self,
        state: dict,
        metrics: dict,
        tam_buffer: List[float],
    ) -> None:
        """
        Atualiza todos os campos do cartão com dados do frame atual.

        Este método é chamado pela FingerCardsPanel a cada emissão de
        result_ready (~30x/s). Deve ser rápido: apenas atualiza texto
        e dados do gráfico, sem nenhum cálculo.

        Estrutura esperada de 'state' (de classify_hand_state()):
            Dedos longos: {"MCP": float, "PIP": float, "DIP": float,
                          "ABD": float, "TAM": float,
                          "closed": bool, "assh_label": str, "assh_color": str}
            Polegar:      {"MCP": float, "IP": float, "TAM": float,
                          "closed": bool, "assh_label": str, "assh_color": str}

        Estrutura esperada de 'metrics' (de compute_realtime_metrics()):
            {"amplitude": float, "vel_media": float, "vel_pico": float,
             "freq_hz": float, "cv": float, "regularidade": str, "n_picos": int}

        Parâmetros:
            state: Dicionário com ângulos atuais e classificação ASSH do dedo.
            metrics: Dicionário com métricas cinéticas calculadas sobre o buffer.
            tam_buffer: Lista com os últimos N valores de TAM para o mini-gráfico.
        """
        # --- TAM principal ---
        tam: float = float(state.get("TAM", 0.0))
        self._lbl_tam_value.setText(f"{tam:.1f}°")

        # --- Classificação ASSH com cor dinâmica ---
        assh_label: str = state.get("assh_label", "—")
        assh_color: str = state.get("assh_color", "#94a3b8")

        self._lbl_assh.setText(assh_label)

        # Aplica a cor da classificação via setStyleSheet.
        # Cada nível ASSH tem uma cor predefinida (verde/amarelo/laranja/vermelho)
        # que o clínico reconhece imediatamente sem precisar ler o texto.
        self._lbl_assh.setStyleSheet(
            f"QLabel {{ color: {assh_color}; font-size: 12px; font-weight: bold; }}"
        )

        # --- Métricas cinéticas ---
        amplitude: float = float(metrics.get("amplitude", 0.0))
        vel_media: float = float(metrics.get("vel_media", 0.0))
        vel_pico: float  = float(metrics.get("vel_pico",  0.0))
        freq_hz: float   = float(metrics.get("freq_hz",   0.0))
        regularidade: str = str(metrics.get("regularidade", "—"))

        # Formata amplitude em graus com uma casa decimal.
        self._metric_labels["Amplitude"].setText(f"{amplitude:.1f}°")

        # Formata velocidades em graus por segundo com uma casa decimal.
        self._metric_labels["Vel. Média"].setText(f"{vel_media:.1f} °/s")
        self._metric_labels["Vel. Pico"].setText(f"{vel_pico:.1f} °/s")

        # Formata frequência em Hz com duas casas decimais.
        # Dois decimais são necessários porque movimentos lentos (0.25Hz) e
        # rápidos (2.00Hz) precisam ser distinguíveis com precisão.
        self._metric_labels["Frequência"].setText(f"{freq_hz:.2f} Hz")

        # Adiciona ícone visual à regularidade para reconhecimento imediato.
        # O clínico pode avaliar de relance sem ler a palavra.
        if regularidade == "Regular":
            reg_text = "✅ Regular"
        elif regularidade == "Irregular":
            reg_text = "❌ Irregular"
        else:
            reg_text = regularidade
        self._metric_labels["Regularidade"].setText(reg_text)

        # --- Mini-gráfico ---
        self._update_mini_chart(tam_buffer)

    def _update_mini_chart(self, tam_buffer: List[float]) -> None:
        """
        Atualiza a curva do mini-gráfico com o buffer de TAM atual.

        Chamado internamente por update() a cada frame. Se PyQtGraph não
        estiver disponível, este método retorna silenciosamente.

        O eixo Y do mini-gráfico é autoscalado pelo PyQtGraph para encaixar
        o range atual dos dados — sem configuração explícita de limites.
        Isso faz a curva preencher sempre toda a altura de 80px, independente
        da amplitude real do movimento.

        Parâmetros:
            tam_buffer: Lista de floats com os valores de TAM históricos.
                        Vazio se ainda não há dados suficientes.
        """
        if self._mini_curve is None or not _PG_AVAILABLE:
            return

        if tam_buffer:
            # setData() com apenas y: eixo X = 0, 1, 2, ... (índice da amostra).
            # Não precisamos de timestamps reais no mini-gráfico — a tendência
            # visual é suficiente para comunicar a evolução do movimento.
            self._mini_curve.setData(y=tam_buffer)
        else:
            # Buffer vazio: limpa o gráfico para não exibir dados antigos.
            self._mini_curve.setData(y=[])

    def clear(self) -> None:
        """
        Redefine todos os campos do cartão para o estado inicial sem dados.

        Chamado pela FingerCardsPanel ao iniciar uma nova sessão, para que
        os dados da sessão anterior não sejam confundidos com os da nova.
        """
        self._lbl_tam_value.setText("—")
        self._lbl_assh.setText("—")
        self._lbl_assh.setStyleSheet(LABEL_CLINICAL_SECONDARY_STYLE)

        for lbl in self._metric_labels.values():
            lbl.setText("—")

        if self._mini_curve is not None:
            self._mini_curve.setData(y=[])


# =============================================================================
# PAINEL COM OS 5 CARTÕES
# =============================================================================

class FingerCardsPanel(QWidget):
    """
    Container que organiza os 5 FingerCardWidgets lado a lado em linha.

    É o único componente deste módulo que a MainWindow instancia diretamente.
    Internamente, cria e gerencia os 5 cartões individuais.

    Ordem dos cartões (da esquerda para a direita):
        Polegar | Indicador | Médio | Anelar | Mínimo

    A ordem segue a anatomia da mão vista de frente, facilitando a correlação
    visual entre o cartão na tela e o dedo real do paciente durante a sessão.

    Uso na MainWindow:
        self.finger_cards = FingerCardsPanel()
        layout.addWidget(self.finger_cards)
        # Em _on_result():
        self.finger_cards.update_all(
            result.hand_state["finger_states"],
            result.metrics_per_finger,
            result.tam_buffers_snapshot,
        )
    """

    # Ordem de exibição dos cartões (da esquerda para a direita).
    # THUMB primeiro porque é anatomicamente o primeiro dedo na mão vista de frente.
    DISPLAY_ORDER: List[str] = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Inicializa o painel criando os 5 cartões em linha.

        Cada cartão recebe sua chave de dedo, nome em português e cor
        identificadora a partir dos dicionários de config.py.

        Parâmetros:
            parent: Widget pai Qt (opcional).
        """
        super().__init__(parent)

        # Layout horizontal: os 5 cartões ficam lado a lado.
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # Espaçamento de 6px entre cartões — visualmente separados mas compactos.
        layout.setSpacing(6)

        # Dicionário para acesso rápido em update_all() — chave = nome do dedo.
        self._cards: Dict[str, FingerCardWidget] = {}

        for finger_key in self.DISPLAY_ORDER:
            name_pt = config.FINGER_NAMES_PT.get(finger_key, finger_key)
            color_hex = config.FINGER_COLORS.get(finger_key, "#ffffff")

            card = FingerCardWidget(
                finger_key=finger_key,
                name_pt=name_pt,
                color_hex=color_hex,
                parent=self,
            )
            self._cards[finger_key] = card
            layout.addWidget(card)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

    def update_all(
        self,
        finger_states: Dict[str, dict],
        metrics_per_finger: Dict[str, dict],
        tam_buffers_per_finger: Dict[str, List[float]],
    ) -> None:
        """
        Atualiza todos os 5 cartões com os dados do frame atual.

        Chamado pela MainWindow a cada emissão de result_ready do ProcessingWorker.
        Itera sobre os 5 dedos e delega a atualização de cada cartão para
        o FingerCardWidget correspondente.

        Tolerância a dados ausentes:
            Se um dedo não está presente em finger_states ou metrics_per_finger
            (ex: MediaPipe perdeu o rastreamento de um dedo específico),
            usamos dicionários vazios como fallback para que o cartão exiba
            "—" em vez de lançar KeyError.

        Parâmetros:
            finger_states: Dicionário {dedo: state_dict} retornado por
                           classify_hand_state()["finger_states"].
                           Contém ângulos atuais e classificação ASSH de cada dedo.

            metrics_per_finger: Dicionário {dedo: metrics_dict} onde cada dict é
                                a saída de compute_realtime_metrics() para aquele dedo.
                                Contém amplitude, vel_media, vel_pico, freq_hz, etc.

            tam_buffers_per_finger: Dicionário {dedo: [float]} com o histórico
                                    de TAM para o mini-gráfico de cada cartão.
                                    Geralmente vem de ProcessingResult.tam_buffers_snapshot.
        """
        for finger_key, card in self._cards.items():
            # Acesso seguro com fallback: se o dedo não foi detectado neste frame,
            # passamos dicionários vazios e o card exibirá "—" nos campos.
            state = finger_states.get(finger_key, {})
            metrics = metrics_per_finger.get(finger_key, {})
            tam_buf = tam_buffers_per_finger.get(finger_key, [])

            card.update(state=state, metrics=metrics, tam_buffer=tam_buf)

    def clear_all(self) -> None:
        """
        Redefine todos os 5 cartões para o estado inicial sem dados.

        Chamado pela MainWindow ao iniciar uma nova sessão, para limpar
        todos os dados da sessão anterior antes da câmera ser ligada.
        """
        for card in self._cards.values():
            card.clear()

    def get_card(self, finger_key: str) -> Optional[FingerCardWidget]:
        """
        Retorna o FingerCardWidget de um dedo específico.

        Útil para operações direcionadas (ex: destacar o cartão de um dedo
        específico durante uma análise individual, ou ocultar temporariamente
        um dedo não avaliado).

        Parâmetros:
            finger_key: Chave do dedo. Ex: "INDEX", "THUMB", "PINKY".

        Retorno:
            FingerCardWidget correspondente, ou None se a chave não existir.
        """
        return self._cards.get(finger_key)
