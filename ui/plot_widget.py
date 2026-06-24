"""
ui/plot_widget.py — Gráfico de TAM em tempo real para todos os 5 dedos
=======================================================================

Este módulo implementa o GoniometryPlotWidget: um gráfico de linhas ao vivo
que exibe o histórico de TAM (Total Active Motion) de cada dedo simultaneamente,
permitindo que o fisioterapeuta acompanhe a evolução do movimento em tempo real.

O que é TAM (Total Active Motion)?
    TAM é a métrica clínica mais importante na avaliação da função da mão.
    Definido pela ASSH (American Society for Surgery of the Hand), representa
    a SOMA das amplitudes ativas de todas as articulações de um dedo:

        TAM = MCP + PIP + DIP  (para dedos longos: Indicador, Médio, Anelar, Mínimo)
        TAM = MCP + IP          (para o Polegar, que não tem DIP)

    Um TAM de 270° (soma máxima teórica de MCP 90° + PIP 110° + DIP 70°)
    indica função plena. O gráfico permite visualizar se o TAM está:
    - Aumentando (melhora funcional ao longo da sessão).
    - Estável (manutenção).
    - Diminuindo (fadiga ou piora).

Por que PyQtGraph e não Matplotlib?
    Matplotlib gera gráficos ESTÁTICOS — redesenhar a cada frame (~30x/s)
    seria catastroficamente lento (150-400ms por redesenho). PyQtGraph é
    otimizado para dados em tempo real: usa OpenGL quando disponível e
    atualiza apenas os pixels que mudaram. Atualizações a 30 FPS com
    PyQtGraph custam ~1-3ms, versus 150-400ms com Matplotlib.

Estrutura de dados:
    Um deque(maxlen=BUFFER_SIZE) por dedo mantém os últimos N valores de TAM.
    A cada frame, o novo TAM é adicionado e o mais antigo é descartado
    automaticamente pelo deque. O eixo X do gráfico é implicitamente o
    índice da amostra (0 a N-1) — não representa tempo absoluto.
"""

from collections import deque
from typing import Deque, Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

# Importa pyqtgraph com verificação de disponibilidade.
try:
    import pyqtgraph as pg
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False

import config


class GoniometryPlotWidget(QWidget):
    """
    Gráfico de linhas ao vivo com o histórico de TAM dos 5 dedos da mão.

    Exibe 5 curvas simultâneas (uma por dedo), cada uma com sua cor
    identificadora (definida em config.FINGER_COLORS), e uma legenda
    com os nomes em português (config.FINGER_NAMES_PT).

    Por que herdar de QWidget em vez de pg.PlotWidget diretamente?
        Herdar diretamente de pg.PlotWidget limita a flexibilidade de layout:
        não conseguimos adicionar widgets extras (ex: título, controles) sem
        criar um container externo. Ao herdar de QWidget e CONTER um PlotWidget
        interno, mantemos total controle sobre o layout e podemos adicionar
        elementos futuros sem refatorar.

    Degradação graciosa:
        Se PyQtGraph não estiver instalado, o widget exibe uma mensagem
        informativa em vez de travar a aplicação. Isso permite que o
        resto da interface funcione mesmo sem o gráfico.

    Uso na MainWindow:
        self.plot_widget = GoniometryPlotWidget()
        layout.addWidget(self.plot_widget)
        # Em _on_result():
        self.plot_widget.update_data(result.angles_smooth, result.hand_detected)
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Inicializa o gráfico com 5 curvas, legenda, grid e buffers circulares.

        Configura o PyQtGraph ANTES de instanciar qualquer widget, pois
        pg.setConfigOption() deve ser chamada antes da criação de PlotWidgets
        para ter efeito. Configurações feitas depois são ignoradas.

        Parâmetros:
            parent: Widget pai Qt (opcional). Geralmente o container do layout.
        """
        super().__init__(parent)

        # Layout vertical que contém apenas o PlotWidget (ou a mensagem de erro).
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Altura fixa para o gráfico — suficiente para ver as 5 curvas com
        # amplitude visível, sem dominar o layout da janela principal.
        self.setFixedHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        if not _PG_AVAILABLE:
            # PyQtGraph não instalado — exibe aviso sem travar a aplicação.
            from PyQt6.QtWidgets import QLabel
            lbl = QLabel("⚠️ PyQtGraph não instalado.\nExecute: pip install pyqtgraph")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #ef4444; font-size: 13px;")
            layout.addWidget(lbl)
            # Dicionários vazios para que update_data() não quebre.
            self._curves: Dict[str, object] = {}
            self._buffers: Dict[str, Deque[float]] = {}
            self._plot_widget = None
            return

        # --- Configurações globais do PyQtGraph ---
        # Devem ser definidas ANTES de qualquer instância de PlotWidget.

        # Fundo padrão para todos os PlotWidgets criados após esta chamada.
        # Usamos o mesmo tom escuro do tema da aplicação (COLOR_BG_MEDIUM).
        pg.setConfigOption("background", "#16213e")

        # Cor padrão dos eixos e textos do gráfico.
        pg.setConfigOption("foreground", "#94a3b8")

        # Desabilita OpenGL por padrão para compatibilidade máxima no Windows.
        # Se o sistema suportar OpenGL, pode ser habilitado em app_pyqt.py
        # com pg.setConfigOption('useOpenGL', True) ANTES de criar este widget.
        pg.setConfigOption("useOpenGL", False)

        # --- Criação do PlotWidget ---
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("#16213e")

        # Remove a borda padrão do PlotWidget — o QGroupBox pai já tem borda.
        self._plot_widget.setStyleSheet("border: none;")

        layout.addWidget(self._plot_widget)

        # --- Configuração do PlotItem (o gráfico dentro do PlotWidget) ---
        plot_item: pg.PlotItem = self._plot_widget.getPlotItem()

        # --- Título do gráfico ---
        plot_item.setTitle(
            "TAM em Tempo Real — Total Active Motion por Dedo",
            color="#94a3b8",
            size="11pt",
        )

        # --- Rótulos dos eixos ---
        # Eixo Y: "TAM (°)" — a grandeza exibida e sua unidade.
        plot_item.setLabel("left", "TAM", units="°", color="#94a3b8")

        # Eixo X: sem rótulo porque representa apenas o índice sequencial
        # de amostras, não um tempo absoluto. Exibir "Amostras" ou "Frames"
        # seria tecnicamente correto mas confuso para o fisioterapeuta.
        plot_item.hideAxis("bottom")

        # --- Grid sutil ---
        # alpha=0.3: grade visível mas discreta — não compete com as curvas.
        # Valores maiores (0.5+) tornam a grade proeminente demais e dificultam
        # a leitura das curvas coloridas sobrepostas.
        plot_item.showGrid(x=True, y=True, alpha=0.3)

        # --- Limites do eixo Y ---
        # TAM varia de 0° (mão completamente fechada sem abertura) a ~270°
        # para dedos longos e ~130° para o polegar. Fixamos em 280° para
        # que o gráfico não "pule" ao aproximar-se do limite superior.
        self._plot_widget.setYRange(0, 280, padding=0.05)

        # --- Legenda ---
        # addLegend() cria a legenda no canto superior direito por padrão.
        # offset=(10, 10): posição relativa ao canto — 10px de margem.
        legend = plot_item.addLegend(offset=(10, 10))
        legend.setLabelTextColor("#e2e8f0")

        # --- Criação das curvas e buffers ---
        # Um PlotDataItem por dedo + um deque por dedo.
        self._curves: Dict[str, pg.PlotDataItem] = {}
        self._buffers: Dict[str, Deque[float]] = {}

        for finger in config.FINGERS:
            # Recupera a cor hex deste dedo do dicionário de configuração.
            color_hex: str = config.FINGER_COLORS.get(finger, "#ffffff")

            # Nome em português para a legenda — legível pelo clínico.
            name_pt: str = config.FINGER_NAMES_PT.get(finger, finger)

            # Cria a curva com:
            # - pen: caneta com a cor do dedo e largura 2px (espessura legível).
            # - name: nome que aparece na legenda (em português).
            # Não passamos x/y aqui — serão definidos em update_data() via setData().
            curve = plot_item.plot(
                pen=pg.mkPen(color=color_hex, width=2),
                name=name_pt,
            )
            self._curves[finger] = curve

            # Buffer circular por dedo.
            # Por que deque(maxlen=BUFFER_SIZE) em vez de uma lista crescente?
            #   1. Memória FIXA: uma lista crescente nunca descarta dados antigos
            #      e cresceria indefinidamente ao longo de uma sessão longa.
            #      A 30 FPS por 60 minutos = 108.000 floats por dedo = ~840KB por dedo.
            #      Com BUFFER_SIZE=500, limitamos a ~3.9KB por dedo independente
            #      da duração da sessão.
            #   2. FIFO automático: ao adicionar um novo valor quando maxlen é
            #      atingido, o mais antigo é descartado automaticamente —
            #      sem código adicional de gerenciamento.
            #   3. Janela deslizante: o gráfico sempre exibe os ÚLTIMOS N pontos,
            #      criando o efeito de "janela que avança com o tempo".
            #
            # Por que NÃO usar um array NumPy no deque?
            #   Um deque de floats Python é estruturalmente uma lista duplamente
            #   encadeada. Para uso em PyQtGraph com setData(list(deque)), a
            #   conversão list() é O(n) e rápida. Usar numpy arrays como elementos
            #   do deque adicionaria overhead de alocação por frame sem benefício
            #   (o PyQtGraph converte internamente para numpy de qualquer forma).
            self._buffers[finger] = deque(maxlen=config.BUFFER_SIZE)

    # =========================================================================
    # ATUALIZAÇÃO DOS DADOS DO GRÁFICO
    # =========================================================================

    def update_data(self, angles_smooth: dict, hand_detected: bool) -> None:
        """
        Atualiza o gráfico com os ângulos suavizados do frame atual.

        Chamado pela MainWindow a cada emissão de result_ready do ProcessingWorker
        (~30 vezes/segundo). Por isso, deve ser rápido: apenas append() no deque
        e setData() na curva — sem cálculos, sem acesso a disco.

        Por que não adicionar ponto se hand_detected é False?
            Quando a mão não está visível (fora do campo, coberta), o pipeline
            retorna ângulos zerados ou da última detecção válida. Adicionar
            zeros no gráfico criaria quedas abruptas para 0 que não representam
            movimento real — são artefatos de ausência de detecção. Mantendo
            o histórico estático, o gráfico "pausa" enquanto aguarda a mão
            voltar ao campo de visão.

        Parâmetros:
            angles_smooth: Dicionário {dedo: {articulação: ângulo}} retornado
                           por GoniometryFilterBank.smooth_all(). Ex:
                           {"INDEX": {"MCP": 45.2, "PIP": 88.1, "DIP": 62.3, "TAM": 195.6}}
            hand_detected: True se MediaPipe detectou a mão neste frame.
                           False quando nenhuma mão está visível.
        """
        # Se PyQtGraph não está disponível, não há curvas para atualizar.
        if not _PG_AVAILABLE:
            return

        # Sem detecção, mantemos o histórico atual sem adicionar pontos zerados.
        if not hand_detected:
            return

        for finger in config.FINGERS:
            # Extrai o TAM deste dedo do dicionário de ângulos suavizados.
            # TAM é escolhido como métrica do gráfico porque:
            #   1. Resume em UM NÚMERO toda a função do dedo (soma de todos os ângulos).
            #   2. É a métrica clínica oficial da ASSH para avaliação funcional.
            #   3. É suficientemente estável para visualização em tempo real
            #      (não oscila como MCP ou PIP individuais durante movimento).
            # .get(finger, {}).get("TAM", 0.0): acesso seguro com fallback 0.0
            # para o caso de o dicionário não conter este dedo (ex: oclusão parcial).
            tam: float = float(angles_smooth.get(finger, {}).get("TAM", 0.0))

            # Só adiciona ao buffer se o valor é positivo.
            # TAM = 0.0 indica ausência de dados, não ângulo real.
            # Incluir zeros distorceria a escala do gráfico e a visualização.
            if tam > 0.0:
                self._buffers[finger].append(tam)

            # Converte o deque para list() para passar ao PyQtGraph.
            # list(deque) cria uma cópia linear do deque em O(n).
            # setData() com lista de floats Python é aceito pelo PyQtGraph,
            # que converte internamente para numpy apenas na renderização.
            # Isso é mais eficiente do que manter um array NumPy separado
            # e concatenar a cada frame.
            data = list(self._buffers[finger])

            if data:
                # setData() com apenas y: o eixo X é automaticamente
                # 0, 1, 2, ..., len(data)-1 — o índice da amostra.
                # Não precisamos de um array X explícito porque o gráfico
                # é uma janela deslizante de índices, não de timestamps.
                self._curves[finger].setData(y=data)

    # =========================================================================
    # LIMPEZA DOS DADOS
    # =========================================================================

    def clear_data(self) -> None:
        """
        Limpa todos os buffers e redesenha as curvas vazias.

        Chamado pela MainWindow ao iniciar uma nova sessão, para que os
        dados da sessão anterior não apareçam no gráfico da nova sessão.
        Também útil para "zerar" o gráfico sem reiniciar o widget inteiro.

        Após clear_data(), update_data() começa a construir o histórico
        do zero — as curvas crescem gradualmente da esquerda para a direita
        até preencher BUFFER_SIZE amostras.
        """
        if not _PG_AVAILABLE:
            return

        for finger in config.FINGERS:
            # Esvazia o deque sem recriar o objeto — mais eficiente que
            # substituir por deque(maxlen=BUFFER_SIZE) porque não realoca memória.
            self._buffers[finger].clear()

            # Redesenha a curva com array vazio para limpar visualmente o gráfico.
            # Passar y=[] instrui o PyQtGraph a não desenhar nenhum ponto.
            self._curves[finger].setData(y=[])

    # =========================================================================
    # CONFIGURAÇÃO DE RANGES CLÍNICOS
    # =========================================================================

    def set_y_range(self, y_min: float, y_max: float) -> None:
        """
        Ajusta o range visível do eixo Y do gráfico.

        Útil quando o clínico quer focar em uma faixa específica de TAM,
        por exemplo ao avaliar pacientes com amplitude muito reduzida
        (TAM < 100°) onde a escala padrão 0–280° seria muito espaçada.

        Parâmetros:
            y_min: Valor mínimo do eixo Y em graus. Geralmente 0.0.
            y_max: Valor máximo do eixo Y em graus. Padrão da aplicação: 280.0.
        """
        if not _PG_AVAILABLE or self._plot_widget is None:
            return

        # padding=0: sem margem extra acima e abaixo do range definido.
        # Com padding padrão (~0.05), o PyQtGraph adiciona 5% de espaço
        # além dos limites, o que poderia cortar os labels do eixo.
        self._plot_widget.setYRange(y_min, y_max, padding=0)

    # =========================================================================
    # VISIBILIDADE DAS CURVAS
    # =========================================================================

    def set_finger_visible(self, finger: str, visible: bool) -> None:
        """
        Mostra ou oculta a curva de um dedo específico.

        Permite que o clínico foque em um único dedo ocultando os demais,
        ou reabilite todos após uma avaliação individual.

        Parâmetros:
            finger: Chave do dedo no formato usado pelo pipeline científico.
                    Valores válidos: "INDEX", "MIDDLE", "RING", "PINKY", "THUMB".
            visible: True para exibir a curva, False para ocultá-la.
        """
        if not _PG_AVAILABLE:
            return

        curve = self._curves.get(finger)
        if curve is not None:
            # setVisible() afeta a renderização mas não remove os dados do buffer.
            # Ao tornar a curva visível novamente, os dados históricos são mantidos.
            curve.setVisible(visible)
