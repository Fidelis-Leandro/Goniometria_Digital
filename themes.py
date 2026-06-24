"""
themes.py — Tema visual escuro profissional para a interface PyQt6
==================================================================

Este módulo é responsável pela identidade visual de TODA a aplicação.
Ao centralizar cores, fontes e estilos aqui, garantimos que qualquer
alteração visual futura seja feita em um único lugar, propagando-se
automaticamente para todos os widgets.

Como usar:
    No app_pyqt.py, após criar o QApplication, chame:
        from themes import apply_dark_theme
        apply_dark_theme(app)

    Nos widgets individuais, importe as constantes de estilo:
        from themes import CARD_STYLE, LABEL_TITLE_STYLE

Filosofia de design:
    O tema escuro foi escolhido porque:
    1. Reduz a fadiga ocular em sessões clínicas longas.
    2. Aumenta o contraste dos gráficos coloridos (PyQtGraph).
    3. É o padrão de facto em aplicações científicas e médicas modernas.
    4. O overlay de vídeo com fundo escuro (produzido por goniometry_overlay.py)
       integra-se visualmente de forma natural ao tema escuro.
"""

from PyQt6.QtGui import QColor, QPalette, QFont
from PyQt6.QtWidgets import QApplication


# =============================================================================
# PALETA DE CORES BASE
# =============================================================================
# Estas constantes definem os tons fundamentais do tema.
# Todos os estilos abaixo derivam destas definições.
# Alterar aqui afeta toda a aplicação — use com cuidado.

# Cor de fundo principal das janelas e painéis.
# #1a1a2e: azul-marinho muito escuro. Escolhido por ser menos "chato" que
# preto puro (#000000) e criar profundidade visual sem cansar os olhos.
COLOR_BG_DARK = "#1a1a2e"

# Cor de fundo de widgets secundários (cartões, grupos, painéis internos).
# Ligeiramente mais clara que BG_DARK para criar hierarquia visual sem contraste agressivo.
COLOR_BG_MEDIUM = "#16213e"

# Cor de fundo de elementos interativos em repouso (botões, campos de texto).
COLOR_BG_LIGHT = "#0f3460"

# Cor principal do texto — branco suave.
# Evitamos branco puro (#ffffff) porque no fundo escuro causa vibração visual
# (fenômeno conhecido como "irradiação simultânea"). #e2e8f0 é mais confortável.
COLOR_TEXT_PRIMARY = "#e2e8f0"

# Cor de texto secundário — para legendas, valores menos importantes, placeholders.
COLOR_TEXT_SECONDARY = "#94a3b8"

# Cor de destaque — azul ciano vibrante.
# Usado em bordas de foco, indicadores ativos e elementos de ação primária.
COLOR_ACCENT = "#38bdf8"

# Cor de sucesso — verde para estados positivos (mão detectada, sessão ativa, Regular).
COLOR_SUCCESS = "#22c55e"

# Cor de atenção — amarelo para estados de alerta (classificação "Bom", regularidade moderada).
COLOR_WARNING = "#eab308"

# Cor de perigo — vermelho para estados críticos (mão fechada, erros, "Ruim").
COLOR_DANGER = "#ef4444"

# Cor de borda padrão — cinza escuro para separar seções sem agressividade visual.
COLOR_BORDER = "#334155"

# Cor de fundo dos cartões de métricas — levemente diferente do fundo médio
# para criar "elevação" visual sem usar sombras (que são custosas em PyQt6).
COLOR_CARD_BG = "#1e293b"


# =============================================================================
# FUNÇÃO PRINCIPAL DE APLICAÇÃO DO TEMA
# =============================================================================

def apply_dark_theme(app: QApplication) -> None:
    """
    Aplica o tema escuro profissional à instância do QApplication.

    Esta função deve ser chamada UMA ÚNICA VEZ, logo após a criação do
    QApplication e ANTES de criar qualquer janela ou widget. Isso garante
    que todos os elementos criados posteriormente já herdem o tema correto.

    O Qt propaga automaticamente a QPalette para todos os widgets filhos.
    Por isso, configurar apenas a paleta do QApplication é suficiente —
    não é necessário definir cor por widget individualmente.

    Parâmetros:
        app: A instância do QApplication criada em app_pyqt.py.
             Deve ser o objeto retornado por QApplication(sys.argv).

    Retorno:
        None. A modificação é feita diretamente no objeto app.
    """
    # Cria uma nova paleta de cores do zero para evitar herdar valores
    # inesperados do tema padrão do sistema operacional.
    palette = QPalette()

    # --- Definição da paleta de cores ---
    # O Qt organiza as cores por "grupo" (Normal, Disabled, Inactive) e por "papel" (Role).
    # Configuramos apenas o grupo Normal — os grupos Disabled e Inactive herdam
    # automaticamente com tonalidades suavizadas pelo Qt.

    # Fundo das janelas principais (QMainWindow, QDialog).
    palette.setColor(QPalette.ColorRole.Window, QColor(COLOR_BG_DARK))

    # Texto sobre o fundo de janelas — deve ter contraste suficiente com Window.
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLOR_TEXT_PRIMARY))

    # Fundo de widgets de entrada (QLineEdit, QTextEdit, QComboBox).
    # Usamos BG_MEDIUM para distinguir campos de texto do fundo da janela.
    palette.setColor(QPalette.ColorRole.Base, QColor(COLOR_BG_MEDIUM))

    # Fundo alternado em listas e tabelas (linhas pares × ímpares).
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLOR_BG_LIGHT))

    # Cor do texto dentro de campos de entrada (QLineEdit, QTextEdit).
    palette.setColor(QPalette.ColorRole.Text, QColor(COLOR_TEXT_PRIMARY))

    # Fundo de botões (QPushButton).
    palette.setColor(QPalette.ColorRole.Button, QColor(COLOR_BG_LIGHT))

    # Texto sobre botões — claro para contraste com BG_LIGHT.
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLOR_TEXT_PRIMARY))

    # Cor de destaque: fundo de itens selecionados, barra de progresso, etc.
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLOR_ACCENT))

    # Texto sobre fundo de destaque — escuro para garantir legibilidade
    # quando o item está selecionado (contraste com o azul ciano do Highlight).
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#0f172a"))

    # Cor de texto em campos desabilitados — cinza mais escuro para indicar
    # visualmente que o campo não está disponível.
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.WindowText,
        QColor("#475569"),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        QColor("#475569"),
    )

    # Cor usada para dicas de ferramentas (tooltips).
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(COLOR_BG_LIGHT))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(COLOR_TEXT_PRIMARY))

    # Aplica a paleta configurada a toda a aplicação.
    # Este é o único ponto onde a paleta precisa ser definida —
    # todos os widgets criados posteriormente a herdarão automaticamente.
    app.setPalette(palette)

    # Aplica uma folha de estilos (StyleSheet) global para refinar
    # elementos que a QPalette não controla diretamente.
    # StyleSheet tem precedência sobre QPalette para os mesmos widgets.
    app.setStyleSheet(_build_global_stylesheet())


def _build_global_stylesheet() -> str:
    """
    Constrói e retorna a folha de estilos CSS global da aplicação.

    O Qt usa uma sintaxe semelhante ao CSS padrão para estilizar widgets.
    Esta função centraliza todos os estilos que precisam de mais controle
    do que a QPalette oferece (bordas, border-radius, padding, hover, etc.).

    Retorno:
        str: String contendo toda a folha de estilos no formato Qt StyleSheet.
    """
    return f"""
        /* ── Janela principal ── */
        QMainWindow {{
            background-color: {COLOR_BG_DARK};
        }}

        /* ── Widgets genéricos ── */
        QWidget {{
            background-color: {COLOR_BG_DARK};
            color: {COLOR_TEXT_PRIMARY};
            font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', sans-serif;
            font-size: 13px;
        }}

        /* ── Grupos de widgets (QGroupBox) ──
           Usados como contêineres visuais para cada seção do layout.
           border-radius dá aspecto moderno sem ser excessivo. */
        QGroupBox {{
            background-color: {COLOR_BG_MEDIUM};
            border: 1px solid {COLOR_BORDER};
            border-radius: 8px;
            margin-top: 12px;
            padding: 8px;
            font-weight: bold;
            font-size: 12px;
            color: {COLOR_TEXT_SECONDARY};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 6px;
            color: {COLOR_ACCENT};
        }}

        /* ── Botões principais ──
           Estilo base com cantos arredondados. Os estados hover e pressed
           fornecem feedback visual imediato ao clique. */
        QPushButton {{
            background-color: {COLOR_BG_LIGHT};
            color: {COLOR_TEXT_PRIMARY};
            border: 1px solid {COLOR_BORDER};
            border-radius: 6px;
            padding: 8px 18px;
            font-weight: bold;
            min-height: 32px;
        }}
        QPushButton:hover {{
            background-color: {COLOR_ACCENT};
            color: #0f172a;
            border-color: {COLOR_ACCENT};
        }}
        QPushButton:pressed {{
            background-color: #0284c7;
            color: #ffffff;
        }}
        QPushButton:disabled {{
            background-color: #1e293b;
            color: #475569;
            border-color: #1e293b;
        }}

        /* ── Campos de texto (QLineEdit) ── */
        QLineEdit {{
            background-color: {COLOR_BG_MEDIUM};
            color: {COLOR_TEXT_PRIMARY};
            border: 1px solid {COLOR_BORDER};
            border-radius: 5px;
            padding: 5px 8px;
            min-height: 28px;
        }}
        QLineEdit:focus {{
            border-color: {COLOR_ACCENT};
        }}
        QLineEdit:disabled {{
            color: #475569;
            background-color: #0f172a;
        }}

        /* ── ComboBox (listas suspensas) ── */
        QComboBox {{
            background-color: {COLOR_BG_MEDIUM};
            color: {COLOR_TEXT_PRIMARY};
            border: 1px solid {COLOR_BORDER};
            border-radius: 5px;
            padding: 4px 8px;
            min-height: 28px;
        }}
        QComboBox:focus {{
            border-color: {COLOR_ACCENT};
        }}
        QComboBox QAbstractItemView {{
            background-color: {COLOR_BG_MEDIUM};
            color: {COLOR_TEXT_PRIMARY};
            selection-background-color: {COLOR_ACCENT};
            selection-color: #0f172a;
            border: 1px solid {COLOR_BORDER};
        }}

        /* ── SpinBox (campos numéricos incrementais) ── */
        QSpinBox {{
            background-color: {COLOR_BG_MEDIUM};
            color: {COLOR_TEXT_PRIMARY};
            border: 1px solid {COLOR_BORDER};
            border-radius: 5px;
            padding: 4px 8px;
            min-height: 28px;
        }}
        QSpinBox:focus {{
            border-color: {COLOR_ACCENT};
        }}

        /* ── Área de texto (QTextEdit) — usada no LogWidget ── */
        QTextEdit {{
            background-color: #0d1117;
            color: {COLOR_TEXT_SECONDARY};
            border: 1px solid {COLOR_BORDER};
            border-radius: 5px;
            padding: 4px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 11px;
        }}

        /* ── Barras de rolagem ──
           Finas e discretas para não competir com o conteúdo principal. */
        QScrollBar:vertical {{
            background: {COLOR_BG_DARK};
            width: 8px;
            border-radius: 4px;
        }}
        QScrollBar::handle:vertical {{
            background: {COLOR_BORDER};
            border-radius: 4px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {COLOR_ACCENT};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}

        /* ── Separadores horizontais (QFrame::HLine) ── */
        QFrame[frameShape="4"] {{
            color: {COLOR_BORDER};
            max-height: 1px;
        }}

        /* ── Labels genéricos ── */
        QLabel {{
            color: {COLOR_TEXT_PRIMARY};
            background: transparent;
        }}

        /* ── Dicas de ferramentas (tooltips) ── */
        QToolTip {{
            background-color: {COLOR_BG_LIGHT};
            color: {COLOR_TEXT_PRIMARY};
            border: 1px solid {COLOR_ACCENT};
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
        }}
    """


# =============================================================================
# CONSTANTES DE ESTILO REUTILIZÁVEIS
# =============================================================================
# Estas strings de StyleSheet são importadas pelos widgets individuais
# e aplicadas via widget.setStyleSheet(CONSTANTE). Centralizar aqui evita
# duplicação de código e garante consistência visual entre todos os widgets.

# Estilo base para cartões de métricas (FPS, CPU, RAM, estado da mão).
# Usado por: ui/metrics_widget.py → MetricsWidget
# QFrame com fundo levemente mais claro que o painel e borda sutil para "elevação".
CARD_STYLE: str = f"""
    .QFrame {{
        background-color: {COLOR_CARD_BG};
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        padding: 8px;
    }}
"""

# Estilo para o título dentro de cartões de métricas (ex: "FPS", "CPU").
# Texto pequeno, cor secundária — não deve competir com o valor principal.
# Usado por: ui/metrics_widget.py → rótulos de cartões
LABEL_TITLE_STYLE: str = f"""
    QLabel {{
        color: {COLOR_TEXT_SECONDARY};
        font-size: 11px;
        font-weight: normal;
        background: transparent;
        border: none;
    }}
"""

# Estilo para o valor numérico principal dentro de cartões (ex: "58.3", "24%").
# Texto grande em negrito — deve ser o elemento mais legível do cartão.
# Usado por: ui/metrics_widget.py → valores de cartões
LABEL_VALUE_STYLE: str = f"""
    QLabel {{
        color: {COLOR_TEXT_PRIMARY};
        font-size: 22px;
        font-weight: bold;
        background: transparent;
        border: none;
    }}
"""

# Estilo para cartão de estado da mão quando MÃO ABERTA.
# Fundo verde suave — indicador positivo, não agressivo.
# Usado por: ui/metrics_widget.py → cartão de estado
CARD_HAND_OPEN_STYLE: str = f"""
    .QFrame {{
        background-color: #14532d;
        border: 1px solid {COLOR_SUCCESS};
        border-radius: 8px;
        padding: 8px;
    }}
"""

# Estilo para cartão de estado da mão quando MÃO FECHADA.
# Fundo vermelho escuro — indicador de alerta clínico.
# Usado por: ui/metrics_widget.py → cartão de estado
CARD_HAND_CLOSED_STYLE: str = f"""
    .QFrame {{
        background-color: #7f1d1d;
        border: 1px solid {COLOR_DANGER};
        border-radius: 8px;
        padding: 8px;
    }}
"""

# Estilo para o texto de estado da mão (grande, centralizado, em negrito).
# Usado por: ui/metrics_widget.py → label dentro do cartão de estado
LABEL_HAND_STATE_STYLE: str = f"""
    QLabel {{
        color: {COLOR_TEXT_PRIMARY};
        font-size: 18px;
        font-weight: bold;
        background: transparent;
        border: none;
    }}
"""

# Estilo para o cabeçalho da sessão (SessionHeaderWidget).
# Fundo diferenciado para separar visualmente do restante do layout.
# Usado por: ui/session_header.py → container principal
SESSION_HEADER_STYLE: str = f"""
    .QWidget {{
        background-color: {COLOR_BG_MEDIUM};
        border-bottom: 2px solid {COLOR_ACCENT};
    }}
"""

# Estilo para labels de título de seção dentro do cabeçalho.
# Texto em destaque, cor de acento — identifica claramente o propósito do campo.
# Usado por: ui/session_header.py → rótulos dos campos
LABEL_SECTION_TITLE_STYLE: str = f"""
    QLabel {{
        color: {COLOR_ACCENT};
        font-size: 12px;
        font-weight: bold;
        background: transparent;
    }}
"""

# Estilo para o cartão de cada dedo (FingerCardWidget).
# Mais compacto que CARD_STYLE — 5 cartões ficam lado a lado no layout.
# Usado por: ui/finger_card_widget.py → container de cada dedo
FINGER_CARD_STYLE: str = f"""
    QGroupBox {{
        background-color: {COLOR_CARD_BG};
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        margin-top: 10px;
        padding: 6px;
    }}
    QGroupBox::title {{
        color: {COLOR_ACCENT};
        font-weight: bold;
        font-size: 12px;
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 0 4px;
    }}
"""

# Estilo para labels de valor clínico dentro dos cartões de dedo.
# Tamanho médio — legível mas não domina o cartão.
# Usado por: ui/finger_card_widget.py → TAM, velocidade, frequência
LABEL_CLINICAL_VALUE_STYLE: str = f"""
    QLabel {{
        color: {COLOR_TEXT_PRIMARY};
        font-size: 14px;
        font-weight: bold;
        background: transparent;
    }}
"""

# Estilo para labels de métrica secundária dentro dos cartões de dedo.
# Texto menor — informação de apoio ao valor principal.
# Usado por: ui/finger_card_widget.py → amplitude, regularidade
LABEL_CLINICAL_SECONDARY_STYLE: str = f"""
    QLabel {{
        color: {COLOR_TEXT_SECONDARY};
        font-size: 11px;
        background: transparent;
    }}
"""

# Estilo para o botão de ação primária (Iniciar Sessão).
# Destaque com cor de sucesso para indicar ação positiva e segura.
# Usado por: ui/main_window.py → botão Iniciar
BUTTON_PRIMARY_STYLE: str = f"""
    QPushButton {{
        background-color: #15803d;
        color: #ffffff;
        border: 1px solid {COLOR_SUCCESS};
        border-radius: 6px;
        padding: 8px 18px;
        font-weight: bold;
        min-height: 36px;
        font-size: 13px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_SUCCESS};
        color: #0f172a;
    }}
    QPushButton:pressed {{
        background-color: #166534;
    }}
    QPushButton:disabled {{
        background-color: #1e293b;
        color: #475569;
        border-color: #1e293b;
    }}
"""

# Estilo para o botão de ação destrutiva (Encerrar Sessão).
# Vermelho para sinalizar que esta ação encerra e não pode ser desfeita facilmente.
# Usado por: ui/main_window.py → botão Encerrar
BUTTON_DANGER_STYLE: str = f"""
    QPushButton {{
        background-color: #991b1b;
        color: #ffffff;
        border: 1px solid {COLOR_DANGER};
        border-radius: 6px;
        padding: 8px 18px;
        font-weight: bold;
        min-height: 36px;
        font-size: 13px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_DANGER};
        color: #ffffff;
    }}
    QPushButton:pressed {{
        background-color: #7f1d1d;
    }}
    QPushButton:disabled {{
        background-color: #1e293b;
        color: #475569;
        border-color: #1e293b;
    }}
"""
