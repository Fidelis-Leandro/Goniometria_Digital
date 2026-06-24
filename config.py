"""
config.py — Configuração centralizada do sistema de Goniometria Digital
========================================================================

Este arquivo é o único lugar onde números e parâmetros de configuração
devem ser definidos. Nenhum valor numérico ("número mágico") deve aparecer
espalhado pelo restante do código.

Filosofia de design:
    Em sistemas de tempo real como este, alterar um parâmetro em um só
    lugar (aqui) e ter esse valor propagado para todos os módulos é
    fundamental para manutenção e calibração. Sem esse arquivo, seria
    necessário procurar e substituir valores em múltiplos arquivos, o
    que inevitavelmente gera inconsistências.

Categorias de constantes:
    1. Câmera e captura de vídeo
    2. Pipeline de suavização (EMA → Kalman)
    3. Detecção de mão (MediaPipe)
    4. Interface gráfica (PyQt6 / PyQtGraph)
    5. Gravação de sessão (CSV)
    6. Mapeamento clínico dos dedos
    7. Sistema de logs
"""

# =============================================================================
# 1. CÂMERA E CAPTURA DE VÍDEO
# =============================================================================

# Índice da câmera no sistema operacional.
# 0 = câmera padrão (geralmente a webcam integrada).
# Se o usuário tiver múltiplas câmeras, altere para 1, 2, etc.
CAMERA_INDEX: int = 0

# Resolução alvo da câmera.
# 1280×720 (HD) oferece boa qualidade para rastreamento de landmarks,
# mas consome mais CPU que 640×480. Ajuste se a máquina for lenta.
CAMERA_WIDTH: int = 1280
CAMERA_HEIGHT: int = 720

# Taxa de quadros por segundo solicitada ao driver da câmera.
# O driver pode não honrar exatamente este valor — o FPS real
# é medido e exibido em tempo real pelo CameraWorker.
TARGET_FPS: int = 30

# =============================================================================
# 2. PIPELINE DE SUAVIZAÇÃO (EMA → KALMAN)
# =============================================================================

# Fator de suavização da Média Móvel Exponencial (EMA).
# Valor entre 0 e 1: quanto maior, mais rápido o filtro reage ao movimento,
# mas menos suavização ocorre. Quanto menor, mais suave, mas com mais atraso.
# 0.30 foi calibrado experimentalmente para goniometria clínica em tempo real:
# suaviza tremidos de câmera sem introduzir lag perceptível nos movimentos lentos.
EMA_ALPHA: float = 0.30

# Parâmetros do Filtro de Kalman escalar.
#
# KALMAN_Q (ruído do processo): representa a incerteza no modelo de movimento.
# Valor pequeno (0.01) assume que o ângulo articular muda lentamente e suavemente.
# Aumentar Q faz o filtro reagir mais rápido a mudanças bruscas.
KALMAN_Q: float = 0.01

# KALMAN_R (ruído da medição): representa a incerteza na leitura dos landmarks.
# 0.10 indica que confiamos moderadamente na posição detectada pelo MediaPipe.
# Aumentar R faz o filtro confiar menos na medição e mais na estimativa anterior.
KALMAN_R: float = 0.10

# =============================================================================
# 3. DETECÇÃO DE MÃO (MEDIAPIPE HANDS)
# =============================================================================

# Confiança mínima para DETECTAR uma mão a partir do zero.
# Valor mais alto (0.70) reduz falsos positivos, mas pode perder detecções
# em condições de iluminação ruim. Ajuste entre 0.5 e 0.9.
MP_DETECT_CONF: float = 0.70

# Confiança mínima para RASTREAR uma mão já detectada entre frames.
# Pode ser menor que MP_DETECT_CONF porque rastrear é mais fácil que detectar.
# 0.50 mantém o rastreamento fluido mesmo com oclusões parciais dos dedos.
MP_TRACK_CONF: float = 0.50

# Número de frames consecutivos sem detecção de mão antes de resetar os filtros.
# A 30 FPS, 15 frames ≈ 500ms. Isso evita que o filtro de Kalman "lembre"
# de uma posição anterior quando a mão volta após uma oclusão longa.
NO_HAND_RESET_FRAMES: int = 15

# =============================================================================
# 4. INTERFACE GRÁFICA (PyQt6 / PyQtGraph)
# =============================================================================

# Tamanho do buffer circular para os gráficos de tempo real (PyQtGraph).
# 500 pontos a ~30 FPS = aproximadamente 16 segundos de histórico visível.
# Usar deque(maxlen=BUFFER_SIZE) garante que a memória não cresce indefinidamente.
BUFFER_SIZE: int = 500

# Tamanho máximo da fila entre CameraWorker e ProcessingWorker.
# SEMPRE deve ser 1 em sistemas de tempo real. Com maxsize=1:
# - Se a fila está cheia, o frame antigo é descartado.
# - O processador SEMPRE recebe o frame mais recente.
# - Latência permanece mínima, mesmo se o processamento for temporariamente lento.
QUEUE_SIZE: int = 1

# Intervalo em milissegundos para o QTimer que atualiza os gráficos PyQtGraph.
# 33ms ≈ 30 FPS de atualização visual — suave para o olho humano.
# Aumentar este valor reduz uso de CPU; diminuir deixa a interface mais fluida.
PANEL_REFRESH_MS: int = 33

# A cada quantos frames o overlay goniométrico é recalculado.
# Desenhar o esqueleto e os ângulos é custoso. Com OVERLAY_FRAME_INTERVAL = 3,
# o overlay é atualizado a cada 3 frames, economizando ~67% do custo de desenho
# sem impacto visual perceptível (o olho humano não distingue diferenças tão rápidas).
OVERLAY_FRAME_INTERVAL: int = 3

# =============================================================================
# 5. GRAVAÇÃO DE SESSÃO (CSV)
# =============================================================================

# A cada quantos frames os ângulos são gravados no arquivo CSV.
# CSV_LOG_INTERVAL = 3 com TARGET_FPS = 30 resulta em ~10 linhas/segundo,
# suficiente para análise clínica sem gerar arquivos excessivamente grandes.
CSV_LOG_INTERVAL: int = 3

# =============================================================================
# 6. MAPEAMENTO CLÍNICO DOS DEDOS
# =============================================================================

# Ordem de processamento dos dedos — deve ser mantida consistente
# em todos os módulos do sistema para evitar bugs de indexação.
# THUMB (polegar) é listado por último por ter anatomia diferente
# dos demais dedos (apenas MCP e IP, sem DIP ou ABD).
FINGERS: list[str] = ["INDEX", "MIDDLE", "RING", "PINKY", "THUMB"]

# Mapeamento de nomes técnicos em inglês para nomes clínicos em português.
# Usado na interface gráfica para exibir etiquetas legíveis ao profissional de saúde.
FINGER_NAMES_PT: dict[str, str] = {
    "INDEX":  "Indicador",
    "MIDDLE": "Médio",
    "RING":   "Anelar",
    "PINKY":  "Mínimo",
    "THUMB":  "Polegar",
}

# Cores em formato hexadecimal para cada dedo nos gráficos PyQtGraph.
# As cores foram escolhidas com alto contraste entre si no fundo escuro
# e com distinção adequada para usuários com daltonismo parcial
# (evita combinações vermelho/verde puras).
FINGER_COLORS: dict[str, str] = {
    "INDEX":  "#38bdf8",   # Azul claro — Indicador
    "MIDDLE": "#4ade80",   # Verde claro — Médio
    "RING":   "#facc15",   # Amarelo dourado — Anelar
    "PINKY":  "#f87171",   # Vermelho salmão — Mínimo
    "THUMB":  "#c084fc",   # Roxo claro — Polegar
}

# Cores para cada dedo no formato (R, G, B) com valores 0–255.
# Usado pelo PyQtGraph para definir a cor das curvas de plotagem,
# pois o PyQtGraph aceita tanto hex quanto tuplas RGB.
FINGER_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "INDEX":  (56, 189, 248),
    "MIDDLE": (74, 222, 128),
    "RING":   (250, 204, 21),
    "PINKY":  (248, 113, 113),
    "THUMB":  (192, 132, 252),
}

# =============================================================================
# 7. SISTEMA DE LOGS
# =============================================================================

# Diretório onde o arquivo de log da aplicação será salvo.
# Será criado automaticamente se não existir (ver app_pyqt.py).
LOG_DIR: str = "logs"

# Nome do arquivo de log da aplicação PyQt6.
# Separado do CSV de sessão — este contém eventos do sistema,
# erros e informações de inicialização, não dados clínicos.
LOG_FILENAME: str = "app_pyqt.log"

# Formato das mensagens de log.
# Inclui: data/hora, nível (INFO/WARNING/ERROR), nome do módulo e mensagem.
# Facilita rastrear qual módulo gerou cada evento durante depuração.
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# =============================================================================
# 8. JANELA PRINCIPAL
# =============================================================================

# Título exibido na barra de título da janela do sistema operacional.
APP_TITLE: str = "Goniometria Digital da Mão"

# Tamanho mínimo da janela principal em pixels (largura × altura).
# Garante que todos os widgets fiquem visíveis mesmo em monitores menores.
WINDOW_MIN_WIDTH: int = 1280
WINDOW_MIN_HEIGHT: int = 800
