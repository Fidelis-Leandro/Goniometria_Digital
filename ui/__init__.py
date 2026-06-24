"""
ui/__init__.py
==============

Este arquivo vazio transforma o diretório ui/ em um pacote Python,
permitindo importações com a notação:

    from ui.video_widget import VideoWidget
    from ui.metrics_widget import MetricsWidget
    from ui.main_window import MainWindow

Módulos contidos neste pacote (construídos incrementalmente):
    - video_widget.py       → VideoWidget(QLabel): exibe frame BGR em tempo real
    - metrics_widget.py     → MetricsWidget(QGroupBox): cartões FPS, CPU, estado da mão
    - plot_widget.py        → GoniometryPlotWidget: gráfico TAM 5 dedos (PyQtGraph)
    - finger_card_widget.py → FingerCardWidget + FingerCardsPanel: cartões individuais
    - session_header.py     → SessionHeaderWidget: formulário do paciente + cronômetro
    - log_widget.py         → LogWidget(QTextEdit): log de eventos com timestamp
    - main_window.py        → MainWindow(QMainWindow): janela principal e orquestração
"""
