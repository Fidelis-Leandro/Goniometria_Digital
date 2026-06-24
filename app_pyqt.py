"""
app_pyqt.py — Ponto de entrada da interface Desktop (PyQt6)
==========================================================

Este é o NOVO ponto de entrada para a aplicação de Goniometria Digital,
substituindo a interface Web baseada em Streamlit.

O arquivo 'main.py' permanece intocado como fallback de testes (OpenCV puro).

Responsabilidades deste script:
1. Configurar o sistema de logging global (arquivo e console).
2. Otimizar renderização gráfica no Windows (desligando OpenGL se necessário).
3. Inicializar a aplicação Qt e aplicar o tema visual escuro.
4. Instanciar e exibir a MainWindow.
5. Capturar exceções fatais não tratadas para evitar encerramento silencioso.

Comandos de execução:
- Interface PyQt6 (nova)   : python app_pyqt.py
- Interface Streamlit (leg): streamlit run app.py
- Fallback OpenCV (teste)  : python main.py
"""

import logging
import os
import sys
import traceback

from PyQt6.QtWidgets import QApplication

import config
import themes
from ui.main_window import MainWindow

# Tenta carregar pyqtgraph. Em algumas máquinas Windows, o pyqtgraph tenta
# usar OpenGL e trava se os drivers de vídeo forem básicos.
try:
    import pyqtgraph as pg
    # Desativa OpenGL nativo por precaução. O renderizador por software (raster)
    # do PyQtGraph é extremamente rápido e mais que suficiente para os nossos
    # gráficos de linha 2D, e é 100% estável em qualquer PC.
    pg.setConfigOption("useOpenGL", False)
except ImportError:
    # Tratado dentro dos widgets que usam pyqtgraph.
    pass


def setup_logging() -> None:
    """
    Configura o logging global do sistema para console e arquivo.

    Por que configurar isso globalmente aqui no ponto de entrada?
        Qualquer módulo (MainWindow, CameraWorker, etc.) pode usar
        logging.getLogger(__name__) e automaticamente herdar essa formatação,
        sem precisar configurar o logger individualmente em cada arquivo.

    Formato:
        "2026-06-23 14:35:12,123 | INFO | ui.main_window | Sessão iniciada"
    """
    # Garante que o diretório de logs existe
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_file = os.path.join(config.LOG_DIR, "app.log")

    # Formato padronizado: data/hora | nível | módulo | mensagem
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    # Configura o logger raiz
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),  # Arquivo
            logging.StreamHandler(sys.stdout),                # Console (terminal)
        ]
    )


def exception_hook(exc_type, exc_value, exc_traceback) -> None:
    """
    Manipulador global de exceções não capturadas.

    Por que usar um sys.excepthook?
        Em aplicações PyQt, exceções lançadas dentro de slots ou sinais
        às vezes são engolidas pelo Qt e o programa crasha silenciosamente.
        O excepthook garante que NENHUMA exceção fatal passe despercebida:
        todas serão registradas no arquivo app.log com traceback completo
        antes de o programa encerrar.
    """
    # Se a exceção for interrupção pelo teclado (Ctrl+C no terminal),
    # não tratamos como erro fatal, apenas deixamos a aplicação fechar.
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Registra o erro fatal no logger com toda a pilha de chamadas (traceback)
    logger = logging.getLogger("sys.excepthook")
    logger.critical(
        "Exceção Fatal Não Capturada:\n",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


def main() -> None:
    """
    Ponto de entrada principal da aplicação.
    Configura o ambiente, cria a UI e inicia o loop de eventos do Qt.
    """
    # 1. Configura logging e captura global de exceções
    setup_logging()
    sys.excepthook = exception_hook

    logger = logging.getLogger("app_pyqt")
    logger.info("Inicializando Goniometria Digital (Interface PyQt6)...")

    # Envolve toda a execução do app em try/except para garantir que
    # problemas na inicialização sejam sempre logados.
    try:
        # 2. Cria a instância principal da aplicação Qt
        # sys.argv permite que a aplicação aceite parâmetros de linha de comando
        # (ex: parâmetros de estilo nativos do Qt)
        app = QApplication(sys.argv)

        # 3. Nome da aplicação (usado internamente pelo SO e pelo Qt)
        app.setApplicationName(config.APP_TITLE)

        # 4. Aplica o tema escuro padronizado em todos os componentes nativos
        themes.apply_dark_theme(app)

        # 5. Instancia a janela principal, que orquestra todo o resto
        window = MainWindow()

        # 6. Exibe a janela (maximiza para usar a tela inteira se necessário,
        #    mas show() padrão respeita os limites da tela).
        window.show()

        logger.info("Interface iniciada com sucesso. Loop de eventos Qt ativo.")

        # 7. Inicia o loop de eventos (bloqueante até a janela fechar)
        # sys.exit passa o código de retorno do app.exec() para o SO
        sys.exit(app.exec())

    except Exception as e:
        logger.critical("Falha fatal ao iniciar a aplicação: %s", e, exc_info=True)
        # Sai com código de erro 1
        sys.exit(1)


# Se este arquivo for executado diretamente pelo terminal (python app_pyqt.py)
if __name__ == "__main__":
    main()
