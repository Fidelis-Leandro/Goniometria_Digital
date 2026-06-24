"""
ui/log_widget.py — Widget de log de eventos com timestamps
===========================================================

Este módulo implementa o LogWidget: uma área de texto somente leitura
que registra cronologicamente todos os eventos relevantes do sistema
(inicialização, erros, avisos, encerramento de sessão, etc.).

Responsabilidade:
    Receber mensagens de texto de qualquer componente da aplicação e
    exibi-las com timestamp "HH:MM:SS" no formato de log de sistema.
    Rola automaticamente para mostrar sempre a mensagem mais recente.

Por que um log em vez de QMessageBox para cada evento?
    QMessageBox bloqueia o programa aguardando o clique do usuário.
    Em tempo real (~30 FPS de processamento), qualquer bloqueio seria
    catastrófico — frames seriam perdidos e a câmera ficaria sem leitura.
    O LogWidget registra eventos sem interromper nenhum processamento.

Quem usa o LogWidget:
    - MainWindow: eventos de sessão ("Sessão iniciada", "Sessão encerrada").
    - CameraWorker (via sinal camera_error): erros de câmera.
    - ProcessingWorker (via sinal processing_error): erros de processamento.
    - Qualquer módulo futuro que precise comunicar eventos ao clínico.

Integração na MainWindow:
    self.log_widget = LogWidget()
    camera_worker.camera_error.connect(
        lambda msg: self.log_widget.log(f"ERRO CÂMERA: {msg}")
    )
    processing_worker.processing_error.connect(self.log_widget.log)
"""

from datetime import datetime
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import QSizePolicy, QTextEdit, QWidget


class LogWidget(QTextEdit):
    """
    Área de log de eventos do sistema com timestamps automáticos.

    Herda de QTextEdit para aproveitar o suporte nativo a texto multi-linha,
    rolagem automática e seleção de texto (útil para copiar mensagens de erro).
    Sobrescrevemos o comportamento padrão apenas para forçar somente leitura
    e aplicar o estilo visual do tema escuro.

    Características:
        - Somente leitura (o usuário não pode editar).
        - Rolagem automática para o fim ao receber nova mensagem.
        - Fonte monoespaçada para alinhamento consistente dos timestamps.
        - Altura máxima de 120px para não dominar o layout da janela.
        - Cada linha segue o formato: "HH:MM:SS  mensagem"

    Exemplo de conteúdo:
        14:35:12  Aplicação iniciada.
        14:35:15  Sessão iniciada — Paciente: João Silva | Mão: Direita | Sessão 1
        14:36:02  ERRO CÂMERA: Câmera perdida após 10 frames inválidos.
        14:36:03  Sessão encerrada. Duração: 00:00:48
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Inicializa o LogWidget com estilo visual e configurações de comportamento.

        Parâmetros:
            parent: Widget pai Qt (opcional). Geralmente o container do layout.
        """
        super().__init__(parent)

        # Somente leitura: o clínico pode visualizar e copiar, mas não editar.
        # Edição acidental poderia apagar mensagens de erro importantes.
        self.setReadOnly(True)

        # Altura máxima de 120px — o log não deve dominar o layout.
        # Com fonte tamanho 9, cada linha tem ~14px de altura:
        # 120px ÷ 14px ≈ 8 linhas visíveis simultaneamente.
        self.setMaximumHeight(120)

        # Política de tamanho: expande horizontalmente, altura controlada pelo max.
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Maximum,
        )

        # Fonte monoespaçada para alinhamento dos timestamps.
        # Consolas: disponível no Windows. Courier New: fallback universal.
        # Tamanho 9: compacto o suficiente para 8 linhas em 120px de altura.
        mono_font = QFont()
        mono_font.setFamilies(["Consolas", "Courier New", "Monospace"])
        mono_font.setPointSize(9)
        self.setFont(mono_font)

        # Estilo visual compatível com o tema escuro da aplicação.
        # Fundo quase preto (#0d1117) para contraste máximo com o texto cinza.
        # Borda sutil para demarcar a área do log sem chamar atenção excessiva.
        self.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117;
                color: #94a3b8;
                border: 1px solid #1e293b;
                border-radius: 4px;
                padding: 4px 6px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 9pt;
            }
            QScrollBar:vertical {
                background: #0d1117;
                width: 6px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #334155;
                border-radius: 3px;
                min-height: 16px;
            }
            QScrollBar::handle:vertical:hover {
                background: #38bdf8;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        # Registra uma mensagem inicial para confirmar que o widget está funcional.
        # Isso também serve como validação visual durante o desenvolvimento.
        self.log("Sistema de Goniometria Digital inicializado.")

    # =========================================================================
    # INTERFACE PÚBLICA
    # =========================================================================

    def log(self, message: str) -> None:
        """
        Adiciona uma mensagem ao log com timestamp de hora atual.

        Formata a mensagem como "HH:MM:SS  mensagem" e a insere no final
        do conteúdo atual. Rola automaticamente para mostrar a nova entrada.

        Este método é seguro para ser conectado diretamente a pyqtSignal(str),
        como os sinais camera_error e processing_error dos workers. O Qt garante
        que chamadas via sinal sempre acontecem na thread principal (onde o
        widget vive), mesmo que o sinal tenha sido emitido de uma QThread.

        Parâmetros:
            message: Texto da mensagem a registrar. Pode conter qualquer string,
                     incluindo mensagens de erro com detalhes técnicos.
        """
        # Obtém o timestamp atual no formato de hora HH:MM:SS.
        # Não incluímos a data porque todas as entradas do log são da mesma
        # sessão de uso do programa. A data pode ser inferida do nome do arquivo CSV.
        timestamp: str = datetime.now().strftime("%H:%M:%S")

        # Formata a linha completa: timestamp + dois espaços de separação + mensagem.
        # Dois espaços (não um tab) para separação visual consistente em fontes
        # monoespaçadas onde tab pode ter largura variável conforme configuração do SO.
        log_line: str = f"{timestamp}  {message}"

        # Inserir no QTextEdit: usamos append() em vez de setPlainText() porque
        # append() ADICIONA ao conteúdo existente sem apagar o histórico.
        # setPlainText() substituiria tudo — perdendo todo o log anterior.
        self.append(log_line)

        # Rola a barra de rolagem para o fim para mostrar a mensagem mais recente.
        # Sem isso, o scroll permaneceria na posição anterior — a nova mensagem
        # seria inserida no fim mas ficaria fora da área visível.
        self._scroll_to_bottom()

    def log_error(self, message: str) -> None:
        """
        Adiciona uma mensagem de erro ao log com prefixo visual de destaque.

        Variante de log() para erros críticos. Prefixa automaticamente com
        "❌ ERRO:" para destaque visual imediato — o clínico identifica erros
        sem precisar ler linha por linha.

        Parâmetros:
            message: Descrição do erro. Ex: "Câmera perdida após 10 frames."
        """
        self.log(f"❌ ERRO: {message}")

    def log_warning(self, message: str) -> None:
        """
        Adiciona uma mensagem de aviso ao log com prefixo visual de alerta.

        Para situações que não são erros fatais mas merecem atenção:
        ex: performance degradada, dados fora do range clínico esperado.

        Parâmetros:
            message: Descrição do aviso.
        """
        self.log(f"⚠️ AVISO: {message}")

    def log_success(self, message: str) -> None:
        """
        Adiciona uma mensagem de confirmação positiva ao log.

        Para eventos bem-sucedidos importantes: sessão iniciada, PDF gerado,
        CSV exportado. Distingue visualmente de mensagens informativas comuns.

        Parâmetros:
            message: Descrição do evento bem-sucedido.
        """
        self.log(f"✅ {message}")

    def clear_log(self) -> None:
        """
        Apaga todo o conteúdo do log e insere uma mensagem de reinicialização.

        Chamado pela MainWindow ao iniciar uma nova sessão, para que o log
        da sessão anterior não polua o log da nova. A mensagem de reinicialização
        garante que o log nunca apareça completamente vazio — evita ambiguidade
        entre "log limpo" e "nenhum evento ocorreu".
        """
        # clear() apaga todo o conteúdo do QTextEdit de uma vez.
        # Mais eficiente que setPlainText("") porque não gera evento de mudança
        # de conteúdo que poderia acionar outros slots desnecessariamente.
        self.clear()
        self.log("Log reiniciado para nova sessão.")

    # =========================================================================
    # MÉTODOS INTERNOS
    # =========================================================================

    def _scroll_to_bottom(self) -> None:
        """
        Rola a barra de rolagem vertical para mostrar a última linha do log.

        Usa QTextCursor posicionado no End do documento para garantir que
        a rolagem seja para o final REAL do conteúdo, não apenas para uma
        posição aproximada calculada pela barra de rolagem.

        Por que QTextCursor em vez de verticalScrollBar().setValue(maximum())?
            O maximum() da scrollbar pode estar desatualizado no momento em que
            o novo texto foi inserido — o Qt ainda não recalculou o layout.
            QTextCursor.End é sempre o fim real do documento.
        """
        # Move o cursor para o fim do documento.
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Define o cursor modificado no widget — isso causa a rolagem.
        self.setTextCursor(cursor)

        # ensureCursorVisible() garante que o cursor (agora no fim) esteja
        # dentro da área visível do widget, rolando se necessário.
        self.ensureCursorVisible()
