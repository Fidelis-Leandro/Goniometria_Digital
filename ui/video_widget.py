"""
ui/video_widget.py — Widget de exibição de vídeo em tempo real
===============================================================

Este módulo implementa o VideoWidget: um QLabel especializado que recebe
frames NumPy BGR do ProcessingWorker e os exibe na tela com a menor
latência possível.

Responsabilidade única (princípio SRP):
    Este widget FAZ UMA COISA APENAS: transformar um array NumPy (formato
    de câmera/OpenCV) em uma imagem visível no Qt. Ele não processa pixels,
    não analisa a imagem, não calcula ângulos — apenas exibe.

    Toda a análise aconteceu antes, no ProcessingWorker. O VideoWidget
    recebe o resultado já pronto (frame com overlay desenhado) e o exibe.

Por que QLabel em vez de QWidget customizado?
    QLabel já tem suporte nativo para exibir QPixmap (imagens) de forma
    otimizada. Herdar de QLabel nos dá setPixmap(), setAlignment() e
    redimensionamento automático gratuitamente, sem precisar implementar
    paintEvent() do zero para desenhar a imagem base.
    Sobrescrevemos paintEvent() APENAS para adicionar o overlay de FPS
    em cima da imagem já renderizada pelo QLabel pai.

Fluxo de dados:
    ProcessingWorker
        → pyqtSignal result_ready(ProcessingResult)
        → MainWindow._on_result()
        → video_widget.update_frame(result.frame_overlay)  ← entrada deste widget
        → cv2.cvtColor(BGR→RGB)
        → QImage → QPixmap → self.setPixmap()              ← saída: pixels na tela
"""

from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy

import config


class VideoWidget(QLabel):
    """
    Widget de exibição de vídeo em tempo real para a interface de goniometria.

    Herda de QLabel para aproveitar o suporte nativo a QPixmap, adicionando:
    - Conversão automática de formato BGR (OpenCV) para RGB (Qt).
    - Overlay de FPS desenhado via QPainter, sem impacto na imagem principal.
    - Estado de "sem sinal" com mensagem visual quando a câmera falha.
    - Redimensionamento proporcional da imagem ao redimensionar a janela.

    Uso típico na MainWindow:
        self.video_widget = VideoWidget()
        layout.addWidget(self.video_widget)
        processing_worker.result_ready.connect(
            lambda result: self.video_widget.update_frame(result.frame_overlay)
        )
        camera_worker.camera_error.connect(self.video_widget.set_no_signal)
    """

    def __init__(self, parent=None) -> None:
        """
        Inicializa o VideoWidget com configurações visuais padrão.

        Configura tamanho mínimo, alinhamento, política de redimensionamento
        e estado inicial ("sem sinal"). Não abre câmera nem processa nenhum dado.

        Parâmetros:
            parent: Widget Qt pai (opcional). Geralmente o container do layout.
        """
        super().__init__(parent)

        # Tamanho mínimo garantido para o widget — abaixo disso o layout
        # não permite que a janela seja encolhida.
        # 480×360 é o menor tamanho que ainda permite visualizar o overlay
        # goniométrico com os labels de ângulo legíveis.
        self.setMinimumSize(480, 360)

        # Centraliza o conteúdo (pixmap) dentro do espaço do QLabel.
        # Sem isso, a imagem ficaria colada no canto superior esquerdo
        # quando o widget for maior que o frame.
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Permite que o widget cresça e encolha livremente no layout,
        # mas respeita o tamanho mínimo definido acima.
        # Expanding em ambas as direções permite que o widget preencha
        # o espaço disponível na coluna esquerda do layout principal.
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        # Fundo preto escuro enquanto não há frame disponível.
        # Combina com o tema escuro da aplicação e não pisca ao
        # exibir o primeiro frame real.
        self.setStyleSheet("QLabel { background-color: #0d1117; }")

        # Armazena o FPS atual para ser desenhado no paintEvent.
        # Inicializado como None para indicar que ainda não há leitura de FPS.
        self._fps: Optional[float] = None

        # Flag que indica se o widget está no estado "sem sinal".
        # Controla qual texto é exibido quando não há frame disponível.
        self._no_signal: bool = True

        # Exibe o estado inicial de "sem sinal" imediatamente.
        self.set_no_signal()

    # =========================================================================
    # ATUALIZAÇÃO DO FRAME DE VÍDEO
    # =========================================================================

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        """
        Recebe um frame BGR do ProcessingWorker e o exibe no widget.

        Este é o método crítico de desempenho do widget. Ele é chamado a
        cada frame processado (~30 vezes/segundo) e deve ser rápido.

        Sequência obrigatória de conversão:
            BGR (OpenCV/câmera) → RGB (Qt) → QImage → QPixmap → tela

        Por que BGR → RGB é obrigatório?
            OpenCV usa a ordem Blue-Green-Red por herança histórica do padrão
            DirectShow do Windows. O Qt usa Red-Green-Blue (padrão moderno).
            Sem essa conversão, todas as cores ficam invertidas: pele humana
            aparece azulada, texto vermelho aparece azul, etc.

        Por que bytes_per_line é crítico?
            QImage precisa saber quantos bytes existem por linha de pixels.
            Para uma imagem de largura W com 3 canais (RGB), cada linha tem
            exatamente W*3 bytes. Se omitirmos este parâmetro, o Qt pode
            assumir um valor diferente (baseado em alinhamento de memória),
            fazendo a imagem aparecer distorcida diagonalmente — um bug
            sutil e difícil de diagnosticar.

        Por que usar contiguous() antes de criar QImage?
            Arrays NumPy nem sempre são contíguos na memória (ex: após
            operações de slice ou reshape). O QImage espera dados
            contíguos. Garantimos isso explicitamente.

        Parâmetros:
            frame_bgr: Array NumPy de shape (altura, largura, 3), dtype uint8,
                       em formato BGR. Geralmente é o frame_overlay do ProcessingResult.
        """
        # Sai do estado "sem sinal" ao receber um frame válido.
        self._no_signal = False

        # Converte BGR → RGB porque o Qt espera canais na ordem R-G-B.
        # cv2.cvtColor é otimizado internamente com SIMD — muito mais rápido
        # que inverter os canais manualmente com numpy slicing.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Garante que o array está contíguo na memória antes de criar o QImage.
        # Arrays não-contíguos causam leituras incorretas de pixels pelo Qt.
        frame_rgb = np.ascontiguousarray(frame_rgb)

        # Extrai dimensões para calcular bytes_per_line.
        height, width, channels = frame_rgb.shape

        # bytes_per_line: número de bytes em uma única linha horizontal da imagem.
        # Para RGB sem padding, é sempre largura × 3 canais.
        # Este valor DEVE ser passado explicitamente ao QImage — não confiar
        # no valor padrão, que pode diferir em sistemas com alinhamento de memória.
        bytes_per_line: int = channels * width

        # Cria o QImage referenciando diretamente a memória do array NumPy.
        # Format_RGB888 = 3 bytes por pixel, ordem R-G-B, sem canal alpha.
        # ATENÇÃO: o array frame_rgb deve permanecer em memória enquanto o
        # QImage existir. Como convertemos para QPixmap logo abaixo, isso é seguro.
        q_image = QImage(
            frame_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

        # Converte QImage → QPixmap (formato otimizado para exibição em tela).
        # QPixmap é mantido em memória de vídeo (GPU quando disponível),
        # enquanto QImage vive na memória principal (CPU). A conversão é feita
        # uma vez aqui e o QPixmap resultante é exibido sem custo adicional.
        pixmap = QPixmap.fromImage(q_image)

        # Redimensiona o pixmap para caber no widget atual, mantendo a proporção.
        # KeepAspectRatio: nunca distorce a imagem, adiciona barras pretas se necessário.
        # SmoothTransformation: usa interpolação bilinear — mais lenta que Fast,
        # mas produz imagem sem serrilhado (aliasing), especialmente ao reduzir.
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Atualiza o conteúdo do QLabel com o novo frame.
        # setPixmap() agenda automaticamente um repaint — não precisamos chamar
        # update() ou repaint() manualmente.
        self.setPixmap(scaled_pixmap)

    # =========================================================================
    # OVERLAY DE FPS
    # =========================================================================

    def set_fps(self, fps: float) -> None:
        """
        Armazena o valor de FPS para ser desenhado no próximo paintEvent.

        Por que armazenar em vez de desenhar imediatamente?
            Desenhar sobre o pixmap diretamente (modificando o QPixmap) seria
            irreversível — o texto ficaria "queimado" na imagem e acumularia
            a cada atualização. Armazenar o valor e redesenhar via QPainter
            em paintEvent() garante que o texto sempre apareça limpo,
            sobre a imagem atual, sem modificar o pixmap original.

            Também evita double-draw: se chamássemos update() aqui, o widget
            seria repintado DUAS vezes por frame (uma por setPixmap em
            update_frame, outra aqui). Armazenar o valor e usar paintEvent
            consolida ambas as operações em um único ciclo de renderização.

        Parâmetros:
            fps: Valor atual de FPS do pipeline de processamento (float).
                 Recebido do CameraWorker via sinal fps_updated.
        """
        self._fps = fps
        # Solicita repaint apenas se houver um pixmap exibido.
        # Evita redesenhar desnecessariamente no estado "sem sinal".
        if self.pixmap() and not self.pixmap().isNull():
            self.update()

    def paintEvent(self, event) -> None:
        """
        Evento de pintura do Qt — chamado sempre que o widget precisa ser redesenhado.

        Sobrescrevemos paintEvent() para adicionar o overlay de FPS em cima
        do conteúdo padrão do QLabel (o pixmap). A sequência é:
            1. Chama super().paintEvent() para desenhar o pixmap normalmente.
            2. Desenha o texto de FPS por cima, usando QPainter.

        Por que sombra preta + texto branco?
            O texto branco puro (#FFFFFF) pode desaparecer sobre áreas claras
            da imagem (fundo claro da câmera, iluminação intensa). A sombra
            preta deslocada em 1px cria um contorno escuro que torna o texto
            legível em QUALQUER fundo — técnica padrão em HUDs de jogos e
            aplicações de vídeo.

        Parâmetros:
            event: QPaintEvent fornecido pelo Qt automaticamente.
                   Contém a região que precisa ser redesenhada (rect()).
        """
        # Primeiro, deixa o QLabel desenhar normalmente (o pixmap, o alinhamento,
        # o fundo). Sem esta chamada, o frame de vídeo desaparece.
        super().paintEvent(event)

        # Só sobrepõe o FPS se tivermos um valor válido para exibir.
        if self._fps is None:
            return

        # Inicia o QPainter sobre este widget (não sobre o pixmap —
        # pintar no pixmap seria permanente; pintar no widget é temporário
        # e redesenhado a cada paintEvent).
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # Formata o texto com uma casa decimal — "FPS: 28.7"
        fps_text: str = f"FPS: {self._fps:.1f}"

        # Fonte em negrito, tamanho 13 — legível sem ocupar muito espaço.
        font = QFont("Segoe UI", 13, QFont.Weight.Bold)
        painter.setFont(font)

        # Área do texto: canto superior direito com margem de 10px.
        # QRect(x, y, largura, altura) — largura 120 é suficiente para "FPS: XX.X"
        text_rect = QRect(self.width() - 130, 10, 120, 28)

        # --- Sombra preta deslocada 1 pixel ---
        # Deslocar o texto em (+1, +1) cria a ilusão de sombra projetada.
        shadow_rect = QRect(text_rect.x() + 1, text_rect.y() + 1,
                            text_rect.width(), text_rect.height())
        painter.setPen(QPen(QColor("#000000")))
        painter.drawText(shadow_rect, Qt.AlignmentFlag.AlignRight, fps_text)

        # --- Texto branco principal ---
        painter.setPen(QPen(QColor("#FFFFFF")))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignRight, fps_text)

        # Finaliza o QPainter — OBRIGATÓRIO para liberar o contexto de pintura.
        # Sem end(), o Qt pode deixar o dispositivo de pintura bloqueado,
        # causando artefatos visuais ou crashes em versões mais antigas do Qt.
        painter.end()

    # =========================================================================
    # ESTADO SEM SINAL
    # =========================================================================

    def set_no_signal(self, message: str = "") -> None:
        """
        Coloca o widget no estado visual de "sem sinal de câmera".

        Chamado quando:
        - O widget é inicializado (antes da câmera ser aberta).
        - O CameraWorker emite camera_error (câmera desconectada, driver falhou).
        - A sessão é encerrada e a câmera é liberada.

        Cria um QPixmap preto com texto centralizado explicando a situação,
        evitando que o widget fique vazio ou exiba conteúdo desatualizado.

        Parâmetros:
            message: Mensagem de erro opcional do CameraWorker para exibir
                     abaixo do texto padrão "Sem sinal de câmera".
                     Se vazio, exibe apenas a mensagem padrão.
        """
        self._no_signal = True
        self._fps = None

        # Cria um pixmap preto do mesmo tamanho atual do widget.
        # Se o widget ainda não tem tamanho definido (ex: antes do show()),
        # usa o tamanho mínimo configurado no __init__.
        w = max(self.width(), 480)
        h = max(self.height(), 360)

        # Cria um pixmap vazio (não inicializado) e o preenche de preto.
        no_signal_pixmap = QPixmap(w, h)
        no_signal_pixmap.fill(QColor("#0d1117"))

        # Inicia um QPainter sobre o pixmap para desenhar o texto.
        painter = QPainter(no_signal_pixmap)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # Linha principal: "📷 Sem sinal de câmera"
        # Fonte grande para ser visível mesmo com a janela reduzida.
        font_main = QFont("Segoe UI", 18, QFont.Weight.Bold)
        painter.setFont(font_main)
        painter.setPen(QPen(QColor("#64748b")))

        # Área central do pixmap para o texto principal.
        main_rect = QRect(0, h // 2 - 40, w, 40)
        painter.drawText(
            main_rect,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            "Sem sinal de câmera",
        )

        # Instrução secundária para o usuário.
        font_sub = QFont("Segoe UI", 12)
        painter.setFont(font_sub)
        painter.setPen(QPen(QColor("#334155")))

        sub_rect = QRect(0, h // 2 + 10, w, 30)
        painter.drawText(
            sub_rect,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            "Clique em Iniciar Sessão para ativar a câmera",
        )

        # Se há mensagem de erro específica do CameraWorker, exibe em vermelho.
        if message:
            font_err = QFont("Consolas", 10)
            painter.setFont(font_err)
            painter.setPen(QPen(QColor("#ef4444")))

            err_rect = QRect(20, h // 2 + 50, w - 40, 50)
            painter.drawText(
                err_rect,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
                | Qt.TextFlag.TextWordWrap,
                f"Erro: {message}",
            )

        # Finaliza o painter antes de usar o pixmap.
        painter.end()

        # Exibe o pixmap de "sem sinal" no QLabel.
        self.setPixmap(no_signal_pixmap)

    # =========================================================================
    # REDIMENSIONAMENTO RESPONSIVO
    # =========================================================================

    def resizeEvent(self, event) -> None:
        """
        Chamado pelo Qt sempre que o widget é redimensionado pelo usuário.

        Reescalamos o último pixmap exibido para preencher o novo tamanho
        do widget, mantendo a proporção. Sem isso, a imagem ficaria com o
        tamanho fixo do primeiro frame recebido — ao redimensionar a janela,
        apareceriam barras pretas desnecessárias ou a imagem ficaria cortada.

        Parâmetros:
            event: QResizeEvent fornecido pelo Qt com o novo tamanho (newSize)
                   e o tamanho anterior (oldSize).
        """
        super().resizeEvent(event)

        # Se estamos no estado "sem sinal", recriar o pixmap de erro
        # com o novo tamanho para preencher corretamente.
        if self._no_signal:
            self.set_no_signal()
            return

        # Se há um pixmap válido exibido, reescalá-lo ao novo tamanho.
        current_pixmap = self.pixmap()
        if current_pixmap and not current_pixmap.isNull():
            scaled = current_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
