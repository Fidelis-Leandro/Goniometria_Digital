"""
workers/__init__.py
===================

Este arquivo vazio transforma o diretório workers/ em um pacote Python.

Um "pacote" no Python é simplesmente uma pasta que contém um arquivo
__init__.py. Isso permite importar os módulos internos com a notação:

    from workers.camera_worker import CameraWorker
    from workers.processing_worker import ProcessingWorker

Sem este arquivo, o Python não reconheceria workers/ como um pacote
e as importações acima falhariam com ModuleNotFoundError.

Módulos contidos neste pacote:
    - camera_worker.py    → CameraWorker(QThread): captura de frames da webcam
    - processing_worker.py → ProcessingWorker(QThread): MediaPipe + goniometria
"""
