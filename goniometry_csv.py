"""
goniometry_csv.py — Registro de Sessão Goniométrica em CSV
============================================================
Grava uma linha por frame com timestamp ISO-8601 e todos os ângulos calculados.
Modo append: não sobrescreve sessões anteriores.

Uso:
    logger = GoniometryCSVLogger("session_goniometry.csv")
    logger.log(frame_id=42, angles=gonio.compute_all(landmarks))
    logger.close()
"""

import csv
import os
from datetime import datetime
from typing import Dict, Any


# ─── Cabeçalho do CSV (ordem canônica) ────────────────────────────────────────

CSV_FIELDS = [
    "timestamp", "frame_id",
    "INDEX_MCP",  "INDEX_PIP",  "INDEX_DIP",  "INDEX_ABD",  "INDEX_TAM",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_ABD", "MIDDLE_TAM",
    "RING_MCP",   "RING_PIP",   "RING_DIP",   "RING_ABD",   "RING_TAM",
    "PINKY_MCP",  "PINKY_PIP",  "PINKY_DIP",  "PINKY_ABD",  "PINKY_TAM",
    "THUMB_MCP",  "THUMB_IP",
]


class GoniometryCSVLogger:
    """
    Logger de sessão goniométrica.

    Parâmetros
    ----------
    filepath : str
        Caminho do arquivo CSV. Se já existir, faz append (preserva histórico).
    """

    def __init__(self, filepath: str = "session_goniometry.csv"):
        self.filepath = filepath
        self._file_exists = os.path.isfile(filepath) and os.path.getsize(filepath) > 0
        self._file = open(filepath, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)

        # Escreve cabeçalho apenas se arquivo estiver vazio / novo
        if not self._file_exists:
            self._writer.writeheader()
            self._file.flush()

    def log(self, frame_id: int, angles: Dict[str, Dict[str, float]]) -> None:
        """
        Registra um frame no CSV.

        Parâmetros
        ----------
        frame_id : int
            Identificador sequencial do frame.
        angles : dict
            Saída de DigitalGoniometer.compute_all() — formato canônico.
        """
        row: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "frame_id":  frame_id,
        }

        # Dedos com MCP/PIP/DIP/ABD/TAM
        for finger in ("INDEX", "MIDDLE", "RING", "PINKY"):
            data = angles.get(finger, {})
            row[f"{finger}_MCP"] = round(data.get("MCP", 0.0), 2)
            row[f"{finger}_PIP"] = round(data.get("PIP", 0.0), 2)
            row[f"{finger}_DIP"] = round(data.get("DIP", 0.0), 2)
            row[f"{finger}_ABD"] = round(data.get("ABD", 0.0), 2)
            row[f"{finger}_TAM"] = round(data.get("TAM", 0.0), 2)

        # Polegar
        thumb = angles.get("THUMB", {})
        row["THUMB_MCP"] = round(thumb.get("MCP", 0.0), 2)
        row["THUMB_IP"]  = round(thumb.get("IP",  0.0), 2)

        self._writer.writerow(row)

    def flush(self) -> None:
        """Força escrita no disco (útil para monitoramento em tempo real)."""
        self._file.flush()

    def close(self) -> None:
        """Fecha o arquivo CSV. Chamar ao encerrar a sessão."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
