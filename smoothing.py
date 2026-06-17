"""
smoothing.py — Suavização obrigatória EMA → Kalman
==================================================

Este módulo implementa a camada de suavização do sistema.

Pipeline por série temporal:
raw_angle → EMA → Kalman → smoothed_angle

As classes aqui são independentes de OpenCV e MediaPipe.
Elas operam apenas sobre valores numéricos, o que facilita testes e reutilização.
"""

from typing import Dict, Optional

# =============================================================================
# FILTRO DE SÉRIE ESCALAR
# =============================================================================


class SeriesFilter:
    """
    Filtro escalar para uma série temporal angular.

    Cada articulação de cada dedo recebe uma instância independente
    para manter seu próprio histórico e estado do Kalman.
    """

    def __init__(
        self,
        ema_alpha: float = 0.30,
        kalman_q: float = 0.01,
        kalman_r: float = 0.10,
    ):
        # Parâmetros da EMA.
        self.ema_alpha = ema_alpha

        # Parâmetros do Kalman escalar.
        self.q = kalman_q
        self.r = kalman_r

        # Estado interno.
        self._ema_value: Optional[float] = None
        self._x: Optional[float] = None
        self._p: float = 1.0
        self._k_gain: float = 1.0
        self._n_updates: int = 0

    def update(self, raw: float) -> float:
        """
        Processa um novo valor bruto e devolve o valor suavizado.

        Etapa 1 — EMA:
        reduz jitter de alta frequência entre frames.

        Etapa 2 — Kalman:
        modela a estimativa recursiva do valor verdadeiro
        e sua incerteza residual.
        """
        self._n_updates += 1

        # EMA
        if self._ema_value is None:
            self._ema_value = raw
        else:
            self._ema_value = (
                self.ema_alpha * raw
                + (1.0 - self.ema_alpha) * self._ema_value
            )

        ema_output = self._ema_value

        # Kalman
        if self._x is None:
            self._x = ema_output
            return float(self._x)

        p_minus = self._p + self.q
        self._k_gain = p_minus / (p_minus + self.r)
        self._x = self._x + self._k_gain * (ema_output - self._x)
        self._p = (1.0 - self._k_gain) * p_minus

        return float(self._x)

    @property
    def kalman_gain(self) -> float:
        """
        Retorna o último ganho de Kalman calculado.
        """
        return self._k_gain

    @property
    def stability(self) -> str:
        """
        Classifica a estabilidade atual do filtro com base no ganho de Kalman.
        """
        if self._k_gain < 0.15:
            return "stable"
        elif self._k_gain < 0.40:
            return "converging"
        else:
            return "unstable"

    @property
    def is_initialized(self) -> bool:
        """
        Informa se a série já foi inicializada com pelo menos uma amostra.
        """
        return self._x is not None

    def reset(self, seed_value: Optional[float] = None) -> None:
        """
        Reinicia o estado interno do filtro.
        """
        self._ema_value = seed_value
        self._x = seed_value
        self._p = 1.0
        self._k_gain = 1.0 if seed_value is None else 0.5
        self._n_updates = 0


# =============================================================================
# BANCO DE FILTROS
# =============================================================================

class GoniometryFilterBank:
    """
    Banco de filtros indexado por (dedo, articulação).

    Esta classe coordena todas as séries temporais do sistema
    e fornece uma API única para suavizar o dicionário completo de ângulos.
    """

    def __init__(
        self,
        ema_alpha: float = 0.30,
        kalman_q: float = 0.01,
        kalman_r: float = 0.10,
    ):
        self._ema_alpha = ema_alpha
        self._kalman_q = kalman_q
        self._kalman_r = kalman_r
        self._filters: Dict[str, SeriesFilter] = {}

    def update(self, finger: str, joint: str, raw_angle: float) -> float:
        """
        Atualiza uma série específica do banco.
        """
        key = f"{finger}_{joint}"

        if key not in self._filters:
            self._filters[key] = SeriesFilter(
                ema_alpha=self._ema_alpha,
                kalman_q=self._kalman_q,
                kalman_r=self._kalman_r,
            )

        return self._filters[key].update(raw_angle)

    def smooth_all(self, angles: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Suaviza todo o dicionário de ângulos de uma vez.

        O formato de entrada e saída é preservado para facilitar integração.
        """
        filtered: Dict[str, Dict[str, float]] = {}

        for finger, metrics in angles.items():
            filtered[finger] = {}
            for joint, raw in metrics.items():
                filtered[finger][joint] = round(self.update(finger, joint, raw), 2)

        return filtered

    def get_stability(self, finger: str, joint: str) -> str:
        """
        Retorna o estado qualitativo do filtro para uma série.
        """
        key = f"{finger}_{joint}"
        if key not in self._filters:
            return "uninitialized"
        return self._filters[key].stability

    def get_all_gains(self) -> Dict[str, float]:
        """
        Retorna o ganho de Kalman de todas as séries ativas.
        """
        return {k: f.kalman_gain for k, f in self._filters.items()}

    def reset_finger(self, finger: str) -> None:
        """
        Reinicia todas as séries associadas a um dedo.
        """
        for key, filt in self._filters.items():
            if key.startswith(finger):
                filt.reset()

    def reset_all(self, seed_angles: Optional[Dict] = None) -> None:
        """
        Reinicia todas as séries do banco.
        """
        if seed_angles is None:
            for filt in self._filters.values():
                filt.reset()
        else:
            for finger, metrics in seed_angles.items():
                for joint, val in metrics.items():
                    key = f"{finger}_{joint}"
                    if key in self._filters:
                        self._filters[key].reset(seed_value=val)

    def configure(
        self,
        ema_alpha: float = None,
        kalman_q: float = None,
        kalman_r: float = None,
    ) -> None:
        """
        Reconfigura os parâmetros globais do banco.

        A alteração limpa os filtros para que as novas séries já nasçam
        com a configuração atualizada.
        """
        if ema_alpha is not None:
            self._ema_alpha = ema_alpha
        if kalman_q is not None:
            self._kalman_q = kalman_q
        if kalman_r is not None:
            self._kalman_r = kalman_r

        self._filters.clear()

    @property
    def active_series_count(self) -> int:
        """
        Quantidade de séries atualmente ativas.
        """
        return len(self._filters)

    def __repr__(self) -> str:
        return (
            f"GoniometryFilterBank("
            f"alpha={self._ema_alpha}, "
            f"Q={self._kalman_q}, "
            f"R={self._kalman_r}, "
            f"series={self.active_series_count})"
        )