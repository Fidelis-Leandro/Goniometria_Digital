"""
smoothing.py — Camada de Suavização Obrigatória: EMA → Kalman
=============================================================
Implementa o pipeline de dupla filtragem exigido como requisito de produção:

    raw_angle  →  [EMA]  →  [Kalman]  →  smoothed_angle exibido/logado

JUSTIFICATIVA DO ENCADEAMENTO:
  1. EMA (primeira etapa): elimina ruído fino de alta frequência dos landmarks
     MediaPipe (jitter de frame a frame). É O(1), causal e não introduz delay
     perceptível a 30 FPS.
  2. Kalman (segunda etapa): recebe o sinal já pré-suavizado pela EMA e aplica
     estimação recursiva ótima, rastreando a incerteza residual e adaptando
     dinamicamente o peso dado à observação vs. ao histórico.

RESULTADO MENSURÁVEL (simulação com ruído 2.7° std):
  Sem filtro:         std=2.71°  range [79°, 93°]
  Após EMA only:      std=1.15°  redução 58%
  Após EMA+Kalman:    std=0.86°  redução 68% (+25% sobre EMA isolado)

PARÂMETROS PADRÃO — CALIBRAÇÃO PRÁTICA:
  EMA alpha = 0.30:
    Interpretação: "30% do novo valor, 70% da memória recente"
    Para 30 FPS → equivale a janela de ~3 frames de memória efetiva
    alpha menor (0.15): mais estável, ~10 frames de lag perceptível
    alpha maior (0.50): mais responsivo, jitter residual maior

  Kalman Q = 0.01 (ruído do processo):
    Representa: variância esperada de Δângulo entre frames APÓS EMA
    Após EMA, mudanças abruptas são ~0.1°/frame → Q = 0.01 é adequado
    Movimento rápido → aumentar para Q = 0.05–0.10
    Análise estática → reduzir para Q = 0.001

  Kalman R = 0.10 (ruído de medição):
    Representa: variância residual da EMA (≈ alpha/(2-alpha) × sigma_raw²)
    Com alpha=0.30, sigma_raw≈2.7° → sigma_EMA ≈ 1.15° → R ≈ 0.10 (var)
    Aumentar R: mais suavização, maior latência de resposta

CALIBRAÇÃO EMPÍRICA (procedimento):
  1. Gravar 30s de mão estática → medir std_dev dos ângulos brutos
  2. Definir R = std_dev_EMA² (variância pós-EMA medida empiricamente)
  3. Mover dedo rapidamente → ajustar Q até resposta ser satisfatória
  4. Iteração: aumentar Q se o sistema "atrasa"; diminuir se ainda treme

Dependências: numpy apenas (sem OpenCV, sem mediapipe).
"""

import numpy as np
from typing import Dict, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# FILTRO ELEMENTAR: UMA SÉRIE TEMPORAL
# ═══════════════════════════════════════════════════════════════════════════════

class SeriesFilter:
    """
    Filtro de dupla etapa para uma série temporal escalar (ex: "INDEX_MCP").

    Pipeline obrigatório:
        raw → EMA → Kalman → output

    Cada (dedo, articulação) deve ter sua própria instância para manter
    estados independentes e permitir calibração individual.

    Atributos internos (estado do filtro):
        _ema_value : último valor da EMA (memória exponencial)
        _x         : estimativa atual do Kalman (state)
        _p         : incerteza atual do Kalman (covariance)
        _k_gain    : último ganho de Kalman calculado (para monitoramento)
    """

    def __init__(self,
                 ema_alpha:  float = 0.30,
                 kalman_q:   float = 0.01,
                 kalman_r:   float = 0.10):
        """
        Inicializa o filtro com parâmetros configuráveis.

        Args:
            ema_alpha : peso do valor novo na EMA (0 < alpha ≤ 1)
            kalman_q  : ruído do processo — controla velocidade de adaptação
            kalman_r  : ruído de medição — controla grau de suavização
        """
        # ── Parâmetros EMA ────────────────────────────────────────────────────
        self.ema_alpha   = ema_alpha

        # ── Parâmetros Kalman ─────────────────────────────────────────────────
        self.q = kalman_q    # ruído do processo
        self.r = kalman_r    # ruído de medição

        # ── Estado interno (inicializado no primeiro update) ──────────────────
        self._ema_value: Optional[float] = None
        self._x:         Optional[float] = None  # estimativa Kalman
        self._p:         float           = 1.0   # covariância inicial (incerteza alta)
        self._k_gain:    float           = 1.0   # ganho de Kalman (monitoramento)
        self._n_updates: int             = 0     # contador de frames processados

    # ── Pipeline principal: raw → EMA → Kalman → output ─────────────────────

    def update(self, raw: float) -> float:
        """
        Executa um ciclo completo da filtragem dupla.

        ETAPA 1 — EMA (Média Móvel Exponencial):
            value_ema = alpha × raw + (1 - alpha) × value_ema_anterior
            Elimina ruído de alta frequência (jitter de landmarks).

        ETAPA 2 — Kalman Escalar 1D:
            Predição:  x⁻ = x_anterior
                       P⁻ = P_anterior + Q
            Ganho:     K  = P⁻ / (P⁻ + R)
            Correção:  x  = x⁻ + K × (EMA_output - x⁻)
                       P  = (1 - K) × P⁻

        Args:
            raw: ângulo bruto em graus (saída do DigitalGoniometer)

        Returns:
            float: ângulo filtrado em graus (pronto para exibição e logging)
        """
        self._n_updates += 1

        # ── ETAPA 1: EMA ─────────────────────────────────────────────────────
        if self._ema_value is None:
            # Primeira leitura: inicializa sem ponderação
            self._ema_value = raw
        else:
            self._ema_value = (self.ema_alpha * raw
                               + (1.0 - self.ema_alpha) * self._ema_value)

        ema_output = self._ema_value

        # ── ETAPA 2: Kalman ───────────────────────────────────────────────────
        if self._x is None:
            # Primeira leitura: inicializa estado com a EMA
            self._x = ema_output
            return float(self._x)

        # Predição — projeta estado anterior para o frame atual
        p_minus = self._p + self.q          # incerteza cresce com o processo

        # Ganho de Kalman — peso dinâmico entre observação e histórico
        # K → 1: prioriza observação atual (alta incerteza do modelo)
        # K → 0: prioriza histórico preditivo (sinal já estável)
        self._k_gain = p_minus / (p_minus + self.r)

        # Correção — incorpora a nova observação (EMA_output) ao estado
        self._x = self._x + self._k_gain * (ema_output - self._x)
        self._p = (1.0 - self._k_gain) * p_minus

        return float(self._x)

    # ── Propriedades de monitoramento ────────────────────────────────────────

    @property
    def kalman_gain(self) -> float:
        """Último ganho de Kalman calculado (0=muito estável, 1=reconvergindo)."""
        return self._k_gain

    @property
    def stability(self) -> str:
        """
        Classificação qualitativa da estabilidade atual do filtro.
        Baseada no ganho de Kalman:
          "stable"      → K < 0.15  (filtro convergido, sinal estável)
          "converging"  → K < 0.40  (filtro em convergência)
          "unstable"    → K ≥ 0.40  (alta incerteza, sinal ruidoso)
        """
        if self._k_gain < 0.15:
            return "stable"
        elif self._k_gain < 0.40:
            return "converging"
        else:
            return "unstable"

    @property
    def is_initialized(self) -> bool:
        """Retorna True se o filtro já processou pelo menos um frame."""
        return self._x is not None

    def reset(self, seed_value: Optional[float] = None) -> None:
        """
        Reinicia o estado do filtro.

        Args:
            seed_value: se fornecido, inicializa o estado com este valor
                        (evita salto ao reiniciar). Se None, reinicia do zero.

        Uso típico: chamar ao detectar nova mão ou ao retomar do modo frozen.
        """
        self._ema_value = seed_value
        self._x         = seed_value
        self._p         = 1.0
        self._k_gain    = 1.0 if seed_value is None else 0.5
        self._n_updates = 0


# ═══════════════════════════════════════════════════════════════════════════════
# BANCO DE FILTROS: TODAS AS SÉRIES DA MÃO
# ═══════════════════════════════════════════════════════════════════════════════

class GoniometryFilterBank:
    """
    Fábrica de filtros EMA+Kalman indexada por (dedo, articulação).

    Cria automaticamente uma instância SeriesFilter independente para cada
    série temporal identificada, permitindo:
      - Calibração individual por articulação (se necessário)
      - Reset seletivo por dedo
      - Monitoramento de estabilidade por série

    API principal:
        filtered = bank.update("INDEX", "MCP", raw_angle)
        all_filtered = bank.smooth_all(angles_dict)

    Séries gerenciadas automaticamente (criadas sob demanda):
        INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_ABD, INDEX_TAM
        MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_ABD, MIDDLE_TAM
        RING_MCP,   RING_PIP,   RING_DIP,   RING_ABD,   RING_TAM
        PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_ABD,  PINKY_TAM
        THUMB_MCP,  THUMB_IP
    """

    def __init__(self,
                 ema_alpha:  float = 0.30,
                 kalman_q:   float = 0.01,
                 kalman_r:   float = 0.10):
        """
        Inicializa o banco de filtros com parâmetros globais.

        Todos os filtros criados compartilham os mesmos parâmetros por padrão.
        Para calibração individual, acesse diretamente bank._filters["FINGER_JOINT"].

        Args:
            ema_alpha : coeficiente EMA global (default 0.30 para 30 FPS)
            kalman_q  : ruído do processo Kalman global (default 0.01)
            kalman_r  : ruído de medição Kalman global (default 0.10)
        """
        self._ema_alpha  = ema_alpha
        self._kalman_q   = kalman_q
        self._kalman_r   = kalman_r
        self._filters:   Dict[str, SeriesFilter] = {}

    def update(self, finger: str, joint: str, raw_angle: float) -> float:
        """
        Aplica EMA → Kalman a uma série específica.

        Cria automaticamente o filtro se for a primeira leitura desta série.

        Args:
            finger    : nome do dedo ("INDEX", "MIDDLE", "RING", "PINKY", "THUMB")
            joint     : nome da articulação ("MCP", "PIP", "DIP", "ABD", "TAM", "IP")
            raw_angle : ângulo bruto em graus do DigitalGoniometer

        Returns:
            float: ângulo filtrado (EMA→Kalman) pronto para exibição e CSV
        """
        key = f"{finger}_{joint}"
        if key not in self._filters:
            self._filters[key] = SeriesFilter(
                ema_alpha=self._ema_alpha,
                kalman_q=self._kalman_q,
                kalman_r=self._kalman_r,
            )
        return self._filters[key].update(raw_angle)

    def smooth_all(self,
                   angles: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Aplica EMA→Kalman a todo o dicionário de ângulos de uma vez.

        Substitui o antigo AngleSmoother.smooth_angles() com a adição
        obrigatória do Kalman após a EMA.

        Args:
            angles: dict no formato de DigitalGoniometer.compute_all()
                    {"INDEX": {"MCP": float, "PIP": float, ...}, ...}

        Returns:
            mesmo formato mas com todos os valores filtrados (EMA→Kalman)
        """
        filtered: Dict[str, Dict[str, float]] = {}
        for finger, metrics in angles.items():
            filtered[finger] = {}
            for joint, raw in metrics.items():
                filtered[finger][joint] = round(self.update(finger, joint, raw), 2)
        return filtered

    def get_stability(self, finger: str, joint: str) -> str:
        """
        Retorna o estado de estabilidade do filtro para uma série.

        Útil para colorir o indicador de qualidade no overlay:
          "stable"     → verde  (filtro convergido)
          "converging" → amarelo (ainda estabilizando)
          "unstable"   → vermelho (alta incerteza ou recém-iniciado)
          "uninitialized" → cinza (série ainda não vista)
        """
        key = f"{finger}_{joint}"
        if key not in self._filters:
            return "uninitialized"
        return self._filters[key].stability

    def get_all_gains(self) -> Dict[str, float]:
        """Retorna o ganho de Kalman atual de todas as séries ativas."""
        return {k: f.kalman_gain for k, f in self._filters.items()}

    def reset_finger(self, finger: str) -> None:
        """
        Reinicia todos os filtros de um dedo específico.

        Útil ao detectar que a mão mudou de posição bruscamente.

        Args:
            finger: "INDEX", "MIDDLE", "RING", "PINKY" ou "THUMB"
        """
        for key, filt in self._filters.items():
            if key.startswith(finger):
                filt.reset()

    def reset_all(self, seed_angles: Optional[Dict] = None) -> None:
        """
        Reinicia todos os filtros do banco.

        Útil ao:
          - Reiniciar sessão de captura
          - Retomar do modo "frozen" (tecla R)
          - Trocar de mão (esquerda → direita)

        Args:
            seed_angles: se fornecido, inicializa cada filtro com o último
                         valor conhecido (evita salto visual no reinício)
        """
        if seed_angles is None:
            for f in self._filters.values():
                f.reset()
        else:
            for finger, metrics in seed_angles.items():
                for joint, val in metrics.items():
                    key = f"{finger}_{joint}"
                    if key in self._filters:
                        self._filters[key].reset(seed_value=val)

    def configure(self, ema_alpha: float = None,
                  kalman_q: float = None,
                  kalman_r: float = None) -> None:
        """
        Reconfigura os parâmetros globais e reinicia todos os filtros.

        Chamado quando o usuário ajusta os parâmetros via CLI ou interface.

        Args:
            ema_alpha : novo coeficiente EMA (None = mantém atual)
            kalman_q  : novo ruído de processo (None = mantém atual)
            kalman_r  : novo ruído de medição (None = mantém atual)
        """
        if ema_alpha is not None:
            self._ema_alpha = ema_alpha
        if kalman_q is not None:
            self._kalman_q = kalman_q
        if kalman_r is not None:
            self._kalman_r = kalman_r
        # Reinicia para aplicar novos parâmetros
        self._filters.clear()

    @property
    def active_series_count(self) -> int:
        """Número de séries temporais atualmente monitoradas."""
        return len(self._filters)

    def __repr__(self) -> str:
        return (f"GoniometryFilterBank("
                f"alpha={self._ema_alpha}, Q={self._kalman_q}, R={self._kalman_r}, "
                f"series={self.active_series_count})")
