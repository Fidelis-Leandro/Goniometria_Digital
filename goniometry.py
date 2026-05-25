"""
goniometry.py — Módulo de Goniometria Digital Computacional
============================================================
Calcula ângulos articulares dos dedos a partir dos landmarks 3D do MediaPipe,
replicando o padrão clínico de goniometria manual (terapia ocupacional/fisioterapia).

FUNDAMENTOS CLÍNICOS (incorporados da literatura — ver PDF de referência):
  - Erro < 5° é aceitável para uso clínico de rotina (reabilitação)
  - Erro < 2° é necessário para contextos de pesquisa mais exigentes
  - TAM ASSH: ≥260° = excelente | 195–259° = bom | 130–194° = razoável | <129° = ruim
  - Produto escalar + vetorial é a abordagem correta para ângulos com sinal clínico
  - O plano anatômico da mão é essencial para diferenciar flexão de hiperextensão

DEPENDÊNCIAS: numpy apenas (sem OpenCV) — testável de forma independente.

Landmarks MediaPipe (0–20):
  0=WRIST | 1-4=THUMB | 5-8=INDEX | 9-12=MIDDLE | 13-16=RING | 17-20=PINKY
  Cada dedo: MCP=N, PIP=N+1, DIP=N+2, TIP=N+3
"""

import numpy as np
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# ÍNDICES DOS LANDMARKS MEDIAPIPE
# ═══════════════════════════════════════════════════════════════════════════════

WRIST = 0

# Polegar — anatomia diferente dos outros dedos (CMC + MCP + IP + TIP)
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4

# Dedos (MCP, PIP, DIP, TIP)
INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP  = 5,  6,  7,  8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9,  10, 11, 12
RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP   = 13, 14, 15, 16
PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP  = 17, 18, 19, 20


# ═══════════════════════════════════════════════════════════════════════════════
# RANGES NORMAIS CLÍNICOS (graus)
# Fonte: Physio-Pedia, ASSH, protocolos de terapia ocupacional
# Ref PDF: "Padrões clínicos de goniometria manual"
# ═══════════════════════════════════════════════════════════════════════════════

NORMAL_RANGES: Dict[str, tuple] = {
    "MCP_flex":  (85.0, 90.0),    # Flexão normal MCP dedos
    "MCP_hyper": (0.0,  45.0),    # Hiperextensão tolerada (valor positivo)
    "PIP_flex":  (100.0, 120.0),  # Flexão normal PIP — "urso" completo
    "DIP_flex":  (60.0,  80.0),   # Flexão normal DIP
    "ABD":       (15.0,  20.0),   # Abdução normal por dedo em relação ao médio
    "TAM":       (250.0, 270.0),  # Total Active Motion — dedo completamente fletido
    "THUMB_MCP": (50.0,  60.0),   # Flexão MCP polegar
    "THUMB_IP":  (70.0,  90.0),   # Flexão IP polegar
}

# Classificação funcional TAM — padrão ASSH (American Society for Surgery of the Hand)
# Fonte PDF: "Medição de TAM (Total Active Motion)" — seção de literatura
TAM_CLASSIFICATION = [
    (260.0, float('inf'), "Excelente", (50,  220, 130)),  # ≥ 260° → verde
    (195.0, 260.0,        "Bom",       (40,  200, 255)),  # 195–259° → amarelo
    (130.0, 195.0,        "Razoável",  (50,  130, 255)),  # 130–194° → laranja
    (0.0,   130.0,        "Ruim",      (60,   60, 255)),  # < 130° → vermelho
]


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES VETORIAIS AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════════

def _lm_to_array(landmark: Any) -> np.ndarray:
    """
    Converte um NormalizedLandmark do MediaPipe em array numpy [x, y, z].
    As coordenadas permanecem normalizadas [0,1] — NÃO converter para pixels
    nos cálculos angulares (garante invariância de escala da câmera).
    """
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float64)


def _normalize(v: np.ndarray) -> np.ndarray:
    """
    Normaliza um vetor para comprimento unitário.
    Retorna vetor zero se norma ≈ 0 (evita divisão por zero em posições degeneradas).
    """
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def angle_between_vectors_3d(v1: np.ndarray, v2: np.ndarray,
                              normal: np.ndarray) -> float:
    """
    Calcula o ângulo com SINAL entre dois vetores 3D.

    Replicação digital do goniômetro manual:
      v1 = braço estacionário (alinhado ao segmento proximal)
      v2 = braço móvel       (alinhado ao segmento distal)

    Algoritmo:
      1. Produto escalar → magnitude do ângulo (módulo, sempre 0–180°)
      2. Produto vetorial → plano de rotação
      3. Sinal via dot(v1×v2, normal):
           > 0 → v1×v2 paralelo ao normal → flexão (+)
           < 0 → v1×v2 anti-paralelo      → extensão / hiperextensão (-)

    Parâmetros
    ----------
    v1     : vetor do braço estacionário
    v2     : vetor do braço móvel
    normal : vetor normal do plano de referência da mão
             (determina qual direção é "para a palma" = flexão positiva)

    Retorno
    -------
    float em graus:
      > 0 → flexão / abdução
      < 0 → déficit de extensão / adução / hiperextensão
    """
    v1 = _normalize(v1)
    v2 = _normalize(v2)

    # Passo 1: Magnitude via produto escalar
    cos_angle = float(np.clip(np.dot(v1, v2), -1.0, 1.0))  # clamp p/ domínio arccos
    angle_deg = float(np.degrees(np.arccos(cos_angle)))

    # Passo 2 e 3: Sinal via produto vetorial
    cross = np.cross(v1, v2)  # vetor perpendicular ao plano de rotação
    sign  = float(np.dot(cross, normal))  # projeção sobre o normal da mão

    return angle_deg if sign >= 0 else -angle_deg


def _hand_normal(landmarks: List[Any]) -> np.ndarray:
    """
    Calcula o vetor normal do plano da mão.

    Fórmula:
      normal = (WRIST → MCP_INDEX) × (WRIST → MCP_PINKY)

    Este vetor aponta para fora do dorso da mão e serve como referência
    global de orientação para todos os cálculos de sinal:
      - Flexão → segmento distal se move para o lado da palma (oposto ao normal)
      - Extensão → segmento distal se move para o lado do dorso (mesmo lado do normal)

    Nota: O sinal pode variar conforme a orientação da câmera (dorso vs. palma).
    A consistência é garantida pelo fato de que TODOS os ângulos usam o mesmo normal.
    """
    wrist     = _lm_to_array(landmarks[WRIST])
    mcp_index = _lm_to_array(landmarks[INDEX_MCP])
    mcp_pinky = _lm_to_array(landmarks[PINKY_MCP])

    v1 = mcp_index - wrist   # WRIST → MCP_INDEX
    v2 = mcp_pinky - wrist   # WRIST → MCP_PINKY
    return _normalize(np.cross(v1, v2))

# NOTA: AngleSmoother foi movido para smoothing.py (EMA + Kalman obrigatórios)
# Importe de smoothing.py: from smoothing import GoniometryFilterBank

# ═══════════════════════════════════════════════════════════════════════════════
# GONIÔMETRO DIGITAL
# ═══════════════════════════════════════════════════════════════════════════════

class DigitalGoniometer:
    """
    Goniômetro digital que replica o protocolo clínico de goniometria manual.

    Baseado em:
      - Padrão ouro: alinhamento de braços do goniômetro com metacarpos e falanges
      - Referência de eixo: articulação (MCP, PIP, DIP)
      - Cálculo: produto escalar (magnitude) + produto vetorial (sinal)

    Todos os cálculos usam coordenadas normalizadas [0,1] — invariância de escala.
    Sem Euler / quaternions — apenas álgebra vetorial fundamental.

    Ref PDF: "Confirma que a abordagem landmarks + trigonometria escala para real time
    e é compatível com 30+ FPS" — pipeline determinístico e leve.
    """

    # ── 1. MCP Flexão / Hiperextensão ────────────────────────────────────────

    def mcp_flex(self, landmarks: List[Any], mcp_idx: int,
                 pip_idx: int, normal: np.ndarray) -> float:
        """
        Mede a flexão/hiperextensão da articulação MCP.

        Goniômetro clínico (padrão ouro):
          Braço estacionário = WRIST → MCP  (representa o metacarpo)
          Braço móvel        = MCP  → PIP   (representa a falange proximal)
          Eixo de rotação    = articulação MCP

        Interpretação clínica do retorno:
          +90°  → flexão plena (MCP dobrado, palma fechada)
           0°   → posição neutra (dedo estendido alinhado ao metacarpo)
          -30°  → hiperextensão (dedo "dobrado para trás" — normal até ~45°)
        """
        wrist = _lm_to_array(landmarks[WRIST])
        mcp   = _lm_to_array(landmarks[mcp_idx])
        pip   = _lm_to_array(landmarks[pip_idx])

        # Braço estacionário: vetor do metacarpo (WRIST aponta para MCP)
        v_stationary = mcp - wrist
        # Braço móvel: vetor da falange proximal (MCP aponta para PIP)
        v_mobile     = pip - mcp

        return angle_between_vectors_3d(v_stationary, v_mobile, normal)

    # ── 2. PIP Flexão / Extensão ──────────────────────────────────────────────

    def pip_flex(self, landmarks: List[Any], mcp_idx: int,
                 pip_idx: int, dip_idx: int, normal: np.ndarray) -> float:
        """
        Mede a flexão/extensão da articulação PIP.

        Goniômetro clínico:
          Braço estacionário = MCP → PIP  (falange proximal)
          Braço móvel        = PIP → DIP  (falange média)
          Eixo de rotação    = articulação PIP

        Interpretação:
          100°–120° → flexão normal ("garra de urso" completa)
          < 0°      → déficit de extensão (dedo não fecha completamente)
        """
        mcp = _lm_to_array(landmarks[mcp_idx])
        pip = _lm_to_array(landmarks[pip_idx])
        dip = _lm_to_array(landmarks[dip_idx])

        v_stationary = pip - mcp   # falange proximal = braço estacionário
        v_mobile     = dip - pip   # falange média = braço móvel

        return angle_between_vectors_3d(v_stationary, v_mobile, normal)

    # ── 3. DIP Flexão / Extensão ──────────────────────────────────────────────

    def dip_flex(self, landmarks: List[Any], pip_idx: int,
                 dip_idx: int, tip_idx: int, normal: np.ndarray) -> float:
        """
        Mede a flexão/extensão da articulação DIP.

        Goniômetro clínico:
          Braço estacionário = PIP → DIP  (falange média)
          Braço móvel        = DIP → TIP  (falange distal)
          Eixo de rotação    = articulação DIP

        Interpretação:
          60°–80° → flexão normal
          < 0°    → déficit de extensão (limitação funcional)
        """
        pip = _lm_to_array(landmarks[pip_idx])
        dip = _lm_to_array(landmarks[dip_idx])
        tip = _lm_to_array(landmarks[tip_idx])

        v_stationary = dip - pip   # falange média = braço estacionário
        v_mobile     = tip - dip   # falange distal = braço móvel

        return angle_between_vectors_3d(v_stationary, v_mobile, normal)

    # ── 4. MCP Abdução / Adução ───────────────────────────────────────────────

    def mcp_abduction(self, landmarks: List[Any], mcp_idx: int) -> float:
        """
        Mede a abdução/adução da articulação MCP no plano dorsal da mão.

        Referência clínica:
          Braço estacionário = WRIST → MCP_MIDDLE (eixo central da mão)
          Braço móvel        = WRIST → MCP do dedo avaliado
          Plano de medição   = plano dorsal da mão (não plano XZ fixo)

        Implementação robusta:
          Projeta ambos os vetores no plano da mão (remove componente do normal),
          funcionando corretamente tanto para landmarks 2D (Z≈0) quanto 3D completos.

        Interpretação:
          > 0 → abdução (afastando do dedo médio)
          < 0 → adução  (aproximando do dedo médio)
          Normal: ~15°–20° entre dedos adjacentes
        """
        wrist      = _lm_to_array(landmarks[WRIST])
        mcp_middle = _lm_to_array(landmarks[MIDDLE_MCP])
        mcp_finger = _lm_to_array(landmarks[mcp_idx])
        normal     = _hand_normal(landmarks)

        ref_vec    = mcp_middle - wrist   # referência: linha central (dedo médio)
        finger_vec = mcp_finger - wrist   # vetor do dedo avaliado

        # Projeção no plano da mão (remove a componente perpendicular ao plano dorsal)
        # Isso isola o movimento de abertura/fechamento lateral dos dedos
        ref_proj    = ref_vec    - np.dot(ref_vec,    normal) * normal
        finger_proj = finger_vec - np.dot(finger_vec, normal) * normal

        # Normal perpendicular à linha de referência no plano da mão (define sinal de ABD)
        side_normal = np.cross(_normalize(ref_proj), normal)

        return angle_between_vectors_3d(ref_proj, finger_proj, side_normal)

    # ── 5. TAM — Total Active Motion ──────────────────────────────────────────

    def total_active_motion(self, mcp: float, pip: float, dip: float) -> float:
        """
        Calcula o TAM (Total Active Motion) pelo protocolo clínico ASSH.

        Fórmula clínica:
          TAM = (MCP_flex + PIP_flex + DIP_flex) – (soma dos déficits de extensão)

        Onde:
          - Valores positivos → contribuem para a flexão total (somam)
          - Valores negativos → são déficits de extensão (subtraem do total)

        Classificação ASSH (American Society for Surgery of the Hand):
          ≥ 260°        → Excelente (função plena)
          195° – 259°   → Bom
          130° – 194°   → Razoável
          < 130°        → Ruim (limitação funcional significativa)

        Ref PDF: "TAM é consolidado como métrica funcional composta em terapia da mão"
        """
        flexion_sum      = max(0.0, mcp) + max(0.0, pip) + max(0.0, dip)
        extension_deficit = abs(min(0.0, mcp)) + abs(min(0.0, pip)) + abs(min(0.0, dip))
        return flexion_sum - extension_deficit

    @staticmethod
    def classify_tam(tam_value: float) -> Dict[str, Any]:
        """
        Classifica o TAM segundo as faixas clínicas da ASSH.

        Parâmetro: tam_value em graus

        Retorno: dict com 'label', 'color_bgr', 'pct_of_normal'
          pct_of_normal = tam_value / 270° * 100  (relativo ao TAM excelente)
        """
        # Garante que o TAM não seja negativo para a classificação
        tam_abs = max(0.0, tam_value)
        pct = min(tam_abs / 270.0, 1.0) * 100.0

        for lo, hi, label, color in TAM_CLASSIFICATION:
            if lo <= tam_abs < hi:
                return {"label": label, "color_bgr": color, "pct": round(pct, 1)}

        # Fallback (não deveria acontecer)
        return {"label": "Ruim", "color_bgr": (60, 60, 255), "pct": round(pct, 1)}

    # ── Polegar: MCP e IP ─────────────────────────────────────────────────────

    def thumb_mcp_flex(self, landmarks: List[Any], normal: np.ndarray) -> float:
        """
        Flexão do MCP do polegar.

        Braço estacionário = THUMB_CMC → THUMB_MCP (metacarpo do polegar)
        Braço móvel        = THUMB_MCP → THUMB_IP  (falange proximal)
        Normal: 50°–60°
        """
        cmc = _lm_to_array(landmarks[THUMB_CMC])
        mcp = _lm_to_array(landmarks[THUMB_MCP])
        ip  = _lm_to_array(landmarks[THUMB_IP])
        return angle_between_vectors_3d(mcp - cmc, ip - mcp, normal)

    def thumb_ip_flex(self, landmarks: List[Any], normal: np.ndarray) -> float:
        """
        Flexão da articulação IP do polegar.

        Braço estacionário = THUMB_MCP → THUMB_IP   (falange proximal)
        Braço móvel        = THUMB_IP  → THUMB_TIP  (falange distal)
        Normal: 70°–90°
        """
        mcp = _lm_to_array(landmarks[THUMB_MCP])
        ip  = _lm_to_array(landmarks[THUMB_IP])
        tip = _lm_to_array(landmarks[THUMB_TIP])
        return angle_between_vectors_3d(ip - mcp, tip - ip, normal)

    # ── Método principal ──────────────────────────────────────────────────────

    def compute_all(self, landmarks: List[Any]) -> Dict[str, Dict[str, float]]:
        """
        Calcula todos os ângulos goniométricos para todos os dedos.

        Pipeline de cálculo por dedo:
          1. Normal da mão (referência de sinal)
          2. MCP flex (metacarpo → falange proximal)
          3. PIP flex (falange proximal → falange média)
          4. DIP flex (falange média → falange distal)
          5. ABD (abertura lateral em relação ao dedo médio)
          6. TAM (soma clínica de flexões − déficits)

        Parâmetro
        ---------
        landmarks : lista de 21 NormalizedLandmark do MediaPipe Hand Landmarker

        Retorno
        -------
        dict estruturado:
        {
          "INDEX":  {"MCP": float, "PIP": float, "DIP": float, "ABD": float, "TAM": float},
          "MIDDLE": {...},
          "RING":   {...},
          "PINKY":  {...},
          "THUMB":  {"MCP": float, "IP": float},
        }

        Convenção de sinal (consistente com protocolo clínico):
          > 0 → flexão / abdução
          < 0 → déficit de extensão / adução / hiperextensão
        """
        normal = _hand_normal(landmarks)  # normal do plano dorsal da mão

        result: Dict[str, Dict[str, float]] = {}

        # Mapeamento: nome do dedo → índices (MCP, PIP, DIP, TIP)
        fingers = {
            "INDEX":  (INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP),
            "MIDDLE": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
            "RING":   (RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP),
            "PINKY":  (PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP),
        }

        for finger_name, (mcp_i, pip_i, dip_i, tip_i) in fingers.items():
            mcp_angle = self.mcp_flex(landmarks, mcp_i, pip_i, normal)
            pip_angle = self.pip_flex(landmarks, mcp_i, pip_i, dip_i, normal)
            dip_angle = self.dip_flex(landmarks, pip_i, dip_i, tip_i, normal)
            abd_angle = self.mcp_abduction(landmarks, mcp_i)
            tam       = self.total_active_motion(mcp_angle, pip_angle, dip_angle)

            result[finger_name] = {
                "MCP": round(mcp_angle, 2),
                "PIP": round(pip_angle, 2),
                "DIP": round(dip_angle, 2),
                "ABD": round(abd_angle, 2),
                "TAM": round(tam, 2),
            }

        # Polegar (anatomia diferenciada: CMC + MCP + IP)
        result["THUMB"] = {
            "MCP": round(self.thumb_mcp_flex(landmarks, normal), 2),
            "IP":  round(self.thumb_ip_flex(landmarks, normal), 2),
        }

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES UTILITÁRIAS EXPORTADAS
# ═══════════════════════════════════════════════════════════════════════════════

def is_in_normal_range(finger: str, metric: str, value: float) -> str:
    """
    Verifica se um valor angular está dentro do range normal clínico.

    Retorno: "normal" | "borderline" | "abnormal"
      - "normal"     → dentro do range padrão clínico
      - "borderline" → dentro de ±15% do range (zona cinza)
      - "abnormal"   → fora do range, inclusive déficits de extensão (< 0°)

    Usado pelo overlay para colorir cada métrica individualmente.
    """
    key_map = {
        ("INDEX",  "MCP"): "MCP_flex", ("MIDDLE", "MCP"): "MCP_flex",
        ("RING",   "MCP"): "MCP_flex", ("PINKY",  "MCP"): "MCP_flex",
        ("INDEX",  "PIP"): "PIP_flex", ("MIDDLE", "PIP"): "PIP_flex",
        ("RING",   "PIP"): "PIP_flex", ("PINKY",  "PIP"): "PIP_flex",
        ("INDEX",  "DIP"): "DIP_flex", ("MIDDLE", "DIP"): "DIP_flex",
        ("RING",   "DIP"): "DIP_flex", ("PINKY",  "DIP"): "DIP_flex",
        ("INDEX",  "ABD"): "ABD",      ("MIDDLE", "ABD"): "ABD",
        ("RING",   "ABD"): "ABD",      ("PINKY",  "ABD"): "ABD",
        ("INDEX",  "TAM"): "TAM",      ("MIDDLE", "TAM"): "TAM",
        ("RING",   "TAM"): "TAM",      ("PINKY",  "TAM"): "TAM",
        ("THUMB",  "MCP"): "THUMB_MCP",
        ("THUMB",  "IP"):  "THUMB_IP",
    }

    range_key = key_map.get((finger, metric))
    if range_key is None:
        return "normal"

    lo, hi = NORMAL_RANGES[range_key]
    margin = (hi - lo) * 0.15  # 15% de margem para zona limítrofe

    if value < 0:
        return "abnormal"   # qualquer déficit de extensão é anormal
    elif lo <= value <= hi:
        return "normal"
    elif lo - margin <= value <= hi + margin:
        return "borderline"
    else:
        return "abnormal"
