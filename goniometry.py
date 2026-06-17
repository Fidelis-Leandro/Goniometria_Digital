"""
goniometry.py — Goniometria digital baseada em landmarks 3D
===========================================================

Este módulo calcula ângulos articulares da mão a partir dos landmarks do MediaPipe.

Responsabilidades:
- converter landmarks em vetores 3D;
- calcular o plano de referência da mão;
- medir ângulos com sinal clínico;
- calcular MCP, PIP, DIP, ABD e TAM dos dedos;
- calcular MCP e IP do polegar;
- classificar TAM e validar ranges clínicos.

O módulo é independente de OpenCV e Streamlit.
"""

from typing import Any, Dict, List

import numpy as np

# =============================================================================
# ÍNDICES DOS LANDMARKS
# =============================================================================

WRIST = 0

THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4

INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# =============================================================================
# REFERÊNCIA CLÍNICA
# =============================================================================

NORMAL_RANGES: Dict[str, tuple] = {
    "MCP_flex":  (70.0, 90.0),   # ASSH: flexão MCP 70–90° é normal em movimento ativo
    "MCP_hyper": (0.0, 45.0),
    "PIP_flex":  (100.0, 120.0),
    "DIP_flex":  (60.0, 80.0),
    "ABD":       (15.0, 20.0),
    "TAM":       (250.0, 270.0),
    "THUMB_MCP": (50.0, 60.0),
    "THUMB_IP":  (70.0, 90.0),
}

TAM_CLASSIFICATION = [
    (260.0, float("inf"), "Excelente", (50, 220, 130)),
    (195.0, 260.0, "Bom", (40, 200, 255)),
    (130.0, 195.0, "Razoável", (50, 130, 255)),
    (0.0, 130.0, "Ruim", (60, 60, 255)),
]


# =============================================================================
# FUNÇÕES VETORIAIS
# =============================================================================

def _lm_to_array(landmark: Any) -> np.ndarray:
    """
    Converte um landmark do MediaPipe para vetor numpy [x, y, z].
    """
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float64)


def _normalize(v: np.ndarray) -> np.ndarray:
    """
    Normaliza um vetor.
    """
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else np.zeros(3, dtype=np.float64)


def angle_between_vectors_3d(v1: np.ndarray, v2: np.ndarray, normal: np.ndarray) -> float:
    """
    Calcula o ângulo com sinal entre dois vetores 3D.

    O sinal usa o plano da mão como referência para distinguir:
    - flexão / abdução;
    - extensão / hiperextensão / adução.
    """
    v1 = _normalize(v1)
    v2 = _normalize(v2)

    cos_angle = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_angle)))

    cross = np.cross(v1, v2)
    sign = float(np.dot(cross, normal))

    return angle_deg if sign >= 0 else -angle_deg


def _hand_normal(landmarks: List[Any], is_right_hand: bool = True) -> np.ndarray:
    """
    Calcula o vetor normal do plano da mão.

    A normal aponta para fora da palma na mão direita.
    Na mão esquerda (is_right_hand=False), o vetor é invertido
    para manter a convenção de sinal correta para flexão/extensão.

    Nota: cv2.flip() espelha o frame visualmente mas não altera as
    coordenadas .x/.y/.z dos landmarks do MediaPipe — por isso a
    correção de handedness deve ser feita aqui, na normal.
    """
    wrist = _lm_to_array(landmarks[WRIST])
    mcp_index = _lm_to_array(landmarks[INDEX_MCP])
    mcp_pinky = _lm_to_array(landmarks[PINKY_MCP])

    v1 = mcp_index - wrist
    v2 = mcp_pinky - wrist

    normal = _normalize(np.cross(v1, v2))
    if not is_right_hand:
        normal = -normal
    return normal


# =============================================================================
# GONIÔMETRO DIGITAL
# =============================================================================

class DigitalGoniometer:
    """
    Implementa os cálculos articulares da mão.

    A classe encapsula as fórmulas clínicas e a convenção de sinal
    para produzir um dicionário estruturado por dedo e articulação.
    """

    def mcp_flex(self, landmarks: List[Any], mcp_idx: int, pip_idx: int, normal: np.ndarray) -> float:
        """
        Calcula flexão da MCP de um dedo não polegar.
        """
        wrist = _lm_to_array(landmarks[WRIST])
        mcp = _lm_to_array(landmarks[mcp_idx])
        pip = _lm_to_array(landmarks[pip_idx])

        return angle_between_vectors_3d(mcp - wrist, pip - mcp, normal)

    def pip_flex(
        self,
        landmarks: List[Any],
        mcp_idx: int,
        pip_idx: int,
        dip_idx: int,
        normal: np.ndarray,
    ) -> float:
        """
        Calcula flexão da PIP.
        """
        mcp = _lm_to_array(landmarks[mcp_idx])
        pip = _lm_to_array(landmarks[pip_idx])
        dip = _lm_to_array(landmarks[dip_idx])

        return angle_between_vectors_3d(pip - mcp, dip - pip, normal)

    def dip_flex(
        self,
        landmarks: List[Any],
        pip_idx: int,
        dip_idx: int,
        tip_idx: int,
        normal: np.ndarray,
    ) -> float:
        """
        Calcula flexão da DIP.
        """
        pip = _lm_to_array(landmarks[pip_idx])
        dip = _lm_to_array(landmarks[dip_idx])
        tip = _lm_to_array(landmarks[tip_idx])

        return angle_between_vectors_3d(dip - pip, tip - dip, normal)

    # Referência de abdução por dedo: usa o dedo imediatamente adjacente.
    # Usar o dedo médio como referência absoluta para todos superestimava
    # a abdução do indicador e distorcia o mínimo.
    _ABD_REFERENCE = {
        INDEX_MCP:  MIDDLE_MCP,   # indicador → médio
        MIDDLE_MCP: MIDDLE_MCP,   # médio → si mesmo (resultado 0, sem ABD definida)
        RING_MCP:   MIDDLE_MCP,   # anelar → médio
        PINKY_MCP:  RING_MCP,     # mínimo → anelar
    }

    def mcp_abduction(self, landmarks: List[Any], mcp_idx: int) -> float:
        """
        Calcula abdução da MCP usando o dedo adjacente como referência.

        Referências clínicas:
        - Indicador e Anelar: referência no Médio.
        - Mínimo: referência no Anelar.
        - Médio: retorna 0 (sem referência de abdução definida clinicamente).
        """
        ref_idx = self._ABD_REFERENCE.get(mcp_idx, MIDDLE_MCP)
        wrist = _lm_to_array(landmarks[WRIST])
        ref_mcp = _lm_to_array(landmarks[ref_idx])
        current_mcp = _lm_to_array(landmarks[mcp_idx])

        if mcp_idx == ref_idx:
            return 0.0  # dedo médio não tem referência adjacente

        ref = _normalize(ref_mcp - wrist)
        cur = _normalize(current_mcp - wrist)

        cos_angle = float(np.clip(np.dot(ref, cur), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_angle)))

    def total_active_motion(self, mcp: float, pip: float, dip: float) -> float:
        """
        Calcula TAM (Total Active Motion) pela fórmula ASSH completa.

        Fórmula ASSH:
            TAM = (MCP + PIP + DIP)_flex − (MCP + PIP + DIP)_déficit

        Déficit = ângulo negativo (extensão incompleta / contratura em flexão).
        Um paciente com PIP travado em −30° tem esse déficit subtraído do TAM,
        o que não era refletido na fórmula anterior que ignorava valores negativos.
        """
        flex_sum   = max(mcp, 0.0) + max(pip, 0.0) + max(dip, 0.0)
        deficit_sum = abs(min(mcp, 0.0)) + abs(min(pip, 0.0)) + abs(min(dip, 0.0))
        return max(0.0, flex_sum - deficit_sum)

    def thumb_mcp_flex(self, landmarks: List[Any], normal: np.ndarray) -> float:
        """
        Calcula flexão do MCP do polegar.
        """
        cmc = _lm_to_array(landmarks[THUMB_CMC])
        mcp = _lm_to_array(landmarks[THUMB_MCP])
        ip = _lm_to_array(landmarks[THUMB_IP])

        return angle_between_vectors_3d(mcp - cmc, ip - mcp, normal)

    def thumb_ip_flex(self, landmarks: List[Any], normal: np.ndarray) -> float:
        """
        Calcula flexão da IP do polegar.
        """
        mcp = _lm_to_array(landmarks[THUMB_MCP])
        ip = _lm_to_array(landmarks[THUMB_IP])
        tip = _lm_to_array(landmarks[THUMB_TIP])

        return angle_between_vectors_3d(ip - mcp, tip - ip, normal)

    def compute_all(
        self,
        landmarks: List[Any],
        is_right_hand: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcula todas as métricas articulares da mão.

        Parâmetros:
            landmarks    : lista de landmarks do MediaPipe (21 pontos).
            is_right_hand: True para mão direita, False para mão esquerda.
                           Inverte a normal do plano para corrigir o sinal
                           de flexão/extensão em mãos espelhadas.
        """
        normal = _hand_normal(landmarks, is_right_hand=is_right_hand)

        result: Dict[str, Dict[str, float]] = {}

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
            tam = self.total_active_motion(mcp_angle, pip_angle, dip_angle)

            result[finger_name] = {
                "MCP": round(mcp_angle, 2),
                "PIP": round(pip_angle, 2),
                "DIP": round(dip_angle, 2),
                "ABD": round(abd_angle, 2),
                "TAM": round(tam, 2),
            }

        result["THUMB"] = {
            "MCP": round(self.thumb_mcp_flex(landmarks, normal), 2),
            "IP":  round(self.thumb_ip_flex(landmarks, normal), 2),
        }

        return result

    @staticmethod
    def classify_tam(tam: float) -> Dict[str, object]:
        """
        Classifica um valor de TAM conforme a referência funcional.
        """
        for lo, hi, label, color_bgr in TAM_CLASSIFICATION:
            if lo <= tam < hi:
                return {
                    "label": label,
                    "color_bgr": color_bgr,
                }

        return {
            "label": "Ruim",
            "color_bgr": (60, 60, 255),
        }


# =============================================================================
# CLASSIFICAÇÃO DE RANGE NORMAL
# =============================================================================

def is_in_normal_range(finger: str, metric: str, value: float) -> str:
    """
    Informa se um valor está:
    - dentro do range normal;
    - limítrofe;
    - fora do esperado.
    """
    key_map = {
        ("INDEX", "MCP"): "MCP_flex",
        ("MIDDLE", "MCP"): "MCP_flex",
        ("RING", "MCP"): "MCP_flex",
        ("PINKY", "MCP"): "MCP_flex",
        ("INDEX", "PIP"): "PIP_flex",
        ("MIDDLE", "PIP"): "PIP_flex",
        ("RING", "PIP"): "PIP_flex",
        ("PINKY", "PIP"): "PIP_flex",
        ("INDEX", "DIP"): "DIP_flex",
        ("MIDDLE", "DIP"): "DIP_flex",
        ("RING", "DIP"): "DIP_flex",
        ("PINKY", "DIP"): "DIP_flex",
        ("INDEX", "ABD"): "ABD",
        ("MIDDLE", "ABD"): "ABD",
        ("RING", "ABD"): "ABD",
        ("PINKY", "ABD"): "ABD",
        ("INDEX", "TAM"): "TAM",
        ("MIDDLE", "TAM"): "TAM",
        ("RING", "TAM"): "TAM",
        ("PINKY", "TAM"): "TAM",
        ("THUMB", "MCP"): "THUMB_MCP",
        ("THUMB", "IP"): "THUMB_IP",
    }

    range_key = key_map.get((finger, metric))
    if range_key is None:
        return "normal"

    lo, hi = NORMAL_RANGES[range_key]
    margin = (hi - lo) * 0.15

    if value < 0:
        return "abnormal"
    elif lo <= value <= hi:
        return "normal"
    elif lo - margin <= value <= hi + margin:
        return "borderline"
    else:
        return "abnormal"