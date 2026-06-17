"""
goniometry_overlay.py — Overlay clínico da mão e da goniometria
===============================================================

Este módulo gera o painel visual mostrado sobre a webcam.

Responsabilidades:
- desenhar esqueleto da mão;
- desenhar braços do goniômetro virtual;
- desenhar arcos angulares e labels;
- colorir estruturas conforme faixa clínica;
- mostrar estabilidade qualitativa do filtro por articulação.

A função principal exportada é draw_goniometry_overlay().
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from goniometry import is_in_normal_range, DigitalGoniometer

# =============================================================================
# PALETA VISUAL
# =============================================================================

BG_DARK = (26, 26, 26)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY_LIGHT = (180, 180, 180)
GRAY_MID = (110, 110, 110)
GRAY_DARK = (55, 55, 55)

COLOR_WRIST = (255, 255, 255)
COLOR_MCP = (0, 220, 220)
COLOR_PIP = (220, 220, 0)
COLOR_DIP = (80, 220, 50)
COLOR_TIP = (200, 200, 200)
COLOR_THUMB = (220, 80, 220)

COLOR_STAT = (60, 60, 230)
COLOR_MOB = (230, 120, 50)
COLOR_AXIS = (255, 255, 0)

COLOR_NORMAL = (50, 220, 130)
COLOR_BORDER = (40, 200, 255)
COLOR_ABNORM = (60, 60, 255)
COLOR_SEC = (150, 150, 150)

STAB_STABLE = (50, 220, 130)
STAB_CONV = (40, 200, 255)
STAB_UNSTAB = (60, 60, 255)
STAB_UNINIT = (80, 80, 80)

ARM_LEN = 55
ARC_RAD = 28
ARC_TICK = 3
ARM_TICK = 3

# =============================================================================
# TOPOLOGIA DA MÃO
# =============================================================================

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

WRIST_CONNS = {(0, 5), (0, 9), (0, 13), (0, 17)}

LM_STYLE = {
    **{0: ("wrist", COLOR_WRIST, 8)},
    **{i: ("thumb", COLOR_THUMB, 5) for i in (1, 2, 3, 4)},
    **{i: ("mcp", COLOR_MCP, 7) for i in (5, 9, 13, 17)},
    **{i: ("pip", COLOR_PIP, 6) for i in (6, 10, 14, 18)},
    **{i: ("dip", COLOR_DIP, 5) for i in (7, 11, 15, 19)},
    **{i: ("tip", COLOR_TIP, 4) for i in (8, 12, 16, 20)},
}

GONIO_JOINTS = [
    (5, 0, 6, "MCP", "INDEX"),
    (6, 5, 7, "PIP", "INDEX"),
    (7, 6, 8, "DIP", "INDEX"),
    (9, 0, 10, "MCP", "MIDDLE"),
    (10, 9, 11, "PIP", "MIDDLE"),
    (11, 10, 12, "DIP", "MIDDLE"),
    (13, 0, 14, "MCP", "RING"),
    (14, 13, 15, "PIP", "RING"),
    (15, 14, 16, "DIP", "RING"),
    (17, 0, 18, "MCP", "PINKY"),
    (18, 17, 19, "PIP", "PINKY"),
    (19, 18, 20, "DIP", "PINKY"),
    (2, 1, 3, "MCP", "THUMB"),
    (3, 2, 4, "IP", "THUMB"),
]

# =============================================================================
# FUNÇÕES AUXILIARES DE DESENHO
# =============================================================================

def _stability_color(status: str) -> Tuple[int, int, int]:
    return {
        "stable": STAB_STABLE,
        "converging": STAB_CONV,
        "unstable": STAB_UNSTAB,
        "uninitialized": STAB_UNINIT,
    }.get(status, STAB_UNINIT)


def _clinical_color(status: str) -> Tuple[int, int, int]:
    return {
        "normal": COLOR_NORMAL,
        "borderline": COLOR_BORDER,
        "abnormal": COLOR_ABNORM,
    }.get(status, GRAY_LIGHT)


def _lm_px(lm, idx: int, width: int, height: int) -> Tuple[int, int]:
    return int(lm[idx].x * width), int(lm[idx].y * height)


def _n2(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v


def _dashed(
    img: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash: int = 8,
    gap: int = 5,
) -> None:
    """
    Desenha linha tracejada simples para conexões do punho.
    """
    x1, y1 = p1
    x2, y2 = p2

    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return

    ux = (x2 - x1) / length
    uy = (y2 - y1) / length

    pos = 0.0
    draw = True
    while pos < length:
        seg = dash if draw else gap
        next_pos = min(pos + seg, length)

        if draw:
            a = (int(x1 + ux * pos), int(y1 + uy * pos))
            b = (int(x1 + ux * next_pos), int(y1 + uy * next_pos))
            cv2.line(img, a, b, color, thickness, cv2.LINE_AA)

        draw = not draw
        pos = next_pos


def _arc(
    img: np.ndarray,
    ctr: Tuple[int, int],
    v1: np.ndarray,
    v2: np.ndarray,
    radius: int,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """
    Desenha arco angular entre dois vetores 2D.
    """
    a1 = math.degrees(math.atan2(-v1[1], v1[0]))
    a2 = math.degrees(math.atan2(-v2[1], v2[0]))

    diff = (a2 - a1) % 360
    if diff > 180:
        a1, a2 = a2, a1
        diff = 360 - diff

    steps = max(int(diff / 3), 4)
    pts = [
        (
            int(ctr[0] + radius * math.cos(math.radians(a1 + diff * i / steps))),
            int(ctr[1] - radius * math.sin(math.radians(a1 + diff * i / steps))),
        )
        for i in range(steps + 1)
    ]

    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)


def _alpha_rect(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: Tuple[int, int, int],
    alpha: float = 0.60,
) -> None:
    """
    Desenha um retângulo semitransparente operando apenas no ROI.

    Versão otimizada: em vez de copiar a imagem inteira (img.copy()),
    opera apenas na região do retângulo, reduzindo alocação de memória
    de ~1MB para ~500 bytes por chamada.
    """
    img_h, img_w = img.shape[:2]
    # Clipping para evitar acesso fora dos limites da imagem.
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    blend = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(blend, alpha, roi, 1 - alpha, 0, roi)


def _tam_bar(img: np.ndarray, x: int, y: int, w: int, h: int, pct: float) -> None:
    """
    Desenha barra de TAM normalizada.

    A implementação usa pintura por coluna com numpy/OpenCV,
    mantendo custo baixo para cada frame.
    """
    filled = max(0, min(int(w * pct), w))
    cv2.rectangle(img, (x, y), (x + w, y + h), (45, 45, 45), -1)

    if filled > 0:
        grad = np.zeros((h, filled, 3), dtype=np.uint8)
        t = np.linspace(0, 1, filled, dtype=np.float32)

        left = t < 0.5
        right = ~left

        grad[:, left, 0] = 255
        grad[:, left, 1] = (60 + 140 * (t[left] * 2)).astype(np.uint8)
        grad[:, left, 2] = (40 * (t[left] * 2)).astype(np.uint8)

        grad[:, right, 0] = (255 - 125 * ((t[right] - 0.5) * 2)).astype(np.uint8)
        grad[:, right, 1] = (200 + 20 * ((t[right] - 0.5) * 2)).astype(np.uint8)
        grad[:, right, 2] = (40 + 10 * ((t[right] - 0.5) * 2)).astype(np.uint8)

        img[y:y + h, x:x + filled] = grad

    cv2.rectangle(img, (x, y), (x + w, y + h), GRAY_MID, 1)


def _center_text(
    img: np.ndarray,
    text: str,
    y: int,
    width: int,
    color: Tuple[int, int, int] = WHITE,
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
    cv2.putText(
        img,
        text,
        ((width - tw) // 2, y),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _put(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int] = WHITE,
    scale: float = 0.40,
    thickness: int = 1,
) -> None:
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _tw(text: str, scale: float = 0.40, thickness: int = 1) -> int:
    (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
    return w


# =============================================================================
# PAINEL DE ESQUELETO + GONIÔMETRO
# =============================================================================

def _build_skeleton(
    frame: np.ndarray,
    landmarks: List[Any],
    angles: Dict[str, Dict[str, float]],
    pw: int,
    ph: int,
    frozen: bool,
    stability_map: Optional[Dict],
) -> np.ndarray:
    """
    Constrói o painel com o esqueleto da mão e o goniômetro virtual.
    """
    canvas = np.full((ph, pw, 3), BG_DARK, dtype=np.uint8)
    _center_text(canvas, "GONIOMETRIA DIGITAL", 26, pw, WHITE, 0.60)

    if not landmarks:
        _center_text(canvas, "Aguardando detecção da mão...", ph // 2, pw, GRAY_MID, 0.45)
        return canvas

    width, height = pw, ph

    # Desenho do esqueleto básico.
    for a, b in CONNECTIONS:
        p1 = _lm_px(landmarks, a, width, height)
        p2 = _lm_px(landmarks, b, width, height)

        if (a, b) in WRIST_CONNS or (b, a) in WRIST_CONNS:
            _dashed(canvas, p1, p2, (140, 140, 140), 1)
        else:
            cv2.line(canvas, p1, p2, (185, 185, 185), 2, cv2.LINE_AA)

    # Landmarks por categoria anatômica.
    for idx, (_, color, radius) in LM_STYLE.items():
        pt = _lm_px(landmarks, idx, width, height)
        cv2.circle(canvas, pt, radius + 1, BLACK, -1)
        cv2.circle(canvas, pt, radius, color, -1, cv2.LINE_AA)

    # Goniômetro virtual por articulação.
    for ax_i, prox_i, dist_i, joint_type, finger in GONIO_JOINTS:
        center = _lm_px(landmarks, ax_i, width, height)
        pt_prox = _lm_px(landmarks, prox_i, width, height)
        pt_dist = _lm_px(landmarks, dist_i, width, height)

        vs = np.array([center[0] - pt_prox[0], center[1] - pt_prox[1]], dtype=float)
        vm = np.array([pt_dist[0] - center[0], pt_dist[1] - center[1]], dtype=float)

        vsn = _n2(vs)
        vmn = _n2(vm)

        es = (int(center[0] + vsn[0] * ARM_LEN), int(center[1] + vsn[1] * ARM_LEN))
        em = (int(center[0] + vmn[0] * ARM_LEN), int(center[1] + vmn[1] * ARM_LEN))

        angle_val = angles.get(finger, {}).get(joint_type)
        if angle_val is None:
            continue

        clinical_status = is_in_normal_range(finger, joint_type, angle_val)
        arc_color = _clinical_color(clinical_status)

        stability_status = (stability_map or {}).get(finger, {}).get(joint_type, "uninitialized")
        stability_color = _stability_color(stability_status)

        cv2.line(canvas, center, es, COLOR_STAT, ARM_TICK, cv2.LINE_AA)
        cv2.circle(canvas, es, 4, COLOR_STAT, -1, cv2.LINE_AA)

        cv2.line(canvas, center, em, COLOR_MOB, ARM_TICK, cv2.LINE_AA)
        cv2.circle(canvas, em, 4, COLOR_MOB, -1, cv2.LINE_AA)

        cv2.circle(canvas, center, 6, COLOR_AXIS, -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 6, WHITE, 1, cv2.LINE_AA)

        _arc(canvas, center, vsn, vmn, ARC_RAD, arc_color, ARC_TICK)

        sign = "-" if angle_val < 0 else ""
        label = f"{joint_type} {sign}{abs(angle_val):.0f}"
        lx, ly = center[0] + 10, center[1] - 8

        tw = _tw(label, 0.42)
        _alpha_rect(canvas, lx - 3, ly - 14, tw + 12, 18, (8, 8, 8), 0.72)
        _put(canvas, label, lx, ly, arc_color, 0.42)
        cv2.circle(canvas, (lx + tw + 8, ly - 6), 4, stability_color, -1, cv2.LINE_AA)

    xs = [int(landmarks[i].x * width) for i in range(21)]
    ys = [int(landmarks[i].y * height) for i in range(21)]
    pad = 22

    cv2.rectangle(
        canvas,
        (max(0, min(xs) - pad), max(0, min(ys) - pad)),
        (min(width - 1, max(xs) + pad), min(height - 1, max(ys) + pad)),
        GRAY_MID,
        1,
    )

    lx, ly = 8, ph - 30
    cv2.line(canvas, (lx, ly), (lx + 20, ly), COLOR_STAT, 2)
    _put(canvas, "Estacionario", lx + 24, ly + 4, GRAY_LIGHT, 0.28)

    cv2.line(canvas, (lx + 130, ly), (lx + 150, ly), COLOR_MOB, 2)
    _put(canvas, "Movel", lx + 154, ly + 4, GRAY_LIGHT, 0.28)

    cv2.circle(canvas, (lx + 210, ly), 4, COLOR_AXIS, -1)
    _put(canvas, "Eixo", lx + 218, ly + 4, GRAY_LIGHT, 0.28)

    if frozen:
        _center_text(canvas, "[ FRAME CONGELADO ]", ph - 14, pw, COLOR_BORDER, 0.38)
        cv2.rectangle(canvas, (2, 2), (pw - 2, ph - 2), COLOR_BORDER, 2)

    return canvas


# =============================================================================
# PAINEL DE DADOS CLÍNICOS (com cache de template estático)
# =============================================================================

# Cache global do template estático do painel de dados.
# Chave: (pw, ph) → template numpy array.
_data_template_cache: Dict[Tuple[int, int], np.ndarray] = {}

_DATA_FINGERS = ["INDEX", "MIDDLE", "RING", "PINKY", "THUMB"]
_DATA_LABELS = {
    "INDEX": "Indicador",
    "MIDDLE": "Medio",
    "RING": "Anelar",
    "PINKY": "Minimo",
    "THUMB": "Polegar",
}
_DATA_BLOCK_H = 76
_DATA_START_Y = 58
_DATA_MARGIN_X = 14


def _build_data_template(pw: int, ph: int) -> np.ndarray:
    """
    Constrói o template estático do painel de dados clínicos.

    Este template contém todos os elementos que NÃO mudam entre frames:
    - fundo escuro, título, subtítulo;
    - linhas divisórias entre dedos;
    - labels dos dedos (Indicador, Medio, etc.);
    - label "TAM" de cada bloco;
    - legenda de rodapé (ranges, classificação TAM, estabilidade).

    Chamado uma única vez por dimensão (pw, ph). Nas chamadas seguintes,
    _build_data_panel copia este template e preenche apenas os dados dinâmicos.
    """
    canvas = np.full((ph, pw, 3), BG_DARK, dtype=np.uint8)

    _center_text(canvas, "Dados para cada dedo", 26, pw, WHITE, 0.55)
    _center_text(
        canvas,
        "Amplitude de movimento | Verde=normal | Vermelho=fora",
        44,
        pw,
        GRAY_LIGHT,
        0.32,
    )
    cv2.line(canvas, (10, 52), (pw - 10, 52), GRAY_DARK, 1)

    for fi, finger in enumerate(_DATA_FINGERS):
        by = _DATA_START_Y + fi * _DATA_BLOCK_H

        # Linha divisória.
        cv2.line(canvas, (_DATA_MARGIN_X, by), (pw - _DATA_MARGIN_X, by), GRAY_DARK, 1)

        # Nome do dedo (posição fixa, sem bullet de cor — o bullet é dinâmico).
        bx, bry = _DATA_MARGIN_X + 8, by + 18
        _put(canvas, _DATA_LABELS[finger], bx + 16, bry + 4, WHITE, 0.48, 1)

        # Label "TAM" (fixo).
        bar_y = by + 52
        _put(canvas, "TAM", _DATA_MARGIN_X + 10, bar_y + 9, GRAY_LIGHT, 0.33)

    # Legenda de rodapé (completamente estática).
    fy = ph - 56
    cv2.line(canvas, (10, fy - 4), (pw - 10, fy - 4), GRAY_DARK, 1)

    lines = [
        "Ranges: MCP 85-90 | PIP 100-120 | DIP 60-80 | ABD 15-20",
        "TAM: >=260 Excelente | 195-259 Bom | 130-194 Razoavel | <130 Ruim",
    ]
    for i, line in enumerate(lines):
        y = fy + 4 + i * 14
        _center_text(canvas, line, y, pw, GRAY_LIGHT, 0.29)

    # Legenda de estabilidade.
    y = fy + 4 + 2 * 14
    _put(canvas, "●", 14, y, STAB_STABLE, 0.30)
    _put(canvas, "Estavel", 26, y, GRAY_LIGHT, 0.28)
    _put(canvas, "●", 90, y, STAB_CONV, 0.30)
    _put(canvas, "Convergindo", 102, y, GRAY_LIGHT, 0.28)
    _put(canvas, "●", 208, y, STAB_UNSTAB, 0.30)
    _put(canvas, "Instavel", 220, y, GRAY_LIGHT, 0.28)
    _put(canvas, "(filtro EMA -> Kalman)", pw // 2 + 40, y, GRAY_MID, 0.27)

    return canvas


def _build_data_panel(
    angles: Dict[str, Dict[str, float]],
    pw: int,
    ph: int,
    frozen: bool,
    stability_map: Optional[Dict],
) -> np.ndarray:
    """
    Constrói o painel textual lateral com resumo clínico por dedo.

    Versão otimizada: copia o template estático e desenha apenas os
    elementos dinâmicos (bullet de cor, valores numéricos, barras TAM,
    badge de classificação).
    """
    # Obter ou criar template para estas dimensões.
    cache_key = (pw, ph)
    if cache_key not in _data_template_cache:
        _data_template_cache[cache_key] = _build_data_template(pw, ph)

    canvas = _data_template_cache[cache_key].copy()

    for fi, finger in enumerate(_DATA_FINGERS):
        data = angles.get(finger, {})
        by = _DATA_START_Y + fi * _DATA_BLOCK_H

        # Bullet de cor (dinâmico — depende do status clínico).
        if finger == "THUMB":
            checks = [("MCP", data.get("MCP", 0.0)), ("IP", data.get("IP", 0.0))]
        else:
            checks = [
                ("MCP", data.get("MCP", 0.0)),
                ("PIP", data.get("PIP", 0.0)),
                ("DIP", data.get("DIP", 0.0)),
                ("TAM", data.get("TAM", 0.0)),
            ]

        statuses = [is_in_normal_range(finger, metric, value) for metric, value in checks]
        bullet_color = (
            COLOR_ABNORM if "abnormal" in statuses
            else COLOR_BORDER if "borderline" in statuses
            else COLOR_NORMAL
        )

        bx, bry = _DATA_MARGIN_X + 8, by + 18
        cv2.circle(canvas, (bx, bry), 7, bullet_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (bx, bry), 7, WHITE, 1, cv2.LINE_AA)

        # Valores numéricos dos ângulos (dinâmicos).
        ax, ay = _DATA_MARGIN_X + 10, by + 36

        if finger == "THUMB":
            for metric in ("MCP", "IP"):
                value = data.get(metric, 0.0)
                color = _clinical_color(is_in_normal_range(finger, metric, value))
                text = f"{metric}: {value:.1f}"
                _put(canvas, text, ax, ay, color, 0.37)
                ax += _tw(text, 0.37) + 18
        else:
            for metric in ("MCP", "PIP", "DIP"):
                value = data.get(metric, 0.0)
                color = _clinical_color(is_in_normal_range(finger, metric, value))
                text = f"{metric}:{value:.1f}"
                _put(canvas, text, ax, ay, color, 0.36)
                ax += _tw(text, 0.36) + 10

            abd = data.get("ABD", 0.0)
            _put(canvas, f"ABD:{abd:.1f}", ax, ay, COLOR_SEC, 0.34)

        # Barra TAM (dinâmica).
        tam = data.get("TAM", 0.0)
        tam_info = DigitalGoniometer.classify_tam(tam)
        tam_pct = max(0.0, min(tam / 270.0, 1.0))

        bar_y = by + 52
        bar_x = _DATA_MARGIN_X + 42
        bar_w = 115
        bar_h = 11

        _tam_bar(canvas, bar_x, bar_y, bar_w, bar_h, tam_pct)

        tam_txt = f"{tam:.0f}"
        assh_label = tam_info["label"]
        assh_color = tam_info["color_bgr"]

        vx = bar_x + bar_w + 5
        _put(
            canvas,
            tam_txt,
            vx,
            bar_y + 9,
            _clinical_color(is_in_normal_range(finger, "TAM", tam)),
            0.35,
        )

        bx2 = vx + _tw(tam_txt, 0.35) + 5
        bw2 = _tw(assh_label, 0.30) + 8

        _alpha_rect(canvas, bx2, bar_y, bw2, 13, assh_color, 0.55)
        cv2.rectangle(canvas, (bx2, bar_y), (bx2 + bw2, bar_y + 13), assh_color, 1)
        _put(canvas, assh_label, bx2 + 4, bar_y + 10, WHITE, 0.30)

    if frozen:
        cv2.rectangle(canvas, (2, 2), (pw - 2, ph - 2), COLOR_BORDER, 2)

    return canvas



# =============================================================================
# FUNÇÃO PÚBLICA PRINCIPAL
# =============================================================================

def draw_goniometry_overlay(
    frame: np.ndarray,
    landmarks: List[Any],
    angles: Dict[str, Dict[str, float]],
    panel_w: Optional[int] = None,
    panel_h: Optional[int] = None,
    frozen: bool = False,
    stability_map: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera os painéis principais do sistema visual de goniometria.
    """
    h, w = frame.shape[:2]
    pw = panel_w or w
    ph = panel_h or h

    skeleton = _build_skeleton(frame, landmarks, angles, pw, ph, frozen, stability_map)
    data = _build_data_panel(angles, pw, ph, frozen, stability_map)

    return skeleton, data


def compose_side_by_side(skel: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Compõe dois painéis lado a lado em uma única imagem.
    """
    h1, h2 = skel.shape[0], data.shape[0]
    max_h = max(h1, h2)

    def _pad(img: np.ndarray, target_h: int) -> np.ndarray:
        if img.shape[0] < target_h:
            pad = np.full((target_h - img.shape[0], img.shape[1], 3), BG_DARK, np.uint8)
            return np.vstack([img, pad])
        return img

    sep = np.full((max_h, 2, 3), GRAY_MID, np.uint8)
    return np.hstack([_pad(skel, max_h), sep, _pad(data, max_h)])