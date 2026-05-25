"""
goniometry_overlay.py — Visualização Clínica Goniométrica (v3)
=============================================================
Redesenhado para comunicar INEQUIVOCAMENTE que o sistema está medindo
ângulos com lógica de goniômetro clínico, não apenas rastreando landmarks.

MELHORIAS EM RELAÇÃO À VERSÃO ANTERIOR:
  ✓ Braços do goniômetro mais longos (36→55px): maior impacto visual
  ✓ Arco mais visível (raio 18→28px, espessura 2→3px)
  ✓ Círculo marcador no vértice articular (eixo do goniômetro)
  ✓ Bolinhas nas extremidades dos braços (pontos de referência anatômica)
  ✓ Indicador de estabilidade Kalman por articulação (●verde/●amarelo/●vermelho)
  ✓ Labels de ângulo maiores e mais legíveis (0.38→0.45 scale)
  ✓ Barra de qualidade Kalman no rodapé da janela
  ✓ Suporte ao parâmetro stability_map (saída do GoniometryFilterBank)

FILOSOFIA VISUAL:
  O overlay deve ser legível para um fisioterapeuta ou médico que nunca
  viu o sistema. A primeira impressão deve ser: "isso é um goniômetro".

Dependências: numpy, opencv-python
"""

import cv2
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from goniometry import NORMAL_RANGES, TAM_CLASSIFICATION, is_in_normal_range, DigitalGoniometer


# ═══════════════════════════════════════════════════════════════════════════════
# PALETA DE CORES (BGR)
# ═══════════════════════════════════════════════════════════════════════════════

BG_DARK      = (26,  26,  26)
WHITE        = (255, 255, 255)
BLACK        = (0,   0,   0)
GRAY_LIGHT   = (180, 180, 180)
GRAY_MID     = (110, 110, 110)
GRAY_DARK    = (55,  55,  55)

COLOR_WRIST  = (255, 255, 255)
COLOR_MCP    = (0,   220, 220)
COLOR_PIP    = (220, 220, 0)
COLOR_DIP    = (80,  220, 50)
COLOR_TIP    = (200, 200, 200)
COLOR_THUMB  = (220, 80,  220)

# Braços do goniômetro — vermelho=estacionário, azul=móvel (padrão clínico)
COLOR_STAT   = (60,  60,  230)   # braço estacionário (segmento proximal)
COLOR_MOB    = (230, 120, 50)    # braço móvel (segmento distal)
COLOR_AXIS   = (255, 255, 0)     # vértice articular (eixo do goniômetro)

COLOR_NORMAL = (50,  220, 130)   # verde — dentro do range
COLOR_BORDER = (40,  200, 255)   # amarelo — limítrofe
COLOR_ABNORM = (60,  60,  255)   # vermelho — fora do range
COLOR_SEC    = (150, 150, 150)   # cinza — métricas secundárias

# Cores do indicador de estabilidade Kalman
STAB_STABLE  = (50,  220, 130)   # verde — filtro convergido
STAB_CONV    = (40,  200, 255)   # amarelo — em convergência
STAB_UNSTAB  = (60,  60,  255)   # vermelho — alta incerteza
STAB_UNINIT  = (80,  80,  80)    # cinza — não inicializado

# Comprimentos e espessuras do goniômetro virtual
ARM_LEN  = 55    # comprimento dos braços em pixels (↑ de 36px para maior impacto)
ARC_RAD  = 28    # raio do arco angular (↑ de 18px)
ARC_TICK = 3     # espessura do arco
ARM_TICK = 3     # espessura dos braços


# ═══════════════════════════════════════════════════════════════════════════════
# TOPOLOGIA DE CONEXÕES
# ═══════════════════════════════════════════════════════════════════════════════

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # Polegar
    (0,5),(5,6),(6,7),(7,8),        # Indicador
    (0,9),(9,10),(10,11),(11,12),   # Médio
    (0,13),(13,14),(14,15),(15,16), # Anular
    (0,17),(17,18),(18,19),(19,20), # Mínimo
    (5,9),(9,13),(13,17),           # Palma
]
WRIST_CONNS = {(0,5),(0,9),(0,13),(0,17)}

LM_STYLE = {
    **{0:  ("wrist", COLOR_WRIST, 8)},
    **{i:  ("thumb", COLOR_THUMB, 5) for i in (1,2,3,4)},
    **{i:  ("mcp",   COLOR_MCP,   7) for i in (5,9,13,17)},
    **{i:  ("pip",   COLOR_PIP,   6) for i in (6,10,14,18)},
    **{i:  ("dip",   COLOR_DIP,   5) for i in (7,11,15,19)},
    **{i:  ("tip",   COLOR_TIP,   4) for i in (8,12,16,20)},
}

GONIO_JOINTS = [
    (5, 0, 6,  "MCP","INDEX"),  (6, 5, 7,  "PIP","INDEX"),  (7, 6, 8,  "DIP","INDEX"),
    (9, 0,10,  "MCP","MIDDLE"), (10,9,11,  "PIP","MIDDLE"), (11,10,12, "DIP","MIDDLE"),
    (13,0,14,  "MCP","RING"),   (14,13,15, "PIP","RING"),   (15,14,16, "DIP","RING"),
    (17,0,18,  "MCP","PINKY"),  (18,17,19, "PIP","PINKY"),  (19,18,20, "DIP","PINKY"),
    (2, 1, 3,  "MCP","THUMB"),  (3, 2, 4,  "IP", "THUMB"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE DESENHO
# ═══════════════════════════════════════════════════════════════════════════════

def _stability_color(status: str) -> Tuple[int,int,int]:
    return {
        "stable":        STAB_STABLE,
        "converging":    STAB_CONV,
        "unstable":      STAB_UNSTAB,
        "uninitialized": STAB_UNINIT,
    }.get(status, STAB_UNINIT)


def _clinical_color(status: str) -> Tuple[int,int,int]:
    return {"normal": COLOR_NORMAL, "borderline": COLOR_BORDER,
            "abnormal": COLOR_ABNORM}.get(status, GRAY_LIGHT)


def _lm_px(lm, idx, W, H):
    return (int(lm[idx].x * W), int(lm[idx].y * H))


def _n2(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def _dashed(img, p1, p2, col, t=1, dash=8, gap=5):
    x1,y1=p1; x2,y2=p2
    L=math.hypot(x2-x1,y2-y1)
    if L<1: return
    ux,uy=(x2-x1)/L,(y2-y1)/L
    pos,draw=0.0,True
    while pos<L:
        e=min(pos+(dash if draw else gap),L)
        if draw:
            cv2.line(img,(int(x1+ux*pos),int(y1+uy*pos)),
                     (int(x1+ux*e),int(y1+uy*e)),col,t)
        pos,draw=e,not draw


def _arc(img, ctr, v1n, v2n, radius, color, thickness=ARC_TICK):
    """Arco poligonal entre dois vetores 2D."""
    a1 = math.degrees(math.atan2(-v1n[1], v1n[0]))
    a2 = math.degrees(math.atan2(-v2n[1], v2n[0]))
    diff = (a2 - a1 + 360) % 360
    if diff > 180:
        a1, a2 = a2, a1
        diff = 360 - diff
    steps = max(int(diff / 3), 4)
    pts = [(int(ctr[0] + radius * math.cos(math.radians(a1 + diff*i/steps))),
            int(ctr[1] - radius * math.sin(math.radians(a1 + diff*i/steps))))
           for i in range(steps+1)]
    for i in range(len(pts)-1):
        cv2.line(img, pts[i], pts[i+1], color, thickness)


def _alpha_rect(img, x, y, w, h, col, a=0.60):
    ov = img.copy()
    cv2.rectangle(ov, (x,y), (x+w,y+h), col, -1)
    cv2.addWeighted(ov, a, img, 1-a, 0, img)


def _tam_bar(img, x, y, w, h, pct):
    filled = max(0, min(int(w*pct), w))
    cv2.rectangle(img, (x,y), (x+w,y+h), (45,45,45), -1)
    for i in range(filled):
        t = i/max(filled,1)
        if t<0.5:
            b,g,r = 255, int(60+140*(t*2)), int(40*(t*2))
        else:
            b = int(255-125*((t-0.5)*2))
            g = int(200+20*((t-0.5)*2))
            r = int(40+10*((t-0.5)*2))
        cv2.line(img,(x+i,y),(x+i,y+h),(b,g,r),1)
    cv2.rectangle(img,(x,y),(x+w,y+h),GRAY_MID,1)


def _center_text(img, text, y, W, col=WHITE, sc=0.55, tk=1):
    (tw,_),_ = cv2.getTextSize(text,cv2.FONT_HERSHEY_DUPLEX,sc,tk)
    cv2.putText(img,text,((W-tw)//2,y),cv2.FONT_HERSHEY_DUPLEX,sc,col,tk,cv2.LINE_AA)


def _put(img, text, x, y, col=WHITE, sc=0.40, tk=1):
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,sc,col,tk,cv2.LINE_AA)


def _tw(text, sc=0.40, tk=1):
    (w,_),_ = cv2.getTextSize(text,cv2.FONT_HERSHEY_DUPLEX,sc,tk)
    return w


# ═══════════════════════════════════════════════════════════════════════════════
# PAINEL ESQUERDO — Hand Skeleton + Goniômetro Virtual Clínico
# ═══════════════════════════════════════════════════════════════════════════════

def _build_skeleton(frame, landmarks, angles, pw, ph, frozen, stability_map):
    """
    Constrói o painel de esqueleto com goniômetros virtuais sobrepostos.

    Elementos visuais do goniômetro (por articulação):
      ① Linha VERMELHA (braço estacionário): segmento proximal ao eixo
      ② Linha AZUL (braço móvel): segmento distal ao eixo
      ③ Círculo AMARELO no vértice: eixo articular (fulcro)
      ④ Círculo pequeno nas extremidades: pontos de referência anatômica
      ⑤ Arco colorido: magnitude do ângulo (verde=normal, vermelho=fora)
      ⑥ Label com valor suavizado + indicador de estabilidade Kalman
    """
    canvas = np.full((ph, pw, 3), BG_DARK, dtype=np.uint8)
    _center_text(canvas, "GONIOMETRIA DIGITAL", 26, pw, WHITE, 0.60)

    if not landmarks:
        _center_text(canvas, "Aguardando deteccao da mao...", ph//2, pw, GRAY_MID, 0.45)
        return canvas

    W, H = pw, ph

    # ── ① Ossos (conexões) ────────────────────────────────────────────────────
    for a, b in CONNECTIONS:
        p1 = _lm_px(landmarks, a, W, H)
        p2 = _lm_px(landmarks, b, W, H)
        if (a,b) in WRIST_CONNS or (b,a) in WRIST_CONNS:
            _dashed(canvas, p1, p2, (140,140,140), 1)
        else:
            cv2.line(canvas, p1, p2, (185,185,185), 2, cv2.LINE_AA)

    # ── ② Landmarks coloridos por tipo ───────────────────────────────────────
    for idx, (_, col, rad) in LM_STYLE.items():
        pt = _lm_px(landmarks, idx, W, H)
        cv2.circle(canvas, pt, rad+1, BLACK, -1)
        cv2.circle(canvas, pt, rad, col, -1, cv2.LINE_AA)

    # ── ③ Goniômetros virtuais: braços + arco + labels ────────────────────────
    for (ax_i, prox_i, dist_i, joint_type, finger) in GONIO_JOINTS:
        center  = _lm_px(landmarks, ax_i,   W, H)
        pt_prox = _lm_px(landmarks, prox_i, W, H)
        pt_dist = _lm_px(landmarks, dist_i, W, H)

        # Vetores 2D (pixels) para os braços
        vs = np.array([center[0]-pt_prox[0], center[1]-pt_prox[1]], float)
        vm = np.array([pt_dist[0]-center[0], pt_dist[1]-center[1]], float)
        vsn = _n2(vs)
        vmn = _n2(vm)

        # Extremidades dos braços
        es = (int(center[0]+vsn[0]*ARM_LEN), int(center[1]+vsn[1]*ARM_LEN))
        em = (int(center[0]+vmn[0]*ARM_LEN), int(center[1]+vmn[1]*ARM_LEN))

        angle_val = angles.get(finger, {}).get(joint_type)
        if angle_val is None:
            continue

        # Cores clínicas e de estabilidade
        clinical_st  = is_in_normal_range(finger, joint_type, angle_val)
        arc_col      = _clinical_color(clinical_st)
        stab_st      = (stability_map or {}).get(finger, {}).get(joint_type, "uninitialized")
        stab_col     = _stability_color(stab_st)

        # ─ BRAÇO ESTACIONÁRIO (vermelho): segmento proximal ao eixo ──────────
        cv2.line(canvas, center, es, COLOR_STAT, ARM_TICK, cv2.LINE_AA)
        # Ponto de referência anatômica na extremidade do braço estacionário
        cv2.circle(canvas, es, 4, COLOR_STAT, -1, cv2.LINE_AA)

        # ─ BRAÇO MÓVEL (azul): segmento distal ao eixo ───────────────────────
        cv2.line(canvas, center, em, COLOR_MOB, ARM_TICK, cv2.LINE_AA)
        # Ponto de referência na extremidade do braço móvel
        cv2.circle(canvas, em, 4, COLOR_MOB, -1, cv2.LINE_AA)

        # ─ VÉRTICE ARTICULAR (amarelo): eixo/fulcro do goniômetro ─────────────
        cv2.circle(canvas, center, 6, COLOR_AXIS, -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 6, WHITE, 1, cv2.LINE_AA)

        # ─ ARCO: representa o ângulo medido ──────────────────────────────────
        _arc(canvas, center, vsn, vmn, ARC_RAD, arc_col, ARC_TICK)

        # ─ LABEL: ângulo suavizado + indicador de estabilidade Kalman ─────────
        sign   = "-" if angle_val < 0 else ""
        label  = f"{joint_type} {sign}{abs(angle_val):.0f}\u00b0"
        lx, ly = center[0]+10, center[1]-8

        # Fundo semitransparente
        tw = _tw(label, 0.42)
        _alpha_rect(canvas, lx-3, ly-14, tw+10, 18, (8,8,8), 0.72)
        _put(canvas, label, lx, ly, arc_col, 0.42)

        # Indicador de estabilidade Kalman (bolinha colorida à direita do label)
        # Verde=convergido, Amarelo=convergindo, Vermelho=instável
        cv2.circle(canvas, (lx + tw + 8, ly - 6), 4, stab_col, -1, cv2.LINE_AA)

    # ── ④ Bounding box da mão ─────────────────────────────────────────────────
    xs = [int(landmarks[i].x*W) for i in range(21)]
    ys = [int(landmarks[i].y*H) for i in range(21)]
    pad = 22
    cv2.rectangle(canvas,
                  (max(0,min(xs)-pad), max(0,min(ys)-pad)),
                  (min(W-1,max(xs)+pad), min(H-1,max(ys)+pad)),
                  GRAY_MID, 1)

    # ── ⑤ Legenda compacta dos braços ────────────────────────────────────────
    lx, ly = 8, ph - 30
    cv2.line(canvas, (lx,ly), (lx+20,ly), COLOR_STAT, 2)
    _put(canvas, "Estacionario", lx+24, ly+4, GRAY_LIGHT, 0.28)
    cv2.line(canvas, (lx+125,ly), (lx+145,ly), COLOR_MOB, 2)
    _put(canvas, "Movel", lx+149, ly+4, GRAY_LIGHT, 0.28)
    cv2.circle(canvas, (lx+210,ly), 4, COLOR_AXIS, -1)
    _put(canvas, "Eixo", lx+218, ly+4, GRAY_LIGHT, 0.28)

    # ── ⑥ Overlay de frame congelado ─────────────────────────────────────────
    if frozen:
        _center_text(canvas, "[ FRAME CONGELADO ]  R = retomar",
                     ph-14, pw, COLOR_BORDER, 0.38)
        cv2.rectangle(canvas, (2,2), (pw-2,ph-2), COLOR_BORDER, 2)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
# PAINEL DIREITO — Prontuário Clínico Digital
# ═══════════════════════════════════════════════════════════════════════════════

def _build_data_panel(angles, pw, ph, frozen, stability_map):
    """
    Constrói o painel de dados clínicos com TAM e classificação ASSH.

    Estrutura por dedo:
      [● status] [Nome] [MCP°] [PIP°] [DIP°] [ABD°]
      TAM [████████░░░░] 234° (87%) [Bom]
    """
    canvas = np.full((ph, pw, 3), BG_DARK, dtype=np.uint8)

    _center_text(canvas, "Dados para cada dedo", 26, pw, WHITE, 0.55)
    _center_text(canvas,
        "Amplitude de Movimento  |  Verde=normal  |  Vermelho=fora",
        44, pw, GRAY_LIGHT, 0.32)
    cv2.line(canvas, (10,52), (pw-10,52), GRAY_DARK, 1)

    FINGERS = ["INDEX","MIDDLE","RING","PINKY","THUMB"]
    LABELS  = {"INDEX":"Indicador","MIDDLE":"Medio",
                "RING":"Anular","PINKY":"Minimo","THUMB":"Polegar"}
    BH, SY, MX = 76, 58, 14

    for fi, finger in enumerate(FINGERS):
        data = angles.get(finger, {})
        by   = SY + fi * BH

        cv2.line(canvas, (MX,by), (pw-MX,by), GRAY_DARK, 1)

        # Bullet de status global
        if finger == "THUMB":
            checks = [("MCP",data.get("MCP",0)), ("IP",data.get("IP",0))]
        else:
            checks = [("MCP",data.get("MCP",0)), ("PIP",data.get("PIP",0)),
                      ("DIP",data.get("DIP",0)), ("TAM",data.get("TAM",0))]

        sts = [is_in_normal_range(finger,m,v) for m,v in checks]
        bc  = (COLOR_ABNORM if "abnormal" in sts
               else COLOR_BORDER if "borderline" in sts
               else COLOR_NORMAL)

        bx, bry = MX+8, by+18
        cv2.circle(canvas,(bx,bry),7,bc,-1,cv2.LINE_AA)
        cv2.circle(canvas,(bx,bry),7,WHITE,1,cv2.LINE_AA)
        _put(canvas, LABELS[finger], bx+16, bry+4, WHITE, 0.48, 1)

        ax, ay = MX+10, by+36
        if finger == "THUMB":
            for metric in ("MCP","IP"):
                v   = data.get(metric, 0.0)
                col = _clinical_color(is_in_normal_range(finger, metric, v))
                txt = f"{metric}: {v:.1f}\u00b0"
                _put(canvas, txt, ax, ay, col, 0.37)
                ax += _tw(txt, 0.37) + 18
        else:
            for metric in ("MCP","PIP","DIP"):
                v   = data.get(metric, 0.0)
                col = _clinical_color(is_in_normal_range(finger, metric, v))
                txt = f"{metric}:{v:.1f}\u00b0"
                _put(canvas, txt, ax, ay, col, 0.36)
                ax += _tw(txt, 0.36) + 10

            abd = data.get("ABD", 0.0)
            _put(canvas, f"ABD:{abd:.1f}\u00b0", ax, ay, COLOR_SEC, 0.34)

            # Barra TAM com gradiente + badge ASSH
            tam      = data.get("TAM", 0.0)
            tam_info = DigitalGoniometer.classify_tam(tam)
            tam_pct  = max(0.0, min(tam/270.0, 1.0))

            bar_y  = by + 52
            bar_x  = MX + 42
            bar_w  = 115
            bar_h  = 11

            _put(canvas, "TAM", MX+10, bar_y+9, GRAY_LIGHT, 0.33)
            _tam_bar(canvas, bar_x, bar_y, bar_w, bar_h, tam_pct)

            tam_txt   = f"{tam:.0f}\u00b0"
            assh_lab  = tam_info["label"]
            assh_col  = tam_info["color_bgr"]

            vx = bar_x + bar_w + 5
            _put(canvas, tam_txt, vx, bar_y+9,
                 _clinical_color(is_in_normal_range(finger,"TAM",tam)), 0.35)

            bx2 = vx + _tw(tam_txt, 0.35) + 5
            bw2 = _tw(assh_lab, 0.30) + 8
            _alpha_rect(canvas, bx2, bar_y, bw2, 13, assh_col, 0.55)
            cv2.rectangle(canvas,(bx2,bar_y),(bx2+bw2,bar_y+13),assh_col,1)
            _put(canvas, assh_lab, bx2+4, bar_y+10, WHITE, 0.30)

    # Rodapé com ranges + legenda de estabilidade Kalman
    fy = ph - 56
    cv2.line(canvas, (10,fy-4), (pw-10,fy-4), GRAY_DARK, 1)
    lines = [
        "Ranges: MCP 85-90\u00b0  PIP 100-120\u00b0  DIP 60-80\u00b0  ABD 15-20\u00b0",
        "TAM: >=260\u00b0 Excelente  195-259\u00b0 Bom  130-194\u00b0 Razoavel  <130\u00b0 Ruim",
        "\u25cf Estavel  \u25cf Convergindo  \u25cf Instavel  (Kalman EMA\u2192Kalman)",
        "[S] Congelar   [R] Retomar   [Q] Sair",
    ]
    stab_colors = [GRAY_LIGHT, GRAY_LIGHT,
                   None,          # linha especial com cores mistas
                   COLOR_BORDER]

    for i, (line, col) in enumerate(zip(lines, stab_colors)):
        y = fy + 4 + i*14
        if i == 2:
            # Linha de legenda de estabilidade com círculos coloridos
            _put(canvas, "\u25cf", 14, y, STAB_STABLE,  0.30)
            _put(canvas, "Estavel", 26, y, GRAY_LIGHT, 0.28)
            _put(canvas, "\u25cf", 80, y, STAB_CONV,   0.30)
            _put(canvas, "Convergindo", 92, y, GRAY_LIGHT, 0.28)
            _put(canvas, "\u25cf", 180, y, STAB_UNSTAB, 0.30)
            _put(canvas, "Instavel", 192, y, GRAY_LIGHT, 0.28)
            _put(canvas, "(filtro EMA+Kalman)", pw//2+10, y, GRAY_MID, 0.27)
        else:
            _center_text(canvas, line, y, pw, col, 0.29)

    if frozen:
        cv2.rectangle(canvas,(2,2),(pw-2,ph-2),COLOR_BORDER,2)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL EXPORTADA
# ═══════════════════════════════════════════════════════════════════════════════

def draw_goniometry_overlay(
    frame:         np.ndarray,
    landmarks:     List[Any],
    angles:        Dict[str, Dict[str, float]],
    panel_w:       Optional[int] = None,
    panel_h:       Optional[int] = None,
    frozen:        bool = False,
    stability_map: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera os dois painéis do dashboard de goniometria clínica.

    Args:
        frame         : frame BGR da câmera
        landmarks     : 21 NormalizedLandmark do MediaPipe
        angles        : dict de DigitalGoniometer.compute_all(), JÁ SUAVIZADO
                        (nunca passar valores brutos — apenas EMA→Kalman filtrados)
        panel_w/h     : dimensões dos painéis (default: dimensões do frame)
        frozen        : ativa overlay de frame congelado
        stability_map : dict {"FINGER": {"JOINT": "stable"|"converging"|"unstable"}}
                        gerado por GoniometryFilterBank.get_stability()

    Returns:
        (skeleton_frame, data_panel): dois ndarrays BGR para cv2.imshow()
    """
    h, w = frame.shape[:2]
    pw = panel_w or w
    ph = panel_h or h

    skeleton = _build_skeleton(frame, landmarks, angles, pw, ph, frozen, stability_map)
    data     = _build_data_panel(angles, pw, ph, frozen, stability_map)

    return skeleton, data


def compose_side_by_side(skel: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Compõe os dois painéis em imagem única lado a lado."""
    h1, h2 = skel.shape[0], data.shape[0]
    mh = max(h1, h2)

    def _pad(img, th):
        if img.shape[0] < th:
            pad = np.full((th-img.shape[0], img.shape[1], 3), BG_DARK, np.uint8)
            return np.vstack([img, pad])
        return img

    sep = np.full((mh, 2, 3), GRAY_MID, np.uint8)
    return np.hstack([_pad(skel,mh), sep, _pad(data,mh)])
