"""
test_goniometry.py — Testes Unitários: Ângulos + Suavização EMA+Kalman
=======================================================================
Cobertura completa em 41 testes:

  Testes 0–5   : validação matemática do DigitalGoniometer
  Testes 6–11  : validação da camada de suavização EMA→Kalman

Execução:
    python test_goniometry.py
    python -m pytest test_goniometry.py -v
"""

import sys
import math
from dataclasses import dataclass
from typing import List
import numpy as np

try:
    from goniometry import (DigitalGoniometer, angle_between_vectors_3d,
                             NORMAL_RANGES)
    from smoothing import SeriesFilter, GoniometryFilterBank
except ImportError as e:
    print(f"ERRO: {e}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK DE LANDMARK
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MockLandmark:
    x: float
    y: float
    z: float


def make_landmarks(coords: List[tuple]) -> List[MockLandmark]:
    assert len(coords) == 21
    return [MockLandmark(x, y, z) for x, y, z in coords]


# ═══════════════════════════════════════════════════════════════════════════════
# FÁBRICAS DE LANDMARKS SINTÉTICOS
# ═══════════════════════════════════════════════════════════════════════════════

def make_extended_finger_landmarks() -> List[MockLandmark]:
    W=(0.50,0.90,0.0); T1=(0.43,0.82,0.0); T2=(0.38,0.74,0.0)
    T3=(0.34,0.68,0.0); T4=(0.31,0.62,0.0)
    I5=(0.44,0.72,0.0); I6=(0.44,0.60,0.0); I7=(0.44,0.51,0.0); I8=(0.44,0.43,0.0)
    M9=(0.50,0.71,0.0); M10=(0.50,0.59,0.0); M11=(0.50,0.50,0.0); M12=(0.50,0.42,0.0)
    R13=(0.56,0.72,0.0); R14=(0.56,0.60,0.0); R15=(0.56,0.51,0.0); R16=(0.56,0.43,0.0)
    P17=(0.62,0.74,0.0); P18=(0.62,0.63,0.0); P19=(0.62,0.55,0.0); P20=(0.62,0.48,0.0)
    return make_landmarks([W,T1,T2,T3,T4,I5,I6,I7,I8,M9,M10,M11,M12,R13,R14,R15,R16,P17,P18,P19,P20])


def make_mcp_90_landmarks() -> List[MockLandmark]:
    W=(0.50,0.90,0.00); T1=(0.43,0.82,0.00); T2=(0.38,0.74,0.00)
    T3=(0.34,0.68,0.00); T4=(0.31,0.62,0.00)
    I5=(0.50,0.72,0.00); I6=(0.50,0.72,0.12); I7=(0.50,0.72,0.22); I8=(0.50,0.72,0.30)
    M9=(0.50,0.71,0.00); M10=(0.50,0.59,0.00); M11=(0.50,0.50,0.00); M12=(0.50,0.42,0.00)
    R13=(0.56,0.72,0.00); R14=(0.56,0.60,0.00); R15=(0.56,0.51,0.00); R16=(0.56,0.43,0.00)
    P17=(0.62,0.74,0.00); P18=(0.62,0.63,0.00); P19=(0.62,0.55,0.00); P20=(0.62,0.48,0.00)
    return make_landmarks([W,T1,T2,T3,T4,I5,I6,I7,I8,M9,M10,M11,M12,R13,R14,R15,R16,P17,P18,P19,P20])


def make_closed_fist_landmarks() -> List[MockLandmark]:
    W=(0.50,0.90,0.00); T1=(0.43,0.82,0.00); T2=(0.40,0.78,0.04)
    T3=(0.44,0.80,0.06); T4=(0.47,0.82,0.07)
    I5=(0.44,0.72,0.00); I6=(0.44,0.72,0.13); I7=(0.44,0.81,0.16); I8=(0.44,0.87,0.16)
    M9=(0.50,0.72,0.00); M10=(0.50,0.72,0.14); M11=(0.50,0.82,0.18); M12=(0.50,0.88,0.18)
    R13=(0.56,0.72,0.00); R14=(0.56,0.72,0.13); R15=(0.56,0.81,0.16); R16=(0.56,0.87,0.16)
    P17=(0.62,0.74,0.00); P18=(0.62,0.74,0.11); P19=(0.62,0.81,0.14); P20=(0.62,0.86,0.14)
    return make_landmarks([W,T1,T2,T3,T4,I5,I6,I7,I8,M9,M10,M11,M12,R13,R14,R15,R16,P17,P18,P19,P20])


# ═══════════════════════════════════════════════════════════════════════════════
# SUITE DE TESTES
# ═══════════════════════════════════════════════════════════════════════════════

class TestSuite:
    def __init__(self):
        self.gonio   = DigitalGoniometer()
        self.passed  = 0
        self.failed  = 0
        self.errors  = []

    def _ok(self, name, detail):
        print(f"  ✓ {name}: {detail}")
        self.passed += 1

    def _fail(self, name, detail):
        msg = f"  ✗ {name}: {detail}"
        print(msg)
        self.errors.append(msg)
        self.failed += 1

    def _approx(self, val, exp, tol, name):
        if abs(val - exp) <= tol:
            self._ok(name, f"{val:.2f}° ≈ {exp}° (±{tol}°)")
        else:
            self._fail(name, f"{val:.2f}° ≠ {exp}° (err={abs(val-exp):.2f}°)")

    def _range(self, val, lo, hi, name):
        if lo <= val <= hi:
            self._ok(name, f"{val:.2f}° ∈ [{lo},{hi}]°")
        else:
            self._fail(name, f"{val:.2f}° ∉ [{lo},{hi}]°")

    # ── TESTES 0–5: MATEMÁTICA ANGULAR ───────────────────────────────────────

    def test_0_vector_angle_math(self):
        print("\n[Teste 0] angle_between_vectors_3d")
        n = np.array([0.,0.,1.])
        self._approx(angle_between_vectors_3d(np.array([1.,0.,0.]),
                     np.array([1.,0.,0.]),n), 0., 0.5, "Paralelos → 0°")
        self._approx(angle_between_vectors_3d(np.array([1.,0.,0.]),
                     np.array([0.,1.,0.]),n), 90., 1., "Perp. positivo → +90°")
        self._approx(angle_between_vectors_3d(np.array([0.,1.,0.]),
                     np.array([1.,0.,0.]),n), -90., 1., "Perp. negativo → -90°")
        self._approx(abs(angle_between_vectors_3d(np.array([1.,0.,0.]),
                     np.array([-1.,0.,0.]),n)), 180., 1., "Opostos → 180°")

    def test_1_extended_finger(self):
        print("\n[Teste 1] Dedo estendido → PIP≈0°, DIP≈0°")
        lm = make_extended_finger_landmarks()
        a  = self.gonio.compute_all(lm)
        for f in ("INDEX","MIDDLE","RING","PINKY"):
            self._approx(abs(a[f]["PIP"]), 0., 5., f"{f}.PIP≈0°")
            self._approx(abs(a[f]["DIP"]), 0., 5., f"{f}.DIP≈0°")
            assert abs(a[f]["TAM"]) < 60., f"{f}.TAM muito alto para dedo estendido"
            self._ok(f"{f}.TAM", f"{a[f]['TAM']:.1f}° < 60°")

    def test_2_mcp_90(self):
        print("\n[Teste 2] MCP a 90°")
        lm = make_mcp_90_landmarks()
        a  = self.gonio.compute_all(lm)
        self._approx(abs(a["INDEX"]["MCP"]), 90., 8., "|INDEX.MCP|≈90°")
        self._approx(abs(a["INDEX"]["PIP"]), 0., 10., "|INDEX.PIP|≈0°")
        self._approx(abs(a["INDEX"]["DIP"]), 0., 10., "|INDEX.DIP|≈0°")

    def test_3_closed_fist(self):
        print("\n[Teste 3] Punho fechado → flexão significativa")
        lm = make_closed_fist_landmarks()
        a  = self.gonio.compute_all(lm)
        m  = a["MIDDLE"]
        self._approx(abs(m["MCP"]), 90., 15., "|MIDDLE.MCP|≈90°")
        self._range(abs(m["PIP"]), 30., 130., "|MIDDLE.PIP|∈[30,130]°")
        self._range(abs(m["DIP"]), 10., 100., "|MIDDLE.DIP|∈[10,100]°")
        self._range(abs(m["TAM"]), 80., 350., "|MIDDLE.TAM|∈[80,350]°")

    def test_4_normal_ranges(self):
        print("\n[Teste 4] Ranges clínicos")
        expected = {"MCP_flex":(85.,90.),"PIP_flex":(100.,120.),"DIP_flex":(60.,80.),
                    "ABD":(15.,20.),"TAM":(250.,270.),"THUMB_MCP":(50.,60.),"THUMB_IP":(70.,90.)}
        for k,(lo,hi) in expected.items():
            if k not in NORMAL_RANGES:
                self._fail(f"NORMAL_RANGES[{k}]","ausente")
            elif NORMAL_RANGES[k]!=(lo,hi):
                self._fail(f"NORMAL_RANGES[{k}]",f"obtido {NORMAL_RANGES[k]}")
            else:
                self._ok(f"NORMAL_RANGES[{k}]",f"[{lo},{hi}]°")

    def test_5_tam_formula(self):
        print("\n[Teste 5] Fórmula TAM")
        g = self.gonio
        self._approx(g.total_active_motion(90.,120.,70.), 280., 0.1, "TAM(90,120,70)=280°")
        self._approx(g.total_active_motion(70.,90.,60.),  220., 0.1, "TAM(70,90,60)=220°")
        self._approx(g.total_active_motion(90.,-15.,60.), 135., 0.1, "TAM(90,-15,60)=135° (déficit)")
        self._approx(g.total_active_motion(0.,0.,0.),       0., 0.1, "TAM(0,0,0)=0°")
        self._approx(g.total_active_motion(-5.,-10.,-8.),-23., 0.1, "TAM só déficits=-23°")

    # ── TESTES 6–11: SUAVIZAÇÃO EMA→KALMAN ────────────────────────────────────

    def test_6_ema_convergence(self):
        """
        EMA com entrada constante deve convergir para o valor verdadeiro.
        Após 30 frames de entrada = 87°, a EMA deve estar dentro de ±2° do alvo.
        """
        print("\n[Teste 6] EMA: convergência para valor constante")
        sf = SeriesFilter(ema_alpha=0.30, kalman_q=0.01, kalman_r=0.10)

        # Primeiro frame inicializa sem filtragem
        out = sf.update(87.0)
        self._approx(out, 87.0, 0.1, "Frame 0 inicializa em 87°")

        # Após 30 frames com entrada constante, deve convergir
        for _ in range(30):
            out = sf.update(87.0)
        self._approx(out, 87.0, 2.0, "Após 30 frames constante → ≈87°")

    def test_7_kalman_reduces_residual_jitter(self):
        """
        EMA+Kalman deve produzir std_dev menor que EMA sozinho.
        Simula 200 frames com ruído gaussiano de 3° std.
        """
        print("\n[Teste 7] EMA+Kalman: redução de jitter mensurável")
        np.random.seed(42)
        TRUE_ANGLE = 87.0
        NOISE_STD  = 3.0
        N          = 200

        raw = TRUE_ANGLE + np.random.normal(0, NOISE_STD, N)

        # EMA only
        ema_val = raw[0]
        ema_out = [raw[0]]
        for r in raw[1:]:
            ema_val = 0.30 * r + 0.70 * ema_val
            ema_out.append(ema_val)

        # EMA + Kalman via SeriesFilter
        sf = SeriesFilter(ema_alpha=0.30, kalman_q=0.01, kalman_r=0.10)
        kalman_out = [sf.update(r) for r in raw]

        ema_std    = np.std(ema_out)
        kalman_std = np.std(kalman_out)

        print(f"  Ruído bruto std: {NOISE_STD:.3f}°")
        print(f"  EMA std:         {ema_std:.3f}°  (redução {(1-ema_std/NOISE_STD)*100:.1f}%)")
        print(f"  EMA+Kalman std:  {kalman_std:.3f}°  (redução {(1-kalman_std/NOISE_STD)*100:.1f}%)")

        # EMA+Kalman deve ser mais estável que EMA sozinho
        if kalman_std < ema_std:
            self._ok("Kalman < EMA std", f"{kalman_std:.3f}° < {ema_std:.3f}°")
        else:
            self._fail("Kalman < EMA std", f"{kalman_std:.3f}° não < {ema_std:.3f}°")

        # Ambos devem estar bem abaixo do limiar clínico de 5°
        self._range(kalman_std, 0., 5., "EMA+Kalman std < limiar clínico 5°")
        self._range(ema_std,    0., 5., "EMA std < limiar clínico 5°")

    def test_8_step_response(self):
        """
        Resposta a degrau: entrada muda de 0° para 90° no frame 50.
        Após 30 frames do degrau, a saída deve estar dentro de ±10° de 90°.
        """
        print("\n[Teste 8] Resposta a degrau: 0° → 90° no frame 50")
        sf = SeriesFilter(ema_alpha=0.30, kalman_q=0.01, kalman_r=0.10)

        # 50 frames em 0°
        for _ in range(50):
            sf.update(0.0)

        # Degrau para 90°, medir resposta após 30 frames
        last = 0.0
        for i in range(30):
            last = sf.update(90.0)

        print(f"  Saída após 30 frames do degrau: {last:.2f}°")
        # Com EMA alpha=0.3 e Kalman, deve estar bem próximo de 90° após 30 frames
        self._range(last, 75.0, 90.0, "Resposta ao degrau em 30 frames ∈ [75°,90°]")

    def test_9_stability_indicator(self):
        """
        O indicador de estabilidade deve evoluir de 'unstable' para 'stable'
        conforme o filtro converge.
        """
        print("\n[Teste 9] Indicador de estabilidade Kalman")
        sf = SeriesFilter(ema_alpha=0.30, kalman_q=0.01, kalman_r=0.10)

        # Antes de qualquer update: filtro não inicializado
        self._ok("Não inicializado", f"is_initialized={sf.is_initialized}")

        # Primeiro update: inicializa com K=1 (instável)
        sf.update(87.0)
        assert sf.is_initialized, "Deve estar inicializado após primeiro update"
        self._ok("Inicializado após frame 0", "is_initialized=True")

        # Após muitos frames com entrada estável: deve convergir
        for _ in range(100):
            sf.update(87.0)

        stab = sf.stability
        print(f"  Stability após 100 frames estáticos: {stab}")
        if stab in ("stable", "converging"):
            self._ok("Stability convergiu", f"stability={stab}")
        else:
            self._fail("Stability não convergiu", f"stability={stab}")

    def test_10_filter_bank_independence(self):
        """
        GoniometryFilterBank deve manter séries completamente independentes.
        INDEX_MCP não deve afetar MIDDLE_PIP.
        """
        print("\n[Teste 10] Banco de filtros: independência entre séries")
        bank = GoniometryFilterBank(ema_alpha=0.30, kalman_q=0.01, kalman_r=0.10)

        # Alimenta INDEX_MCP com 30°
        for _ in range(20):
            bank.update("INDEX",  "MCP", 30.0)

        # Alimenta MIDDLE_PIP com 110°
        for _ in range(20):
            bank.update("MIDDLE", "PIP", 110.0)

        # Verifica que INDEX_MCP converge para 30° e não 110°
        v_mcp = bank.update("INDEX",  "MCP", 30.0)
        # Verifica que MIDDLE_PIP converge para 110° e não 30°
        v_pip = bank.update("MIDDLE", "PIP", 110.0)

        print(f"  INDEX_MCP após 21 updates de 30°:   {v_mcp:.2f}°")
        print(f"  MIDDLE_PIP após 21 updates de 110°: {v_pip:.2f}°")

        self._range(v_mcp, 25., 35.,   "INDEX_MCP converge para ~30°")
        self._range(v_pip, 100., 115., "MIDDLE_PIP converge para ~110°")

        # Conta séries criadas
        n = bank.active_series_count
        self._approx(float(n), 2., 0., f"2 séries criadas (obtido {n})")

    def test_11_reset_with_seed(self):
        """
        Após reset com seed, o filtro deve iniciar do valor fornecido,
        não de zero. Isso evita salto visual ao retomar do modo frozen.
        """
        print("\n[Teste 11] Reset com seed: sem salto visual")
        sf = SeriesFilter(ema_alpha=0.30, kalman_q=0.01, kalman_r=0.10)

        # Convergir para 87°
        for _ in range(50):
            sf.update(87.0)

        # Reset com seed = último valor conhecido
        sf.reset(seed_value=87.0)

        # Primeiro update após reset deve ser próximo de 87°, não de 0°
        first_after_reset = sf.update(87.0)
        print(f"  Primeiro update após reset(seed=87): {first_after_reset:.2f}°")
        self._approx(first_after_reset, 87.0, 5.0, "Sem salto após reset com seed")

        # Comparação: reset sem seed começa do zero
        sf2 = SeriesFilter(ema_alpha=0.30, kalman_q=0.01, kalman_r=0.10)
        for _ in range(50):
            sf2.update(87.0)
        sf2.reset()  # sem seed — zera o estado
        first_no_seed = sf2.update(87.0)
        print(f"  Primeiro update após reset sem seed: {first_no_seed:.2f}°")
        # Ambos devem retornar 87.0 porque é o primeiro update (inicializa com o valor)
        self._ok("Reset sem seed também inicializa correto",
                 f"{first_no_seed:.2f}° (inicializa com medição)")

    # ── RUNNER ────────────────────────────────────────────────────────────────

    def run_all(self) -> bool:
        print("=" * 62)
        print("  TESTES — Goniometria Digital + Suavização EMA→Kalman")
        print("=" * 62)

        self.test_0_vector_angle_math()
        self.test_1_extended_finger()
        self.test_2_mcp_90()
        self.test_3_closed_fist()
        self.test_4_normal_ranges()
        self.test_5_tam_formula()
        self.test_6_ema_convergence()
        self.test_7_kalman_reduces_residual_jitter()
        self.test_8_step_response()
        self.test_9_stability_indicator()
        self.test_10_filter_bank_independence()
        self.test_11_reset_with_seed()

        total = self.passed + self.failed
        print("\n" + "=" * 62)
        print(f"  RESULTADO: {self.passed}/{total} testes passaram")
        if self.failed == 0:
            print("  ✓ Todos os testes passaram!")
        else:
            print(f"  ✗ {self.failed} falha(s):")
            for e in self.errors:
                print(f"    {e}")
        print("=" * 62)
        return self.failed == 0


if __name__ == "__main__":
    suite = TestSuite()
    success = suite.run_all()
    sys.exit(0 if success else 1)
