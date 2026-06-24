import pytest
import numpy as np
from goniometry import DigitalGoniometer, TAM_CLASSIFICATION_THUMB
from dashboard_utils import assh_classify, assh_classify_thumb, classify_hand_state
from goniometry_csv import GoniometryCSVLogger
import os
import tempfile
import csv

class MockLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def create_straight_hand():
    """Cria uma mão aberta com todos os dedos retos (ângulos ≈ 0)."""
    lms = [MockLandmark(0, 0, 0) for _ in range(21)]
    
    # Punho
    lms[0] = MockLandmark(0.5, 1.0, 0.0)
    
    # Indicador (colinear)
    lms[5] = MockLandmark(0.4, 0.8, 0.0)
    lms[6] = MockLandmark(0.3, 0.6, 0.0)
    lms[7] = MockLandmark(0.2, 0.4, 0.0)
    lms[8] = MockLandmark(0.1, 0.2, 0.0)
    
    # Médio
    lms[9]  = MockLandmark(0.5, 0.8, 0.0)
    lms[10] = MockLandmark(0.5, 0.6, 0.0)
    lms[11] = MockLandmark(0.5, 0.4, 0.0)
    lms[12] = MockLandmark(0.5, 0.2, 0.0)
    
    # Anelar
    lms[13] = MockLandmark(0.6, 0.8, 0.0)
    lms[14] = MockLandmark(0.7, 0.6, 0.0)
    lms[15] = MockLandmark(0.8, 0.4, 0.0)
    lms[16] = MockLandmark(0.9, 0.2, 0.0)
    
    # Mínimo
    lms[17] = MockLandmark(0.7, 0.8, 0.0)
    lms[18] = MockLandmark(0.9, 0.6, 0.0)
    lms[19] = MockLandmark(1.1, 0.4, 0.0)
    lms[20] = MockLandmark(1.3, 0.2, 0.0)
    
    # Polegar
    lms[1] = MockLandmark(0.3, 0.9, 0.0)
    lms[2] = MockLandmark(0.1, 0.8, 0.0)
    lms[3] = MockLandmark(-0.1, 0.7, 0.0)
    lms[4] = MockLandmark(-0.3, 0.6, 0.0)
    
    return lms

def create_flexed_hand():
    """Cria uma mão fletida para dentro (dobrada no eixo Z)."""
    lms = create_straight_hand()
    
    # Normal será baseado em wrist(0.5, 1.0, 0), index_mcp(0.4, 0.8, 0), pinky_mcp(0.7, 0.8, 0)
    # v1 = index_mcp - wrist = (-0.1, -0.2, 0)
    # v2 = pinky_mcp - wrist = (0.2, -0.2, 0)
    # normal (após correção v2 x v1) = (0.2, -0.2, 0) x (-0.1, -0.2, 0) = (0, 0, -0.04 - 0.02) = (0, 0, -0.06)
    # Então normal aponta para -Z (dorso da mão em coordenadas mediapipe)
    # Flexão deve ocorrer na direção +Z para ter sinal positivo.
    
    # Dobrar indicador em Z positivo
    lms[6] = MockLandmark(0.4, 0.8, 0.2)  # PIP fletida
    lms[7] = MockLandmark(0.4, 0.9, 0.3)  # DIP fletida
    lms[8] = MockLandmark(0.4, 1.0, 0.2)  # TIP fletida
    
    # Dobrar médio
    lms[10] = MockLandmark(0.5, 0.8, 0.2)
    lms[11] = MockLandmark(0.5, 0.9, 0.3)
    lms[12] = MockLandmark(0.5, 1.0, 0.2)
    
    # Dobrar anelar
    lms[14] = MockLandmark(0.6, 0.8, 0.2)
    lms[15] = MockLandmark(0.6, 0.9, 0.3)
    lms[16] = MockLandmark(0.6, 1.0, 0.2)
    
    # Dobrar mínimo
    lms[18] = MockLandmark(0.7, 0.8, 0.2)
    lms[19] = MockLandmark(0.7, 0.9, 0.3)
    lms[20] = MockLandmark(0.7, 1.0, 0.2)
    
    # Dobrar polegar
    lms[3] = MockLandmark(0.1, 0.8, 0.2)
    lms[4] = MockLandmark(0.1, 0.9, 0.3)
    
    return lms

def test_straight_hand_angles():
    # 1. Mão aberta → ângulos próximos de 0°
    lms = create_straight_hand()
    gonio = DigitalGoniometer()
    res = gonio.compute_all(lms, is_right_hand=True)
    
    for finger in ["INDEX", "MIDDLE", "RING", "PINKY"]:
        assert abs(res[finger]["MCP"]) < 5.0
        assert abs(res[finger]["PIP"]) < 5.0
        assert abs(res[finger]["DIP"]) < 5.0
        assert abs(res[finger]["TAM"]) < 10.0

def test_flexed_hand_angles_positive():
    # 2. Mão fechada → ângulos positivos fisiológicos
    # 8. Valores negativos não aparecem em flexão normal
    lms = create_flexed_hand()
    gonio = DigitalGoniometer()
    res = gonio.compute_all(lms, is_right_hand=True)
    
    for finger in ["INDEX", "MIDDLE", "RING", "PINKY"]:
        assert res[finger]["MCP"] > 0
        assert res[finger]["PIP"] > 0
        assert res[finger]["DIP"] > 0

    assert res["THUMB"]["MCP"] > 0
    assert res["THUMB"]["IP"] > 0

def test_tam_increases_with_flexion():
    # 3. TAM dos dedos longos cresce com flexão
    gonio = DigitalGoniometer()
    res_straight = gonio.compute_all(create_straight_hand(), is_right_hand=True)
    res_flexed = gonio.compute_all(create_flexed_hand(), is_right_hand=True)
    
    for finger in ["INDEX", "MIDDLE", "RING", "PINKY"]:
        assert res_flexed[finger]["TAM"] > res_straight[finger]["TAM"]

def test_thumb_tam_is_calculated():
    # 4. TAM do polegar é calculado e incluído no retorno
    gonio = DigitalGoniometer()
    res = gonio.compute_all(create_straight_hand(), is_right_hand=True)
    
    assert "THUMB" in res
    assert "TAM" in res["THUMB"]
    assert res["THUMB"]["TAM"] >= 0

def test_csv_contains_thumb_tam():
    # 5. CSV contém THUMB_TAM — verifica cabeçalho e gravação
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        logger = GoniometryCSVLogger(csv_path)
        
        gonio = DigitalGoniometer()
        angles = gonio.compute_all(create_flexed_hand(), is_right_hand=True)
        
        logger.log(1, angles)
        logger.close()
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert "THUMB_TAM" in reader.fieldnames
            
            rows = list(reader)
            assert len(rows) == 1
            assert float(rows[0]["THUMB_TAM"]) == angles["THUMB"]["TAM"]

def test_thumb_classification_logic():
    # 6. Classificação do polegar não usa régua dos outros dedos
    label_thumb, _ = assh_classify_thumb(100.0)
    label_others, _ = assh_classify(100.0)
    
    # 100 graus é "Bom" para polegar (max ~120) mas "Ruim" para dedo longo (max ~270)
    assert label_thumb == "Bom"
    assert label_others == "Ruim"
    
def test_dashboard_utils_classify_hand_state():
    # 7. Dashboard e funções utilitárias não quebram com THUMB_TAM
    gonio = DigitalGoniometer()
    angles = gonio.compute_all(create_flexed_hand(), is_right_hand=True)
    
    # Força TAM do polegar para fechar (>= 85.0)
    angles["THUMB"]["TAM"] = 90.0
    
    state = classify_hand_state(angles)
    assert "finger_states" in state
    assert "THUMB" in state["finger_states"]
    assert state["finger_states"]["THUMB"]["TAM"] == 90.0
    assert state["finger_states"]["THUMB"]["closed"] == True
    assert state["finger_states"]["THUMB"]["assh_label"] == "Bom"
