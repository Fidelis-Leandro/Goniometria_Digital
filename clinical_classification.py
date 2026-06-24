"""
clinical_classification.py — Lógica de classificação clínica híbrida (TAM + Funcional)
====================================================================================

Responsável por classificar o desempenho do paciente em 3 camadas:
1. Articular (TAM): usando o melhor TAM da sessão.
2. Repetições Funcionais: detecção de picos válidos e contagem de hits.
3. Sessão Híbrida: classificação final justa e inteligente combinando 1 e 2.
"""

from typing import Any, Dict, List

# Cores hexadecimais para as classificações (usadas no PDF e na interface)
COLOR_EXCELENTE = "#22c55e" # verde
COLOR_BOM = "#eab308"       # amarelo
COLOR_REGULAR = "#f97316"   # laranja
COLOR_RUIM = "#ef4444"      # vermelho

def classify_articular_tam(finger: str, best_tam_session: float) -> Dict[str, str]:
    """
    CAMADA 1: Classificação articular clássica baseada no melhor TAM da sessão.
    
    Regras:
    - Para INDEX, MIDDLE, RING, PINKY:
        Excelente: >= 260
        Bom: 195–259
        Regular: 130–194
        Ruim: < 130
    - Para THUMB:
        Excelente: > 120
        Bom: 100–120
        Ruim: < 100
    """
    if finger == "THUMB":
        if best_tam_session > 120.0:
            return {"label": "Excelente", "color": COLOR_EXCELENTE, "source": "articular_tam"}
        elif best_tam_session >= 100.0:
            return {"label": "Bom", "color": COLOR_BOM, "source": "articular_tam"}
        else:
            return {"label": "Ruim", "color": COLOR_RUIM, "source": "articular_tam"}
    else:
        if best_tam_session >= 260.0:
            return {"label": "Excelente", "color": COLOR_EXCELENTE, "source": "articular_tam"}
        elif best_tam_session >= 195.0:
            return {"label": "Bom", "color": COLOR_BOM, "source": "articular_tam"}
        elif best_tam_session >= 130.0:
            return {"label": "Regular", "color": COLOR_REGULAR, "source": "articular_tam"}
        else:
            return {"label": "Ruim", "color": COLOR_RUIM, "source": "articular_tam"}

def detect_valid_repetitions(
    tam_series: List[float],
    time_series: List[float],
    finger: str,
    min_peak_distance_s: float = 0.35,
    reset_ratio: float = 0.55,
) -> Dict[str, Any]:
    """
    CAMADA 2: Detecta repetições válidas baseadas no perfil de TAM temporal.
    
    Conta ciclos de flexão/extensão. Um pico local é computado e precisa
    'resetar' (cair a uma proporção do target_good) antes que o próximo pico valha.
    """
    if finger == "THUMB":
        target_good = 100.0
        target_excellent = 120.0
    else:
        target_good = 180.0
        target_excellent = 220.0

    valid_cycles = 0
    good_hits = 0
    excellent_hits = 0
    peaks_values = []
    
    if len(tam_series) < 3:
        return {
            "valid_cycles": 0,
            "good_hits": 0,
            "excellent_hits": 0,
            "success_rate_good": 0.0,
            "success_rate_excellent": 0.0,
            "best_peak": 0.0,
            "mean_peak": 0.0
        }

    last_peak_time = -999.0
    current_cycle_reset = True
    
    for i in range(1, len(tam_series) - 1):
        prev_val = tam_series[i - 1]
        curr_val = tam_series[i]
        next_val = tam_series[i + 1]
        t = time_series[i]
        
        if curr_val <= target_good * reset_ratio:
            current_cycle_reset = True

        if curr_val > prev_val and curr_val >= next_val:
            if current_cycle_reset and (t - last_peak_time) >= min_peak_distance_s:
                # Filtrar micromovimentos:
                # Só conta se o pico atingir um TAM mínimo funcional (ex: pelo menos reset_ratio*target_good + 10)
                # Para evitar considerar tremidos na linha de base como ciclos
                if curr_val > (target_good * reset_ratio) + 15.0:
                    valid_cycles += 1
                    peaks_values.append(curr_val)
                    last_peak_time = t
                    current_cycle_reset = False
                    
                    if curr_val > target_excellent:
                        excellent_hits += 1
                        good_hits += 1
                    elif curr_val >= target_good:
                        good_hits += 1

    best_peak = max(peaks_values) if peaks_values else 0.0
    mean_peak = sum(peaks_values) / len(peaks_values) if peaks_values else 0.0
    
    success_rate_good = good_hits / valid_cycles if valid_cycles > 0 else 0.0
    success_rate_excellent = excellent_hits / valid_cycles if valid_cycles > 0 else 0.0

    return {
        "valid_cycles": valid_cycles,
        "good_hits": good_hits,
        "excellent_hits": excellent_hits,
        "success_rate_good": float(success_rate_good),
        "success_rate_excellent": float(success_rate_excellent),
        "best_peak": float(best_peak),
        "mean_peak": float(mean_peak)
    }

def classify_functional_session(
    finger: str,
    articular: Dict[str, str],
    repetition_stats: Dict[str, Any],
    realtime_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    CAMADA 3: Classificação funcional da sessão baseada em repetições.
    """
    valid_cycles = repetition_stats["valid_cycles"]
    success_rate_good = repetition_stats["success_rate_good"]
    success_rate_excellent = repetition_stats["success_rate_excellent"]
    
    amplitude = realtime_metrics.get("amplitude", 0.0)
    vel_media = realtime_metrics.get("vel_media", 0.0)
    regularidade = realtime_metrics.get("regularidade", "-")

    min_req = (
        valid_cycles >= 5 and 
        amplitude >= 40.0 and 
        vel_media >= 20.0 and 
        regularidade != "Irregular" and regularidade != "-"
    )

    if min_req and success_rate_excellent >= 0.80:
        label = "Excelente"
        color = COLOR_EXCELENTE
    elif min_req and success_rate_good >= 0.80:
        label = "Bom"
        color = COLOR_BOM
    elif valid_cycles >= 3 and success_rate_good >= 0.50:
        label = "Regular"
        color = COLOR_REGULAR
    else:
        label = "Ruim"
        color = COLOR_RUIM

    return {
        "label": label,
        "color": color,
        "source": "functional_session",
        "valid_cycles": valid_cycles,
        "success_rate_good": success_rate_good,
        "success_rate_excellent": success_rate_excellent
    }

def classify_final_session_result(
    finger: str,
    articular: Dict[str, str],
    functional: Dict[str, Any],
    repetition_stats: Dict[str, Any]
) -> Dict[str, str]:
    """
    CAMADA 4 (Final): Classificação Híbrida do relatório.
    """
    art_label = articular["label"]
    func_label = functional["label"]
    best_peak = repetition_stats["best_peak"]
    valid_cycles = repetition_stats["valid_cycles"]
    success_rate_good = repetition_stats["success_rate_good"]
    success_rate_excellent = repetition_stats["success_rate_excellent"]
    excellent_hits = repetition_stats["excellent_hits"]

    target_good = 100.0 if finger == "THUMB" else 180.0
    target_exc = 120.0 if finger == "THUMB" else 220.0

    final_label = "Ruim"
    explanation = "Desempenho abaixo da meta funcional e articular esperada durante a sessão."

    if art_label in ["Regular", "Ruim"] and func_label == "Excelente" and valid_cycles >= 5 and success_rate_excellent >= 0.80:
        final_label = "Bom"
        explanation = f"atingiu pico funcional excelente em {excellent_hits} de {valid_cycles} repetições válidas, demonstrando boa coordenação apesar do TAM clássico reduzido."
    elif art_label == "Bom" and func_label == "Excelente":
        final_label = "Excelente"
        explanation = f"aliou um bom TAM clássico a uma excelente constância funcional, superando a meta de excelência em {(success_rate_excellent*100):.0f}% dos ciclos."
    elif art_label == "Ruim" and func_label == "Bom" and valid_cycles >= 5 and success_rate_good >= 0.80:
        final_label = "Regular"
        explanation = f"embora o TAM máximo tenha sido baixo, manteve consistência funcional com {(success_rate_good*100):.0f}% das repetições acima da meta."
    else:
        ranks = {"Excelente": 4, "Bom": 3, "Regular": 2, "Ruim": 1}
        r_art = ranks[art_label]
        r_func = ranks[func_label]
        
        if r_art >= 3 and r_func >= 3:
            final_label = art_label
            explanation = "desempenho articular robusto e consistente ao longo das repetições."
        else:
            if r_art <= r_func:
                final_label = art_label
                if r_art == 1:
                    explanation = "limitação articular proeminente restringiu a avaliação geral, independente das tentativas."
                else:
                    explanation = "a pontuação final refletiu a capacidade articular máxima obtida (TAM), que atuou como limitante."
            else:
                final_label = func_label
                explanation = "a falta de constância ou ciclos funcionais válidos limitou a pontuação da sessão."

    # TRAVAS FINAIS
    if final_label == "Excelente" and best_peak < target_exc:
        final_label = "Bom" if best_peak >= target_good else "Regular"
        explanation += " (pontuação teto aplicada pois o pico absoluto não atingiu a meta de excelência)."
        
    if final_label == "Bom" and best_peak < target_good:
        final_label = "Regular" if art_label == "Regular" else "Ruim"
        explanation += " (pontuação teto aplicada pois o pico absoluto não atingiu a meta boa)."

    colors = {
        "Excelente": COLOR_EXCELENTE,
        "Bom": COLOR_BOM,
        "Regular": COLOR_REGULAR,
        "Ruim": COLOR_RUIM
    }

    return {
        "label": final_label,
        "color": colors.get(final_label, COLOR_RUIM),
        "source": "final_hybrid",
        "explanation": explanation
    }

def generate_clinical_observation_text(final_hybrid: Dict[str, str]) -> str:
    """Gera o texto curto e automático baseado na label final."""
    lbl = final_hybrid["label"]
    if lbl == "Excelente":
        return "Desempenho funcional consistente, com repetidas flexões atingindo faixa excelente durante a sessão."
    elif lbl == "Bom":
        return "Boa execução funcional, com múltiplas repetições acima da meta angular esperada."
    elif lbl == "Regular":
        return "Movimento funcional presente, porém com menor consistência ou menor taxa de sucesso."
    else:
        return "Desempenho abaixo da meta funcional esperada durante a sessão."
