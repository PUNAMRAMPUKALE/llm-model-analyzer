from typing import Dict, Any
import textstat

def readability_score(text: str) -> Dict[str, Any]:
    try:
        flesch = textstat.flesch_reading_ease(text)
        grade = textstat.text_standard(text, float_output=True)
    except Exception:
        flesch, grade = 50.0, 10.0
    # Normalize to 0..1 preferring mid-grade readability ~8-10
    score = 1.0 - min(1.0, abs(grade - 9.0) / 9.0)
    return {"REA": float(score), "flesch": float(flesch), "grade": float(grade)}
