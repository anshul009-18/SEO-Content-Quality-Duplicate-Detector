from pathlib import Path
import joblib
import pandas as pd
import numpy as np

def load_model(model_path: Path):
    try:
        if not model_path:
            return None
        p = Path(model_path)
        if not p.exists():
            return None
        model = joblib.load(str(p))
        return model
    except Exception as e:
        # graceful fallback
        print(f"Warning: failed to load model: {e}")
        return None

def rule_based_label(row):
    # High word_count > 1500 AND 50 <= readability <= 70
    wc = row.get("word_count", 0) or 0
    r = row.get("flesch_reading_ease", 0) or 0
    if wc > 1500 and 50 <= r <= 70:
        return "High"
    if wc < 500 or r < 30:
        return "Low"
    return "Medium"

def score_dataframe(df: pd.DataFrame, model=None) -> pd.DataFrame:
    out = df.copy()
    # ensure required columns exist
    if "word_count" not in out.columns or "flesch_reading_ease" not in out.columns:
        out["word_count"] = out.get("body_text", "").apply(lambda t: len(str(t).split()))
        out["flesch_reading_ease"] = out.get("body_text", "").apply(lambda t: 0.0)
    # rule-based label
    out["quality_label_rule"] = out.apply(rule_based_label, axis=1)
    # if model available, try to use it
    if model is not None:
        try:
            # model expects numeric features; try common names
            X = out[["word_count", "sentence_count", "flesch_reading_ease"]].fillna(0)
            preds = model.predict(X)
            # if model outputs labels, use them. If outputs numeric, map roughly.
            out["quality_label_model"] = preds
            out["quality_label"] = out["quality_label_model"].fillna(out["quality_label_rule"])
        except Exception as e:
            print(f"Model scoring failed: {e}")
            out["quality_label_model"] = None
            out["quality_label"] = out["quality_label_rule"]
    else:
        out["quality_label_model"] = None
        out["quality_label"] = out["quality_label_rule"]
    return out
