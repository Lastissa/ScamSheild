"""
tune_thresholds.py - recompute risk-score thresholds on the validation set.

Picks separated HIGH / LOW bands so the output fields stay in sync:

  risk_score = round(100 * calibrated probability)   (integer 0-100)
  risk_level = "high"   if risk_score >= HIGH_SCORE
  risk_level = "low"    if risk_score <= LOW_SCORE
  risk_level = "medium" otherwise
  is_scam    = (risk_level == "high")

Constraints (from the spec), enforced on the validation split:
  - "high" must reach precision >= 0.95
  - "low" must miss very few scams: >= 99% of validation scams score above LOW

Plus a minimum gap so the bands don't collapse (HIGH_SCORE - LOW_SCORE >= 20).

Writes the chosen thresholds into artifacts/model_meta.json.
Run from model/:  python src/tune_thresholds.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROCESSED_DIR, ARTIFACTS_DIR, load_csv

MIN_GAP = 20          # points, so bands never collapse
HIGH_PREC_TARGET = 0.95
LOW_RECALL_TARGET = 0.99   # fraction of scams scoring above LOW


def main() -> None:
    print("=== tuning risk thresholds on validation ===")
    model = joblib.load(ARTIFACTS_DIR / "scam_model.joblib")

    clean = load_csv(PROCESSED_DIR / "clean_dataset.csv")
    assign = load_csv(PROCESSED_DIR / "split_assignments.csv")
    clean = clean.merge(assign[["row_id", "split"]], on="row_id")
    val = clean[clean["split"] == "validation"].reset_index(drop=True)
    y_val = val["label"].to_numpy()
    proba = model.predict_proba(val["text"].tolist())[:, 1]

    def precision_at(t: float) -> float:
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y_val == 1)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        return tp / (tp + fp) if (tp + fp) else 0.0

    def recall_above(t: float) -> float:
        scams = y_val == 1
        if not scams.any():
            return 1.0
        return float((proba[scams] > t).mean())

    grid = np.round(np.linspace(0.0, 1.0, 1001), 4)

    # HIGH: smallest threshold where precision >= 0.95, but clamp up so it's
    # intuitively "high". Precision only grows with threshold, so clamping up
    # keeps precision >= 0.95.
    high_candidates = [t for t in grid if precision_at(t) >= HIGH_PREC_TARGET]
    thr_high = float(min(high_candidates)) if high_candidates else 0.5
    thr_high = max(thr_high, 0.60)   # intuitive floor: >= 60 = high

    # LOW: largest threshold where >= 99% of scams score strictly above it.
    low_candidates = [t for t in grid if recall_above(t) >= LOW_RECALL_TARGET]
    thr_low = float(max(low_candidates)) if low_candidates else 0.20

    # Enforce a gap so low < high with a real medium band.
    if thr_high - thr_low < MIN_GAP / 100:
        mid = (thr_high + thr_low) / 2
        thr_low = max(0.0, mid - MIN_GAP / 200)
        thr_high = min(1.0, mid + MIN_GAP / 200)

    high_score = int(round(thr_high * 100))
    low_score = int(round(thr_low * 100))
    if low_score >= high_score:
        low_score = max(0, high_score - MIN_GAP)

    print(f"thr_high={thr_high} -> risk_score >= {high_score} = high "
          f"(validation precision at/above = {precision_at(thr_high):.4f})")
    print(f"thr_low={thr_low} -> risk_score <= {low_score} = low "
          f"(share of scams above = {recall_above(thr_low):.4f})")
    print(f"gap = {high_score - low_score} points")

    # sanity: show band distribution on validation
    scores = np.round(100 * proba).astype(int)
    hi = int((scores >= high_score).sum())
    lo = int((scores <= low_score).sum())
    print(f"validation: {hi} high / {lo} low / {len(scores)-hi-lo} medium "
          f"(of {len(scores)})")

    # ---- persist to model_meta.json ----
    meta_path = ARTIFACTS_DIR / "model_meta.json"
    meta = json.loads(meta_path.read_text())
    meta.setdefault("thresholds", {})
    meta["thresholds"]["high_score"] = high_score / 100.0
    meta["thresholds"]["low_score"] = low_score / 100.0
    meta["thresholds"]["high_score_int"] = high_score
    meta["thresholds"]["low_score_int"] = low_score
    meta["thresholds"]["high_precision_target"] = HIGH_PREC_TARGET
    meta["thresholds"]["low_scams_recall_target"] = LOW_RECALL_TARGET
    meta["thresholds"]["validation_precision_at_high"] = round(
        float(precision_at(thr_high)), 4)
    meta["thresholds"]["validation_scam_recall_above_low"] = round(
        float(recall_above(thr_low)), 4)
    meta["thresholds"]["gap_points"] = int(high_score - low_score)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote thresholds to {meta_path}")


if __name__ == "__main__":
    main()
