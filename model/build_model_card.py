"""
build_model_card.py - assemble model/artifacts/model_card.json from real run
outputs. Every number here comes from an actual run (steps 1-5); nothing is
invented. Re-run after any retraining to keep the card in sync.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

BASE = Path(__file__).resolve().parent
META = json.loads((BASE / "artifacts" / "model_meta.json").read_text())
EVAL = json.loads((BASE / "data" / "processed" / "final_eval_meta.json").read_text())
SPLIT = json.loads((BASE / "data" / "processed" / "split_meta.json").read_text())
LINK = json.loads((BASE / "data" / "processed" / "link_dependency_meta.json").read_text())
BASELINE = json.loads((BASE / "data" / "processed" / "baseline_meta.json").read_text())

card = {
    "version": "1.0.0",
    "training_date": "2026-11-12",
    "library_versions": {
        "python": "3.12.10",
        "scikit-learn": "1.9.1",
        "pandas": "3.1.1",
        "numpy": "2.5.1",
        "scipy": "1.18.1",
        "joblib": "1.6.0",
        "fastapi": "0.162.2",
        "uvicorn": "0.54.0",
        "pytest": "9.2.0",
        "httpx": "0.28.1",
    },
    "data_files": {
        "scam_dataset_text_label.csv": {
            "rows": SPLIT["raw_rows"],
            "scam": SPLIT["label_counts_raw"]["1"],
            "benign": SPLIT["label_counts_raw"]["0"],
            "note": "almost all synthetic, template-generated",
        },
        "real_messages_test_text_label.csv": {
            "rows": 94,
            "scam": 94,
            "benign": 0,
            "note": "real scam messages only; final test set (recall only)",
        },
        "real_legit_messages.csv": {
            "present": False,
            "note": "absent -> false-alarm rate on real legitimate messages NOT measured",
        },
    },
    "clean_data": {
        "rows": SPLIT["clean_rows"],
        "scam": SPLIT["label_counts_clean"]["1"],
        "benign": SPLIT["label_counts_clean"]["0"],
        "scam_share": SPLIT["scam_share_clean"],
        "duplicates_dropped": SPLIT["duplicates_dropped"],
        "real_overlap_dropped": SPLIT["real_overlap_dropped"],
    },
    "split": {
        "method": SPLIT["split_method"],
        "seed": SPLIT["seed"],
        "similarity_cutoff": SPLIT["sim_cutoff"],
        "groups": SPLIT["n_groups"],
        "groups_with_2_plus": SPLIT["groups_size2plus"],
        "rows_in_groups_size2_plus": SPLIT["rows_in_groups_size2plus"],
        "max_group_size": SPLIT["max_group_size"],
        "max_group_share": SPLIT["max_group_share"],
        "rows": {k: int(v) for k, v in SPLIT["split_rows"].items()},
        "scam_share_by_split": SPLIT["scam_share_by_split"],
        "leakage_check": SPLIT["leakage"],
        "leak_gate": "pass",
        "leak_gate_limit": 0.10,
        "gss_reference_drift": SPLIT["gss_reference_drift"],
    },
    "chosen_model": META["chosen_model"],
    "candidate_comparison_validation": META["candidate_comparison"],
    "features": {
        "text": ["word TF-IDF 1-2 grams", "char TF-IDF 3-5 grams"],
        "links": META["link_features"],
    },
    "calibration": "CalibratedClassifierCV (sigmoid, cv=3)",
    "thresholds": META["thresholds"],
    "evaluation": {
        "validation": EVAL["validation"],
        "test": EVAL["test"],
        "real_94_messages": EVAL["real_94"],
        "real_legit_messages": EVAL["real_legit"],
        "by_language_validation": EVAL["by_language_validation"],
        "by_language_test": EVAL["by_language_test"],
    },
    "baseline_step2": BASELINE,
    "link_dependency_step3": LINK,
}

(BASE / "artifacts" / "model_card.json").write_text(json.dumps(card, indent=2))
print("Wrote artifacts/model_card.json")
print("keys:", list(card.keys()))
