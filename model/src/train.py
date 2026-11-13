"""
train.py - one command that reproduces everything from data to saved model.

Run from the model/ folder:
    python src/train.py

Runs, in order:
  1. src/step1_data_check.py   - data check + leakage-safe grouped split
  2. src/step2_baseline.py     - baseline results (documentation)
  3. src/step3_link_dependency.py - link dependency test (documentation)
  4. src/step4_train_model.py  - candidates, calibration, thresholds -> scam_model.joblib
  5. src/step5_final_evaluation.py - final evaluation -> final_evaluation.md

Everything stays inside model/; original data files are never modified.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

STEPS = [
    "step1_data_check",
    "step2_baseline",
    "step3_link_dependency",
    "step4_train_model",
    "step5_final_evaluation",
]


def main() -> None:
    print("=== ScamShield: full training run ===\n")
    for name in STEPS:
        print(f"--- running {name}.py ---")
        runpy.run_path(str(SRC / f"{name}.py"), run_name="__main__")
        print(f"--- {name}.py done ---\n")
    print("=== ALL STEPS DONE ===")
    print(f"Model:     {SRC.parent / 'artifacts' / 'scam_model.joblib'}")
    print(f"Meta:      {SRC.parent / 'artifacts' / 'model_meta.json'}")
    print(f"Docs:      {SRC.parent / 'docs'}")


if __name__ == "__main__":
    main()
