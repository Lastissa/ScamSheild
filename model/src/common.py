"""
common.py - shared helpers for the ScamShield scam-message detector.

Everything that more than one script needs lives here, so all steps behave the
same way. Rule 1 of the project says: nothing outside the model/ folder, and
never modify the original data files. This module only READS the original data;
cleaned copies are written to model/data/processed/.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fixed project-wide random seed (hard rule 5: use fixed random seeds).
# Every script imports this instead of making up its own number.
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Folder layout. BASE_DIR is the model/ folder (the parent of src/).
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent          # .../model/src
BASE_DIR = SRC_DIR.parent                          # .../model
DATA_DIR = BASE_DIR / "data"                       # original data (read-only!)
PROCESSED_DIR = DATA_DIR / "processed"             # cleaned copies go here
ARTIFACTS_DIR = BASE_DIR / "artifacts"             # saved model + model card
DOCS_DIR = BASE_DIR / "docs"                       # all documentation

MAIN_DATA = DATA_DIR / "scam_dataset_text_label.csv"
REAL_TEST = DATA_DIR / "real_messages_test_text_label.csv"
REAL_LEGIT = DATA_DIR / "real_legit_messages.csv"   # OPTIONAL - may not exist

# Make sure the processed folder exists (creating folders is allowed; the
# original files themselves are never touched).
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    """
    Read one of the project CSV files into a DataFrame.

    The data contains French/Spanish text stored with Latin-1 style accents
    (e.g. 'telephone' with a raw 0xE9 byte), which is NOT valid UTF-8. So we
    first try UTF-8 (this also removes the Excel BOM if present) and fall back
    to Latin-1, which maps every byte and keeps accented letters readable.

    We never write back to these files (hard rule 1).
    """
    path = Path(path)
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        # Latin-1 never raises a decode error; accents come out correct.
        return pd.read_csv(path, encoding="latin-1")


# ---------------------------------------------------------------------------
# Text masking (used for near-duplicate GROUPING only, never for training text)
# ---------------------------------------------------------------------------
# Email pattern (emails are not links, so link_tools.strip_links leaves them).
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.I)

# Money amounts: a currency marker glued to a number.
#   $900   N100   NGN 5,000   CFA150   £12.50   #5,000
_MONEY_RE = re.compile(
    r"(?:[$€£₦#]|(?:NGN|CFA|USD|EUR|GBP|NAIRA)\s?)\s?\d[\d,.]*", re.I
)

# Any remaining standalone number (amounts, dates, account numbers, OTPs...).
_NUMBER_RE = re.compile(r"\d[\d,.]*")


def mask_text(text: str) -> str:
    """
    Return a copy of the text with the volatile parts masked.

    Used ONLY to detect near-duplicate messages (Step 1b). Two messages that
    differ just by the link, the amount or the account number are probably the
    same template, so those parts become fixed tokens:

        links  -> [URL]      (via link_tools.strip_links)
        emails -> [EMAIL]
        money  -> [MONEY]    (currency symbol glued to a number)
        others -> [NUM]

    The TRAINING text is never masked - the model sees the raw message.
    """
    from link_tools import strip_links  # local import: link_tools sits next to us

    out = strip_links(str(text), placeholder="[URL]")
    out = _EMAIL_RE.sub(" [EMAIL] ", out)
    out = _MONEY_RE.sub(" [MONEY] ", out)
    out = _NUMBER_RE.sub(" [NUM] ", out)
    return re.sub(r"\s+", " ", out).strip().lower()


# ---------------------------------------------------------------------------
# Approximate language label (Step 1f) - a heuristic, not a language detector
# ---------------------------------------------------------------------------
# Strong Pidgin markers: a single hit is enough evidence of Pidgin.
_PIDGIN_STRONG = {
    "abeg", "abeg-o", "wetin", "wahala", "oga", "sabi", "una", "abi", "shey",
    "comot", "waka", "chop", "pikin", "oga-at-the-top", "yawa", "japa",
    "no-vex", "gbege", "kolo", "gists", "gist", "dey", "nko", "sha",
}
# Weak Pidgin markers: common in Pidgin but also in normal English, so they
# only count when they appear TOGETHER with another marker.
_PIDGIN_WEAK = {"na", "make", "them", "am", "oo", "o", "you-do"}

# Clear French / Spanish / Portuguese words (only obvious ones, on purpose).
_ROMANCE_MARKERS = {
    # French
    "bonjour", "salut", "merci", "voici", "bientot", "bientôt", "fils", "besoin",
    "urgence", "envoyer", "nouveau", "numéro", "numero", "mon", "ton", "notre",
    "je", "tu", "vous", "nous", "être", "etre", "madame", "monsieur",
    # Spanish
    "hola", "gracias", "por", "favor", "necesito", "dinero", "cuenta", "usted",
    "estimado", "señor", "senor",
    # Portuguese
    "obrigado", "voce", "você", "preciso", "dinheiro", "conta",
}

# Any character outside the basic Latin / Latin-1 supplement area suggests a
# different script (Arabic, Cyrillic, Chinese, ...).
_NON_LATIN_RE = re.compile(r"[^\u0000-\u024F]")


def language_label(text: str) -> str:
    """
    Rough language bucket for reporting: 'pidgin', 'other' or 'english'.

    This is APPROXIMATE (say so in every doc): it is a word list, not a real
    language detector. Order matters: Pidgin is English-based, so we check it
    before calling something 'english'.
    """
    t = str(text).lower()
    words = set(re.findall(r"[a-z']+", t))

    if _NON_LATIN_RE.search(str(text)):
        return "other"                      # a different writing system
    strong = len(words & _PIDGIN_STRONG)
    weak = len(words & _PIDGIN_WEAK)
    if strong >= 1 or weak >= 2:            # one strong marker, or two weak ones
        return "pidgin"
    if words & _ROMANCE_MARKERS:
        return "other"                      # clear French/Spanish/Portuguese
    return "english"


# ---------------------------------------------------------------------------
# Model building + evaluation helpers (shared by Steps 2, 3, 4 and 5)
# ---------------------------------------------------------------------------
def build_baseline_pipeline():
    """
    The Step-2 BASELINE model: word TF-IDF (1-2 grams) + character TF-IDF
    (3-5 grams, lowercase) + logistic regression. No link features.

    Defined here so Step 3 (link dependency test) uses EXACTLY the same
    model spec as Step 2, just re-fitted on different text.
    """
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    union = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
            lowercase=True)),
        ("char", TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
            lowercase=True)),
    ])
    return Pipeline([
        ("features", union),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    """
    All metrics the project reports, from real predictions only.

    proba  = predicted probability of 'scam' (class 1).
    Works even when the data has only one class (e.g. the 94 real scam
    messages): ROC-AUC and PR-AUC are then reported as None, and we return
    recall so the spec's 'recall only' rule is followed.
    """
    import sklearn.metrics as m

    y_true = np.asarray(list(y_true), dtype=int)
    pred = (np.asarray(proba) >= threshold).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    out = {
        "threshold": threshold,
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "f1": (lambda p, r: round(2 * p * r / (p + r), 4) if p and r else 0.0)(
            tp / (tp + fp) if (tp + fp) else 0, tp / (tp + fn) if (tp + fn) else 0),
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,
        "brier": round(float(np.mean((np.asarray(proba) - y_true) ** 2)), 4),
    }
    both_classes = len(set(y_true.tolist())) == 2
    if both_classes:
        out["roc_auc"] = round(float(m.roc_auc_score(y_true, proba)), 4)
        out["pr_auc"] = round(float(m.average_precision_score(y_true, proba)), 4)
    else:
        out["roc_auc"] = None   # undefined with a single class - do not invent
        out["pr_auc"] = None
    return out


# ---------------------------------------------------------------------------
# Small helpers used by several steps
# ---------------------------------------------------------------------------
def prf(tp: int, fp: int, fn: int) -> dict:
    """Precision / recall / F1 from confusion counts (no invented numbers)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)}
