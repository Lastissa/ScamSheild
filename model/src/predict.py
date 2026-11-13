"""
predict.py - ScamDetector: load the exported pipeline once, predict from text.

public API
----------
    det = ScamDetector()                          # loads model/artifacts/scam_model.joblib
    det.predict("Urgent! Verify here: bit.ly/xyz")
    # -> {"is_scam": true, "risk_score": 87, "risk_level": "high",
    #     "reasons": ["Contains a shortened link", ...], "links": ["bit.ly/xyz"]}
    det.predict_batch(["msg1", "msg2"])           # list of the same dicts

Input rule (hard rule): ONE input - the message text (a string). Nothing else.
Validation: rejects empty text, non-strings, and text over 2,000 characters.

Reasons are plain-language and come from REAL signals only:
  * link features from link_tools (shortened link, hyphen in host, risky TLD,
    many subdomains);
  * the highest-weighted words in THIS message, computed with the trained
    TF-IDF vector and the logistic-regression coefficients (an interpretation
    model fitted on the same training data). Nothing is made up.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DEFAULT_MODEL = ARTIFACTS_DIR / "scam_model.joblib"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import LinkFeatureExtractor
from link_tools import extract_links

MAX_CHARS = 2000


class ScamDetector:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"model file not found: {self.model_path} "
                f"(run python src/train.py first)"
            )
        self._pipe = joblib.load(self.model_path)
        # thresholds saved by training (probabilities, 0-1)
        meta_path = ARTIFACTS_DIR / "model_meta.json"
        self._meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        thr = self._meta.get("thresholds", {})
        # Integer risk_score bands (0-100). These are the source of truth so
        # is_scam, risk_level and risk_score always stay in sync.
        self._high_score = int(thr.get("high_score_int", thr.get("high_score", 0.5) * 100))
        self._low_score = int(thr.get("low_score_int", thr.get("low_score", 0.2) * 100))

        # ---- interpretation model for per-message word reasons ----
        # Same features as the trained pipeline + an uncalibrated LR, so each
        # word's contribution to the 'scam' score is real (coefficient * tfidf).
        union = FeatureUnion([
            ("text", FeatureUnion([
                ("word", TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), min_df=2,
                    sublinear_tf=True, lowercase=True)),
                ("char", TfidfVectorizer(
                    analyzer="char", ngram_range=(3, 5), min_df=2,
                    sublinear_tf=True, lowercase=True)),
            ])),
            ("links", LinkFeatureExtractor()),
        ])
        self._interp_union = union
        self._interp_clf = LogisticRegression(max_iter=4000)
        train_texts = self._load_train_texts()
        if train_texts:
            X = union.fit_transform(train_texts)
            labels = self._load_train_labels()
            if labels is not None and len(labels) == len(train_texts):
                self._interp_clf.fit(X, labels)
                self._interp_ready = True
                self._feature_names = np.asarray(union.get_feature_names_out())
                self._coef = self._interp_clf.coef_.ravel()
            else:
                self._interp_ready = False
        else:
            self._interp_ready = False

    # ---------------- input validation ----------------
    @staticmethod
    def _validate(text) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("text must not be empty")
        if len(text) > MAX_CHARS:
            raise ValueError(
                f"text is {len(text)} characters; maximum allowed is {MAX_CHARS}"
            )
        return text

    # ---------------- small data loaders for the interpretation model ----
    def _load_train_texts(self):
        p = Path(__file__).resolve().parent.parent / "data" / "processed" / "clean_dataset.csv"
        if not p.exists():
            return None
        import pandas as pd
        df = pd.read_csv(p)
        train = df[df["split"] == "train"] if "split" in df.columns else df
        return train["text"].astype(str).tolist()

    def _load_train_labels(self):
        p = Path(__file__).resolve().parent.parent / "data" / "processed" / "clean_dataset.csv"
        if not p.exists():
            return None
        import pandas as pd
        df = pd.read_csv(p)
        train = df[df["split"] == "train"] if "split" in df.columns else df
        return train["label"].to_numpy()

    # ---------------- public API ----------------
    def _score(self, text: str) -> float:
        return float(self._pipe.predict_proba([text])[0, 1])

    def _level(self, proba: float) -> str:
        score = round(100 * proba)
        if score >= self._high_score:
            return "high"
        if score <= self._low_score:
            return "low"
        return "medium"

    def _reasons(self, text: str, proba: float) -> list[str]:
        reasons: list[str] = []

        # --- real link reasons ---
        try:
            from link_tools import split_message
            info = split_message(text)
        except Exception:
            info = {"text_no_links": text, "links": [], "link_info": []}
        first = info["link_info"][0] if info["link_info"] else {}
        if info["links"]:
            reasons.append(f"Contains {len(info['links'])} link(s)")
            if first.get("is_shortener"):
                reasons.append("Contains a shortened link")
            if first.get("has_hyphen"):
                reasons.append("Link host contains a hyphen")
            if first.get("n_subdomain_labels", 0) > 0:
                reasons.append("Link uses several subdomains")
            tld = str(first.get("tld", "")).lower()
            if tld:
                reasons.append(f"Link ends in a '.{tld}' domain")

        # --- real word reasons from the interpretation model ---
        if self._interp_ready:
            try:
                X = self._interp_union.transform([text])
                contrib = X.multiply(self._coef)   # value * coefficient per feature
                # word features only
                word_mask = np.array(n.startswith("text__word__") for n in self._feature_names)
                idx = np.flatnonzero(word_mask)
                if len(idx):
                    col = contrib[0, idx].toarray().ravel()
                    top = idx[np.argsort(-col)][:3]
                    words = []
                    for i in top:
                        if col[i] > 0:
                            feat = self._feature_names[i]
                            words.append(feat.split("__", 2)[2])
                    if words:
                        reasons.append("Uses scam-like wording such as " +
                                       ", ".join(f"'{w}'" for w in words))
            except Exception:
                pass  # reasons are best-effort; never break a prediction

        if not reasons:
            reasons.append("No strong scam signals detected")
        return reasons

    def predict(self, text) -> dict:
        text = self._validate(text)
        proba = self._score(text)
        score = round(100 * proba)
        level = self._level(proba)
        return {
            "is_scam": level == "high",
            "risk_score": score,
            "risk_level": level,
            "reasons": self._reasons(text, proba),
            "links": extract_links(text),
        }

    def predict_batch(self, texts) -> list[dict]:
        if not isinstance(texts, (list, tuple)):
            raise TypeError("texts must be a list of strings")
        return [self.predict(t) for t in texts]


if __name__ == "__main__":   # quick manual smoke test
    det = ScamDetector()
    for msg in ["Please call me back when you can",
                "URGENT: verify your account now at bit.ly/3xYz9Q",
                "Abeg send me N5000 for transport, my phone dey off"]:
        print(json.dumps(det.predict(msg), ensure_ascii=False))
