"""
features.py - feature extraction used inside the final scikit-learn Pipeline.

The pipeline receives RAW message text at prediction time. All link features are
computed here from that raw text with link_tools - nothing outside the pipeline.

LinkFeatureExtractor
--------------------
scikit-learn transformer:
    fit(X, y)   -> no-op (stateless)
    transform(X)-> 2D float array, one row per message, columns:

      has_link        1 if the message contains any link
      n_links         number of links
      is_shortener    1 if the first link uses a URL shortener host
      hyphen_host     1 if the first link's host contains a hyphen
      n_subdomains    number of subdomain labels beyond registrable domain
      digit_in_host   1 if the first link's host contains a digit
      tld_risk        1 if the first link's TLD is on a risky-TLD list

The risky-TLD list covers endings that are cheap, lightly vetted and common in
scam links (xyz, top, click, live, icu, help, vip, ...). It is a heuristic
signal, not a verdict.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from link_tools import split_message

# Cheap, lightly vetted endings that appear disproportionately in scam links.
RISKY_TLDS = {
    "xyz", "top", "click", "live", "icu", "help", "vip", "cc", "gy", "gd",
    "buzz", "loan", "bank", "camp", "dating", "download", "faith", "review",
    "trade", "winner", "work", "press", "host", "store", "shop", "site",
    "online", "win", "club", "party", "gq", "cf", "men", "pw",
}


class LinkFeatureExtractor(BaseEstimator, TransformerMixin):
    """Compute link features from raw message text (stateless)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            info = split_message(str(text))
            first = info["link_info"][0] if info["link_info"] else {}
            rows.append([
                float(len(info["links"]) > 0),          # has_link
                float(len(info["links"])),             # n_links
                float(first.get("is_shortener", False)),
                float(first.get("has_hyphen", False)),
                float(first.get("n_subdomain_labels", 0)),
                float(first.get("has_digit_in_host", False)),
                float(first.get("tld", "").lower() in RISKY_TLDS),
            ])
        return check_array(rows, dtype=float)

    def get_feature_names_out(self, names=None):
        return ["links__has_link", "links__n_links", "links__is_shortener",
                "links__hyphen_host", "links__n_subdomains",
                "links__digit_in_host", "links__tld_risk"]
