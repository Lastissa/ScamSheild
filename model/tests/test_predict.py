"""
tests/test_predict.py - tests for src/predict.py (ScamDetector).

Requires artifacts/scam_model.joblib (run python src/train.py first).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from predict import ScamDetector  # noqa: E402

MODEL = Path(__file__).resolve().parent.parent / "artifacts" / "scam_model.joblib"


@pytest.fixture(scope="module")
def det():
    if not MODEL.exists():
        pytest.skip("scam_model.joblib not found - run python src/train.py first")
    return ScamDetector()


def test_valid_output_structure(det):
    out = det.predict("Please call me back when you can")
    assert set(out) == {"is_scam", "risk_score", "risk_level", "reasons", "links"}
    assert isinstance(out["is_scam"], bool)
    assert isinstance(out["risk_score"], int)
    assert 0 <= out["risk_score"] <= 100
    assert out["risk_level"] in ("low", "medium", "high")
    assert isinstance(out["reasons"], list)
    assert all(isinstance(r, str) for r in out["reasons"])
    assert isinstance(out["links"], list)


def test_empty_text_rejected(det):
    with pytest.raises(ValueError):
        det.predict("")
    with pytest.raises(ValueError):
        det.predict("   ")


def test_non_string_rejected(det):
    with pytest.raises(TypeError):
        det.predict(None)
    with pytest.raises(TypeError):
        det.predict(123)


def test_too_long_rejected(det):
    with pytest.raises(ValueError):
        det.predict("x" * 2001)


def test_obvious_scam_ranks_higher_than_obvious_legit(det):
    scam = det.predict(
        "URGENT: verify your GTBank account now: bit.ly/3xYz9Q or lose access")
    legit = det.predict("Hi Mom, I will be home by 6pm, do not worry")
    assert scam["risk_score"] >= legit["risk_score"]


def test_message_with_link_records_links(det):
    out = det.predict("Track your parcel here: www.jumia.com.ng/track")
    assert "www.jumia.com.ng/track" in out["links"] or any(
        "jumia" in l for l in out["links"])


def test_pidgin_message(det):
    out = det.predict("Abeg, wetin happen? my phone no dey work, send me money")
    assert out["risk_level"] in ("low", "medium", "high")
    assert isinstance(out["reasons"], list)


def test_predict_batch(det):
    outs = det.predict_batch(["Hi there", "URGENT verify now bit.ly/abc"])
    assert len(outs) == 2
    assert set(outs[0]) == {"is_scam", "risk_score", "risk_level", "reasons", "links"}


def test_fresh_process_text_only():
    """
    MODEL INPUT RULE: load scam_model.joblib in a FRESH PROCESS and predict
    from a plain string with no other inputs.
    """
    code = (
        "import joblib, sys\n"
        "m = joblib.load(r'" + str(MODEL) + "')\n"
        "p = m.predict_proba(['URGENT verify your account at bit.ly/xyz now'])[0, 1]\n"
        "print(round(float(p), 4))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
        env={**__import__("os").environ, "PYTHONPATH": str(SRC)},
    )
    assert result.returncode == 0, result.stderr
    prob = float(result.stdout.strip())
    assert 0.0 <= prob <= 1.0


def test_links_module_basic():
    from link_tools import extract_links, strip_links, parse_link
    assert extract_links("see bit.ly/abc now") == ["bit.ly/abc"]
    assert strip_links("see bit.ly/abc now") == "see [URL] now"
    info = parse_link("bit.ly/abc")
    assert info["is_shortener"] is True
    info2 = parse_link("https://gtbank-secure.xyz/login")
    assert info2["has_hyphen"] is True
