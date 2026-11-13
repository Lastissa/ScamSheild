# ScamShield - Nigerian scam-message detector

This tool reads the text of a message (English or Nigerian Pidgin SMS/chat) and
tells you whether it looks like a scam, with a risk score from 0 to 100, a risk
level (low / medium / high), and short plain-language reasons. It is meant to
run on a website backend: send it a message, get back a JSON verdict. Nothing
is stored or logged at prediction time.

## What it does (in three sentences)

You give it one message text. It returns whether the message looks like a scam,
a risk score from 0 to 100, a risk level, and short reasons written in plain
language (for example: "Contains a shortened link"). It is a risk score, not a
proof: the model can be wrong, and must never be the only reason to block a
person or delete a message.

## How to try it (exact commands, from a fresh computer)

Open a terminal in the `model/` folder and run:

```bash
# 1. Install the pinned libraries (Python 3.8+ needed)
pip install -r requirements.txt

# 2. Train/export the model from the data (takes a few minutes)
python src/train.py

# 3. Start the backend
uvicorn app.main:app
```

The API is now live at http://127.0.0.1:8000. In another terminal, send a test
message:

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"URGENT: verify your account now at bit.ly/3xYz9Q\"}"
```

(On PowerShell, replace `^` line continuations with backticks or run it as one
line.) You should get back JSON like:

```json
{
  "is_scam": true,
  "risk_score": 100,
  "risk_level": "high",
  "reasons": ["Contains 1 link(s)", "Contains a shortened link", "Link ends in a '.ly' domain"],
  "links": ["bit.ly/3xYz9Q"]
}
```

There is also a `GET /health` endpoint and a `POST /predict-batch` endpoint
that accepts `{"texts": ["msg1", "msg2"]}`. To change which websites may call
the API, edit the `ALLOWED_ORIGINS` list (or the `CORS_ALLOWED_ORIGINS`
environment variable) at the top of `app/main.py`.

Or use it directly in Python:

```python
from src.predict import ScamDetector
det = ScamDetector()
det.predict("Hi Mom, I will be home by 6pm")
```

## Risk score and risk levels

| risk level | what it means | rule |
|---|---|---|
| high | looks like a scam; flag it | score >= 33 |
| low | looks benign | score <= 33 |
| medium | in between | everything else |

The score is `round(100 × the model's calibrated probability)`. It is a **risk
score**, not the real-world chance that the message is a scam: the training
data has far more scams than real life, so the numbers are calibrated to the
training data, not to your inbox.

On the data used to choose them (the validation split), "high" reached a
precision of 0.9516 (target: at least 0.95) and at "low" the model still kept
above 99% of scams above the cut-off. On this data the two cut-offs nearly
meet, so most messages land in low or high.

## Six real example outputs

These are the actual outputs of the exported model (`scam_model.joblib`):

| # | message | risk score | level | verdict |
|---|---|---|---|---|
| 1 | Hi Mom, I will be home by 6pm, do not worry | 57 | high | **wrong** - false alarm |
| 2 | URGENT: verify your GTBank account now at bit.ly/3xYz9Q or lose access | 100 | high | correct |
| 3 | Abeg, my phone no dey work, send me N5000 for transport | 91 | high | correct |
| 4 | Your MTN data bundle is active until Friday. No action needed. | 32 | low | correct |
| 5 | CONGRATULATIONS! You won $10,000 in the promo. Claim now: http://claim-prize.xyz/win | 100 | high | correct |
| 6 | Salut, c'est ton fils, j'ai perdu mon telephone. J'ai besoin de CFA150 de toute urgence. | 92 | high | correct |

Example 1 is a real mistake: the model scored a harmless message 57/high. This
is exactly why the score must not be used alone. Reasons come from real signals
(link shorteners, hyphens in hosts, risky domain endings, scam-like wording)
and are never made up.

## Does it rely on the links? (link dependency test)

We ran the same model three ways: scored on normal text (A), scored on text
with links removed (B), and re-trained on text with links removed (C). The
results barely moved:

| | test precision | test recall | recall on 94 real scams |
|---|---|---|---|
| A - normal text | 1.0000 | 0.9891 | 0.9362 |
| B - links removed at test time | 1.0000 | 0.9852 | 0.9362 |
| C - re-trained without links | 1.0000 | 0.9622 | 0.9255 |

Plain conclusion: **the model does not collapse without links.** It learns from
the wording, not only from the links. Scam links really do look different from
legitimate ones, so links help a little - but they are not the whole story.

## What it cannot do

- It cannot prove a message is a scam. It can be wrong both ways (false alarms
  and missed scams).
- It must not be the only basis for blocking a person or deleting a message.
- It works best on Nigerian-style English and Pidgin SMS/chat messages; other
  languages or styles are less reliable.
- Almost all training data (about 29,650 messages) is synthetic, generated
  from templates, so very high scores on similar synthetic messages can just
  mean template memorisation. Real-message recall was 0.9468 at the operating
  threshold on 94 real scams.
- No real legitimate messages were available for testing, so the false-alarm
  rate on real benign messages was **not measured**.

## Limitations and responsible use

The model can be wrong, and must not be the only basis for blocking a person or
deleting a message. It works best on Nigerian-style English and Pidgin SMS and
chat messages. A human should review anything important.

What would improve it most: real legitimate messages with links (to measure and
lower false alarms on real traffic), and more real scam messages (especially
ones that do not follow the templates in the synthetic data).

## Who to ask for help

See `docs/MODEL_DOCUMENTATION.md` for how the model was built and how to retrain
it safely. Questions about this model: contact the team that deployed it in your
organisation - there is no public vendor behind this research prototype.
