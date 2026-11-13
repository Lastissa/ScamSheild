# Glossary - key terms in plain words

This glossary is for non-technical readers. Every term is explained in one
short sentence, with no jargon where we can avoid it.

**Accuracy** - the share of all messages the model classifies correctly. Not
used as the headline measure here because the data contains far more scams
than benign messages.

**Ablation (link dependency test)** - removing something (here, the links)
from the data to check whether the model's performance collapses. It tells us
whether the model leans on links or on wording.

**Benign** - a legitimate, non-scam message (label 0 in the data).

**Brier score** - a measure of how accurate predicted probabilities are, from
0 (perfect) to 1 (worst). It penalises confident wrong predictions heavily.

**Calibration** - adjusting a model's raw scores so they behave like
probabilities. A calibrated score of 80 means "about 8 out of 10 similar
messages scored 80 turned out to be scams" - in the training data, not real
life.

**Character TF-IDF** - a way of turning text into numbers by counting short
character sequences (here 3-5 letters), which catches spellings and endings
like ".xyz".

**Data leakage** - when information from the test set sneaks into training, so
the model is effectively tested on answers it already saw. Scores then look
better than reality.

**False alarm (false positive)** - a benign message wrongly flagged as a scam.

**False-positive rate** - the share of benign messages wrongly flagged.

**F1 score** - the average of precision and recall; a single number between 0
and 1, higher is better.

**Grouped split** - dividing the data so that near-identical messages always
stay together in either training or testing, never both. Prevents inflated
scores.

**Link features** - facts about links inside a message: whether it is a short
link (bit.ly etc.), whether the host contains a hyphen or digits, how many
subdomains it has, whether its ending (TLD) is risky.

**Logistic regression** - a simple model that combines evidence from words and
link features into a scam score.

**Low / medium / high** - risk levels: 'high' means the message looks like a
scam (precision target 95% at this level), 'low' means very few scams are
missed there, 'medium' is in between.

**N-gram** - a short sequence of n words (word n-gram) or letters (character
n-gram) used as a text feature.

**Precision** - of the messages the model flags as scams, the share that are
really scams. High precision = few false alarms.

**PR-AUC** - a single number summarising precision across all possible risk
thresholds; higher is better, especially when scams outnumber benign messages.

**Recall** - of all real scam messages, the share the model finds. High
recall = few scams missed.

**Risk score** - a number from 0 to 100: round(100 × calibrated probability).
It is a *risk score*, not the real-world chance a message is a scam, because
the training data has far more scams than real life.

**ROC-AUC** - a single number summarising how well the model ranks scams above
benign messages across all thresholds; 1.0 is perfect, 0.5 is random guessing.

**Scam** - a message trying to trick the recipient (label 1 in the data).

**Synthetic data** - messages generated from templates rather than written by
real people. Almost all of this project's training data is synthetic.

**TF-IDF** - "term frequency × inverse document frequency": a way of scoring
words by how common they are in a message but rare across all messages.

**Validation set** - messages held back during training to choose the model
and thresholds, before the final test set is used.
