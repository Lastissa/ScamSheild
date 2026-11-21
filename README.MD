# ScamShield

A scam message detection project for Nigeria, currently in development.

Built by Team Syntropy, University of Ilorin.

> **Project status: Early prototype.** ScamShield is not deployed and has not been tested on real-world traffic. We make no claims about its real-world accuracy. The prototype currently raises false alarms on some ordinary messages. See [Known Limitations](#known-limitations).

## Table of Contents

- [What ScamShield aims to do](#what-scamshield-aims-to-do)
- [What exists today](#what-exists-today)
- [The prototype model](#the-prototype-model)
- [Data](#data)
- [Survey](#survey)
- [Results so far](#results-so-far)
- [Known limitations](#known-limitations)
- [Open issues](#open-issues)
- [Run the prototype](#run-the-prototype)
- [Responsible use, privacy and security](#responsible-use-privacy-and-security)
- [Team](#team)
- [Project information](#project-information)
- [Disclaimer](#disclaimer)

## What ScamShield aims to do

ScamShield aims to help people assess suspicious messages before acting on them.

A user submits the text of a suspicious message and receives a risk score, a risk level and short explanations of the result.

The long-term goal is to develop a consumer web or mobile application and an API that authorised institutions, such as banks and fintech companies, can integrate into their systems.

Neither the consumer application nor the institutional API has been built yet.

## What exists today

| Component | Current state | Location |
| --- | --- | --- |
| Message detector (trained model, training code, data and documentation) | Working prototype, with limited real-world performance | `model` branch, `model/` folder |
| Prediction service (FastAPI): `POST /predict`, `POST /predict-batch`, `GET /health` | Working prototype, without authentication | `model` branch, `model/app/` |
| Automated tests | 16 tests, all passing when last run on 21 November 2026 | `model` branch, `model/tests/` |
| Backend (Django REST Framework) | Skeleton only, with one welcome endpoint and no model integration | `backend` branch (default) |
| AWS architecture plan | Planning document only; nothing deployed | `cloud` branch, `DOCS/ARCHITECTURE.MD` |
| Consumer web or mobile application | Not built | None |
| Institutional API with authentication keys and rate limits | Not built | None |
| Phone number reputation checks | Not built | None |
| Real-world validation or pilot | Not done | None |

## The prototype model

- **Input:** The text of one message, up to 2,000 characters.
- **Output:** A risk score from 0 to 100, a risk level, short explanations and any links found in the message.
- **Method:** A calibrated Linear Support Vector Machine (Linear SVM) using word and character TF-IDF features, together with seven link-related features.
- **Link features:** Whether a message contains a link, number of links, use of a URL shortener, hyphens in the hostname, number of subdomains, digits in the hostname and risky domain endings.
- **Model selection:** The Linear SVM was selected over logistic regression and Naive Bayes using validation data that is mostly synthetic.
- **Risk levels:**
  - Low: 33 or below.
  - Medium: 34 to 59.
  - High: 60 or above.

The risk thresholds were selected using synthetic validation data. They have not been tuned for real-world use.

> **Important:** The risk score is not the probability that a message is a scam. It is a model-generated risk score. The training data contains far more scam messages than would normally occur in everyday communication.

Links are analysed based on how their addresses are written. ScamShield does not open submitted links.

## Data

| Dataset | Size | Description |
| --- | --- | --- |
| Training and evaluation set (`model/data/scam_dataset_text_label.csv`) | 29,650 messages: 20,260 scam and 9,390 legitimate | Almost entirely synthetic, generated from templates with some augmented variations. The data is split 70/15/15 by groups of near-duplicate messages to reduce data leakage between training and evaluation sets. |
| Online-sourced scam examples (`model/data/real_messages_test_text_label.csv`) | 94 messages, all labelled as scams | Advance-fee-style messages collected online. Used only for final testing, not training. The messages have not all been independently verified as actual scam encounters. |
| Real legitimate messages | None | No real legitimate message dataset is currently available. |

Nigerian Pidgin represents a small proportion of the dataset. About 1% of the rows carry a Pidgin language label, while a rough word-list check finds under 3%.

Brand names and amounts in the synthetic data follow Nigerian patterns, but most of the messages are written in English.

## Survey

In June 2026, we surveyed 19 respondents at the University of Ilorin.

The sample is small and should not be treated as representative of Nigerians or all university students.

| Finding | Share | Respondents |
| --- | --- | --- |
| Had received suspicious messages | 89% | 17 of 19 |
| Personally lost money to an online scam | 37% | 7 of 19 |
| Knew someone who had been scammed | 68% | 13 of 19 |
| Had clicked a suspicious link at least once | 58% | 11 of 19 |

## Results so far

All figures below are taken from `model/docs/final_evaluation.md`.

| Test | Result | Interpretation |
| --- | --- | --- |
| Synthetic held-out test set (4,448 messages) | Precision: 0.9993, Recall: 0.9579, False Positive Rate: 0.0014 | Shows how the model performs on generated messages similar to those used during training. It does not establish real-world accuracy. |
| 94 online-sourced scam examples | 58 of 94 flagged as high risk, giving a recall of 0.617 | Measures detection on a narrow collection of advance-fee-style messages. It does not represent all types of Nigerian scams. |
| Real legitimate messages | Not measured | No real legitimate test dataset was available, so the real-world false alarm rate is unknown. |

The synthetic test results are considerably stronger than the results on the online-sourced scam examples.

This difference is important. Strong performance on synthetic data does not mean the model will perform equally well on messages received by real users.

The model still requires broader testing with genuine scam and legitimate messages.

## Known limitations

- **False alarms on everyday messages:** The model has not been tested on a proper dataset of real legitimate messages. In a small informal check involving 35 ordinary messages written by hand, 7 were scored as high risk. These included a business invoice note and a message from a parent. For example, the message "Hi Mom, I will be home by 6pm" receives a score of 57, placing it in the medium-risk category. This informal check is a warning sign, not a statistically representative false-positive rate.
- **Narrow scam coverage:** The 94 online-sourced scam examples are all advance-fee-style messages. They do not cover bank alert phishing, delivery scams or the wider range of SMS and WhatsApp scams.
- **Mostly synthetic training data:** Very high scores on messages resembling the generated templates may indicate that the model has learned patterns specific to those templates.
- **Limited Pidgin coverage:** Nigerian Pidgin makes up only a small proportion of the available data.
- **Simple link analysis:** Link features are based on the written structure of an address and a short list of risky domain endings. There is no external domain reputation lookup.
- **No phone number checks:** Phone number reputation checks are planned but have not been implemented.
- **Not a decision maker:** A low score does not prove that a message is safe, and a high score does not prove that it is a scam. The model must never be the sole basis for blocking a person or deleting a message.

## Open issues

- Collect more genuine scam and legitimate messages, including SMS and WhatsApp examples, with appropriate consent, anonymisation and source records.
- Measure the false alarm rate using real legitimate messages.
- Include more everyday personal and business messages in the legitimate training data.
- Confirm the licence of `model/data/internet_scams_archive.csv`, currently marked as "CC-BY-4.0 (proposed)".
- Check generated messages for phone numbers and email addresses that could belong to real people.
- Connect the model to the backend.
- Add authentication, rate limiting and an appropriate logging policy.
- Complete privacy, security and usability checks before any live pilot.

## Run the prototype

The model prototype is available on the `model` branch.

### Requirements

- Git
- Python 3.13
- Internet connection for installing dependencies

### 1. Clone the repository

```bash
git clone https://github.com/Lastissa/ScamSheild.git
cd ScamSheild
```

### 2. Switch to the model branch

```bash
git checkout model
```

### 3. Enter the model folder

```bash
cd model
```

### 4. Create a virtual environment (recommended)

On Windows:

```powershell
py -m venv .venv
.venv\Scripts\activate
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

A trained model file is already included in `model/artifacts/`, so you do not need to train the model before testing it.

To retrain the model using the available data, run:

```bash
python src/train.py
```

### 6. Start the prediction service

```bash
uvicorn app.main:app
```

By default, the service runs at:

```text
http://127.0.0.1:8000
```

### 7. Test a message

Open a second terminal, enter the repository's `model` folder and send a test request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "URGENT: verify your account now at bit.ly/3xYz9Q"}'
```

The response is JSON containing:

- `is_scam`
- `risk_score`
- `risk_level`
- `reasons`
- `links`

### 8. Run the automated tests

From the `model` folder:

```bash
pytest tests
```

The prototype was last reviewed using Python 3.13.

## Responsible use, privacy and security

ScamShield provides decision support, not guarantees.

Submitted messages may contain personal information. Before any live pilot, the team must define consent procedures, data minimisation, data retention and deletion, access controls and security testing.

Public repositories must not contain private user submissions, passwords, API keys, credentials or other sensitive information.

Feedback must be reviewed before it is used to update training data. A single user report must not automatically label a phone number, person or organisation as fraudulent.

The model file (`scam_model.joblib`) is a pickle file. Load model files only from sources you trust, as loading an untrusted pickle file can execute malicious code.

## Team

ScamShield is being developed by Team Syntropy, a multidisciplinary student team at the University of Ilorin.

| Team member | Role | Responsibility |
| --- | --- | --- |
| Habeeb Ayokunmi Babatunde | Product Lead | Product strategy and team coordination. |
| Zainab Eniola Sayi | Technical Lead and Cybersecurity Analyst | System architecture and security. |
| Abdulsalam Opeyemi Issa | Backend Engineer | Building, managing and securing communication between APIs. |
| Aminat Anjolaoluwa Morakinyo | Cloud Infrastructure Specialist | AWS deployment, performance monitoring and server scaling. |
| Zakeeyah Temitope Abdulfatah | Data Annotation Specialist | Labelling, classifying and reviewing collected data for model training. |
| Oluwaseun Richard Edem | Data Researcher and Assistant ML Specialist | Data research and support for machine learning development. |
| Success Adinoyi Umazayi | Machine Learning Specialist | Developing, training, testing and optimising the scam detection models. |

## Project information

- **Project:** ScamShield
- **Team:** Team Syntropy
- **Institution:** University of Ilorin, Nigeria
- **Repository:** [GitHub - ScamShield](https://github.com/Lastissa/ScamSheild)

## Disclaimer

ScamShield is an evolving project and is not a substitute for independent verification, bank security processes or professional advice.

Users should verify suspicious requests directly through trusted official channels. They should never share passwords, PINs or one-time passwords in response to suspicious messages.

The project does not claim to detect every scam or guarantee that a message assessed as low risk is safe.
