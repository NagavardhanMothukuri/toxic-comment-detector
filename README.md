#  Toxic Comment Detector

A hybrid NLP system that detects toxic comments across six labels:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

##  Features

- Trained on Jigsaw Toxic Comment dataset
- TF-IDF + Logistic Regression (multi-label)
- Tuned thresholds per label (F1-based)
- Extra rule-based layer:
  - sexual content detection
  - threat & violence promotion
  - identity-based hate
  - positive compliment override (e.g. "you are beautiful" → clean)
- Flask backend + modern web UI
- Works fully offline on local machine

##  Tech stack

- Python
- scikit-learn
- pandas
- Flask
- HTML/CSS/JavaScript (vanilla)

##  Run locally

```bash
pip install -r requirements.txt
python run_app.py
# or: python app.py
