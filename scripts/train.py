import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pickle
import os

# -----------------------
# SIMPLE PREPROCESSING
# -----------------------
def preprocess(text):
    text = text.lower()                                      # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)      # remove links
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)               # remove special chars
    text = re.sub(r"\s+", " ", text).strip()                 # remove extra spaces
    return text

# -----------------------
# LOAD DATASET
# -----------------------
df = pd.read_csv("data/train.csv")

# handle missing comments safely
df["comment_text"] = df["comment_text"].astype(str).fillna("")

# Apply preprocessing
df["clean"] = df["comment_text"].apply(preprocess)

# -----------------------
# LABEL COLUMNS
# -----------------------
labels = ["toxic", "severe_toxic", "obscene",
          "threat", "insult", "identity_hate"]

X = df["clean"].values
y = df[labels].astype(int).values

# For stratify: at least keep similar ratio of toxic / non-toxic
toxic_any = (y.sum(axis=1) > 0).astype(int)

# -----------------------
# TRAIN / VALIDATION SPLIT
# -----------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42,
    stratify=toxic_any
)

# -----------------------
# TF-IDF VECTORIZATION
#   - unigrams + bigrams
#   - remove very rare words (min_df=5)
#   - limit vocabulary size
# -----------------------
vectorizer = TfidfVectorizer(
    max_features=40000,
    ngram_range=(1, 2),
    min_df=5
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# -----------------------
# LOGISTIC REGRESSION MODEL
# -----------------------
base_clf = LogisticRegression(
    max_iter=500,
    class_weight="balanced",   # important for imbalanced labels
    solver="liblinear",
    C=0.5                      # a bit more regularization
)

clf = OneVsRestClassifier(base_clf, n_jobs=-1)
clf.fit(X_train_vec, y_train)

# -----------------------
# FIND BEST THRESHOLD PER LABEL (USING F1 ON VAL SET)
# -----------------------
probas_val = clf.predict_proba(X_val_vec)  # shape: (n_samples, n_labels)

thresholds = {}
for i, label in enumerate(labels):
    best_thr = 0.5
    best_f1 = 0.0

    # search between 0.2 and 0.8
    for thr in np.linspace(0.2, 0.8, 13):  # 0.2, 0.25, ..., 0.8
        preds_label = (probas_val[:, i] >= thr).astype(int)
        f1 = f1_score(y_val[:, i], preds_label, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    thresholds[label] = float(best_thr)
    print(f"Best threshold for {label}: {best_thr:.2f} (F1 = {best_f1:.3f})")

# Optional: show overall macro F1 with those thresholds
all_preds = np.zeros_like(y_val)
for i, label in enumerate(labels):
    thr = thresholds[label]
    all_preds[:, i] = (probas_val[:, i] >= thr).astype(int)

macro_f1 = f1_score(y_val, all_preds, average="macro", zero_division=0)
print(f"Validation macro F1 with tuned thresholds: {macro_f1:.3f}")

# -----------------------
# SAVE MODEL + VECTORIZER + THRESHOLDS
# -----------------------
os.makedirs("models", exist_ok=True)
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
pickle.dump(clf, open("models/model.pkl", "wb"))
pickle.dump(thresholds, open("models/thresholds.pkl", "wb"))

print("Training completed successfully with tuned thresholds!")
