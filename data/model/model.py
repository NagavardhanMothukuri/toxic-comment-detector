import pandas as pd
import time
import argparse
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/train.csv"
MODEL_PATH = "model/model.pkl"

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["comment_text"] = df["comment_text"].fillna("")
    df["label"] = df[["toxic", "severe_toxic", "obscene", "insult", "identity_hate", "threat"]].max(axis=1)
    return df["comment_text"], df["label"]

def train_once():
    print("\n🔄 Training model...")

    X, y = load_data()

    vectorizer = TfidfVectorizer(max_features=50000)
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=200)
    model.fit(X_vec, y)

    joblib.dump((vectorizer, model), MODEL_PATH)
    print("✅ Model saved at:", MODEL_PATH)

def train_forever():
    print("\n♾️ Starting infinite training loop...")
    while True:
        train_once()
        print("⏳ Sleeping 12 hours before next training...\n")
        time.sleep(12 * 60 * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--forever", action="store_true", help="Train forever automatically")
    args = parser.parse_args()

    if args.forever:
        train_forever()
    else:
        train_once()
