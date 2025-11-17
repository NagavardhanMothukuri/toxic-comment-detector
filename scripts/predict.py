import pickle
import re

# -----------------------
# IMPORT perform_action CORRECTLY
# -----------------------
# Works both when:
#  - running "python scripts/predict.py"
#  - importing "from scripts.predict import predict_text"
try:
    from .actions import perform_action
except ImportError:
    from actions import perform_action


# -----------------------
# SIMPLE PREPROCESS
# -----------------------
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------
# LOAD MODELS + THRESHOLDS
# -----------------------
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
model = pickle.load(open("models/model.pkl", "rb"))

try:
    THRESHOLDS = pickle.load(open("models/thresholds.pkl", "rb"))
except FileNotFoundError:
    # fallback thresholds (only used if thresholds.pkl missing)
    THRESHOLDS = {
        "toxic":         0.35,
        "severe_toxic":  0.60,
        "obscene":       0.55,
        "threat":        0.40,
        "insult":        0.45,
        "identity_hate": 0.65,
    }

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# -----------------------
# RULE KEYWORDS
# -----------------------
SEXUAL_KEYWORDS = [
    "boobs", "naked", "nude", "nudes", "press your",
    "see you naked", "have sex", "sleep with you"
]

THREAT_KEYWORDS = [
    "kill you", "beat you up", "hit you", "destroy your life"
]

NEGATIVE_WORDS = [
    "disgusting", "trash", "hate", "less than human",
    "should disappear"
]

IDENTITY_WORDS = [
    "race", "religion", "country", "community", "culture", "language"
]

# generic violence + promotion patterns
VIOLENCE_WORDS = [
    "killing", "murder", "genocide", "massacre", "slaughter"
]

PROMOTE_WORDS = [
    "promote", "support", "encourage", "want", "favour", "favor", "like", "love"
]

NEGATIONS = [
    "dont", "don't", "do not", "against", "oppose", "stop", "ban"
]

# clearly positive phrases – overrides if no other bad signal
POSITIVE_PHRASES = [
    "you are beautiful",
    "you are so beautiful",
    "you are very beautiful",
    "you are amazing",
    "you are awesome",
    "you are smart",
    "you are kind",
    "you are great",
    "you are the best",
    "i love you",
    "great job",
    "nice work",
    "well done",
    "thank you",
    "thanks a lot"
]


# -----------------------
# RULE LAYER
# -----------------------
def apply_rules(original_text: str, result: dict) -> dict:
    """
    Deterministic rules on top of ML:
    - sexual content       -> toxic + obscene
    - direct threats       -> toxic + threat
    - promote violence     -> threat = 1
    - identity + negative  -> toxic + identity_hate
    - clear compliments    -> force CLEAN (all labels 0)
    """
    t = original_text.lower()

    # --- detect categories first ---
    sexual_present = any(k in t for k in SEXUAL_KEYWORDS)
    threat_phrase_present = any(k in t for k in THREAT_KEYWORDS)
    violence_present = any(v in t for v in VIOLENCE_WORDS)
    promote_present = any(p in t for p in PROMOTE_WORDS)
    negation_present = any(neg in t for neg in NEGATIONS)
    negative_present = any(n in t for n in NEGATIVE_WORDS)
    identity_present = any(i in t for i in IDENTITY_WORDS)
    positive_present = any(p in t for p in POSITIVE_PHRASES)

    # --- apply "bad" rules ---

    # sexual / obscene
    if sexual_present:
        result["toxic"] = 1
        result["obscene"] = 1

    # direct threats like "i will kill you"
    if threat_phrase_present:
        result["toxic"] = 1
        result["threat"] = 1

    # promoting violence (e.g. "i promote killing")
    if violence_present and promote_present and not negation_present:
        # here we only force threat; toxic left to model / other rules
        result["threat"] = 1

    # identity-based hate: negative + identity word together
    if negative_present and identity_present:
        result["toxic"] = 1
        result["identity_hate"] = 1

    # --- positive override (must be last) ---
    # If it's clearly a compliment and has no bad patterns,
    # we force all labels to 0.
    if positive_present and not (
        sexual_present
        or threat_phrase_present
        or violence_present
        or negative_present
        or identity_present
    ):
        for k in result.keys():
            result[k] = 0

    return result


# -----------------------
# PREDICT FUNCTION
# -----------------------
def predict_text(text: str):
    """
    Full inference pipeline:
    - preprocess
    - TF-IDF transform
    - logistic regression probabilities
    - per-label thresholds
    - rule adjustments
    """
    clean = preprocess(text)
    vec = vectorizer.transform([clean])

    # probabilities for each label
    probas = model.predict_proba(vec)[0]  # shape: (6,)

    preds = []
    for i, label in enumerate(labels):
        p = probas[i]
        thr = THRESHOLDS.get(label, 0.5)
        pred = int(p >= thr)
        preds.append(pred)

    result = {labels[i]: preds[i] for i in range(len(labels))}

    # Apply rules based on original (non-cleaned) text
    result = apply_rules(text, result)

    # Overall status = TOXIC if any category is 1
    toxic_status = "TOXIC" if any(result.values()) else "CLEAN"

    return toxic_status, result


# -----------------------
# TEST FROM TERMINAL
# -----------------------
if __name__ == "__main__":
    text = input("Enter text: ")
    status, details = predict_text(text)
    print("Status:", status)
    print("Label details:", details)

    action = perform_action(status, details, text)
    print("Action:", action)
