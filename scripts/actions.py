import os

# ---------------------------
# FIX: CREATE LOG DIRECTORY SAFELY
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # scripts/
LOG_DIR = os.path.join(BASE_DIR, "logs")               # scripts/logs/

# Make sure logs folder exists
os.makedirs(LOG_DIR, exist_ok=True)


# ---------------------------
# TOXIC ACTION
# ---------------------------
def toxic_action(text, details):
    score = sum(details.values())

    toxic_log_path = os.path.join(LOG_DIR, "toxic_log.txt")

    with open(toxic_log_path, "a", encoding="utf-8") as f:
        f.write(f"[TOXIC] (score={score}) → {text}\n")

    return f"⚠️ Toxic detected | Score={score} | Logged & User flagged."


# ---------------------------
# CLEAN ACTION
# ---------------------------
def clean_action(text):
    clean_log_path = os.path.join(LOG_DIR, "clean_log.txt")

    with open(clean_log_path, "a", encoding="utf-8") as f:
        f.write(f"[CLEAN] → {text}\n")

    return "✅ Clean comment accepted | Reward +1 point."


# ---------------------------
# MAIN DECISION FUNCTION
# ---------------------------
def perform_action(status, details, text):
    if status == "TOXIC":
        return toxic_action(text, details)
    else:
        return clean_action(text)
