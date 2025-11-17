from flask import Flask, request, jsonify, send_from_directory
from scripts.predict import predict_text
from scripts.actions import perform_action

app = Flask(__name__)

# -----------------------
# SERVE FRONTEND
# -----------------------
@app.route("/")
def index():
    # Serve frontend.html from project root
    return send_from_directory(".", "frontend.html")


# -----------------------
# API ENDPOINT FOR PREDICTION
# -----------------------
@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)  # parse JSON body
    text = data.get("text", "")

    if not text.strip():
        return jsonify({
            "error": "Text is empty",
            "status": "CLEAN",
            "details": {
                "toxic": 0,
                "severe_toxic": 0,
                "obscene": 0,
                "threat": 0,
                "insult": 0,
                "identity_hate": 0
            },
            "action": "No text provided."
        }), 400

    # Run your model
    status, details = predict_text(text)
    action = perform_action(status, details, text)

    return jsonify({
        "status": status,
        "details": details,
        "action": action,
        "original_text": text
    })


if __name__ == "__main__":
    # Run on http://127.0.0.1:5000
    app.run(host="127.0.0.1", port=5000, debug=True)
