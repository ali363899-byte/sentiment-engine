import os
import pickle
from flask import Flask, request, jsonify
from model import normalize_text

app = Flask(__name__)

# Load trained pipeline
with open("sentiment_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "active",
        "service": "Bilingual Sentiment Analysis Engine",
        "supported_languages": ["Arabic (Gulf, Levantine, MSA)", "English"],
        "endpoints": {"predict": "/predict [POST]"}
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    clean = normalize_text(text)
    prediction = pipeline.predict([clean])[0]
    probs = pipeline.predict_proba([clean])[0]
    confidence = max(probs)

    return jsonify({
        "input_text": text,
        "normalized_text": clean,
        "sentiment": prediction,
        "confidence": round(float(confidence), 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)