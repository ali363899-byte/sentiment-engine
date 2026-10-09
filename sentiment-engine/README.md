# 🌐 Bilingual Sentiment Analysis Engine

An end-to-end NLP classification pipeline and RESTful microservice built to process dialectical Arabic (Gulf, Levantine, Modern Standard Arabic) and English text.

## 🚀 Features
- **Custom Arabic & English Normalization:** Handles orthographic variations, diacritics removal, and multilingual tokenization.
- **TF-IDF + Ensemble Pipeline:** Evaluates n-grams (1,2) with probability scoring.
- **REST API Serving:** Lightweight Flask microservice returning structured JSON predictions with sub-100ms response times.

## 🛠️ Tech Stack
- Python 3.10+
- scikit-learn
- pandas & NumPy
- Flask

## 📦 Setup & Run Locally
```bash
git clone [https://github.com/ali363899-byte/sentiment-engine.git](https://github.com/ali363899-byte/sentiment-engine.git)
cd sentiment-engine
pip install -r requirements.txt
python model.py
python app.py