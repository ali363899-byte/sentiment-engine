import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def normalize_text(text: str) -> str:
    # Arabic and English text preprocessing & normalization
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)  # Remove Arabic diacritics
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^a-zA-Z\u0600-\u06FF\s]', '', text)
    return text.strip().lower()

# Benchmark bilingual corpus (MSA, Gulf, Levantine, English)
training_data = [
    ("This system is incredibly reliable and fast", "Positive"),
    ("Great performance and clean architecture", "Positive"),
    ("النظام ممتاز جدا وسريع في الاستجابة", "Positive"),
    ("الخدمة ممتازة وتستحق كل التقدير", "Positive"),
    ("خدمة رايعة وممتازة", "Positive"),
    ("Severe latency and frequent crash errors", "Negative"),
    ("Poor architecture and completely unoptimized", "Negative"),
    ("التطبيق سيء جدا وبطيء في المعالجة", "Negative"),
    ("خدمة سيئة وتجربة مستخدم محبطة", "Negative"),
    ("تطبيق بطيء وفيه مشاكل كثيرة", "Negative")
]

df = pd.DataFrame(training_data, columns=["text", "sentiment"])
df["clean_text"] = df["text"].apply(normalize_text)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression())
])

pipeline.fit(df["clean_text"], df["sentiment"])

with open("sentiment_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Pipeline trained and exported successfully to sentiment_pipeline.pkl")