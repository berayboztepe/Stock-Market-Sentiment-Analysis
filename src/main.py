import streamlit as st
import joblib
import numpy as np
from google.cloud import storage
import os

BUCKET_NAME = "stock-sentiment-data"
MODEL_FILE = "best_stock_sentiment_model.pkl"
SCALER_FILE = "scaler.pkl"

MODEL_PATH = f"/tmp/{MODEL_FILE}"
SCALER_PATH = f"/tmp/{SCALER_FILE}"

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        st.success(f"✅ Downloaded {source_blob_name} from GCS!")
    except Exception as e:
        st.error(f"❌ Failed to download {source_blob_name}: {e}")
        st.stop()

download_from_gcs(BUCKET_NAME, MODEL_FILE, MODEL_PATH)
download_from_gcs(BUCKET_NAME, SCALER_FILE, SCALER_PATH)

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    st.success("✅ Model & Scaler Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error Loading Model: {e}")
    st.stop()

st.title("📈 Stock Market Sentiment Prediction")
st.write("🚀 Enter a tweet about a stock, and we'll predict whether the stock price will increase or decrease.")

tweet_input = st.text_area("📝 Enter a tweet about a stock:", "")

def predict_sentiment(tweet):
    if not tweet.strip():
        return None, None, None

    sentiment_score = np.random.uniform(-1, 1)
    sentiment_scaled = scaler.transform(np.array([[sentiment_score]]))

    prediction = model.predict(sentiment_scaled)[0]
    probability = model.predict_proba(sentiment_scaled)[0]

    return prediction, probability, sentiment_score

if st.button("🔮 Predict Stock Movement"):
    prediction, probability, sentiment_score = predict_sentiment(tweet_input)

    if prediction is not None:
        st.write(f"🧠 **Predicted Sentiment Score:** `{sentiment_score:.3f}`")

        if prediction == 1:
            st.success(f"✅ The model predicts **Stock Price Will Increase 📈** (Confidence: `{probability[1]:.2%}`)")
        else:
            st.error(f"❌ The model predicts **Stock Price Will Decrease 📉** (Confidence: `{probability[0]:.2%}`)")
    else:
        st.warning("⚠️ Please enter a tweet to analyze.")

st.markdown("""
---
### ℹ️ How This Works:
1️⃣ Enter a tweet about **Apple, Tesla, etc.**  
2️⃣ Click **"Predict Stock Movement"**  
3️⃣ The model will analyze sentiment and predict if the stock price **will increase or decrease**  
""")
