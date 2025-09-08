!pip install streamlit
import streamlit as st
import pandas as pd
import string
import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load the trained model
model = joblib.load('fake_news_model.pkl')

st.title("📰 Fake News Detector")

# ---------- Helper Functions ---------- #
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return ' '.join(filtered_words)

def preprocess_text(text):
    text = str(text).lower()
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    return text

# ---------- Single Headline Input ---------- #
st.header("🔍 Single Headline Check")
headline = st.text_input("Enter a news headline:")

if headline:
    processed_headline = preprocess_text(headline)
    prediction = model.predict([processed_headline])[0]
    prob = model.predict_proba([processed_headline])[0]
    confidence = max(prob)

    st.markdown(f"**Prediction:** `{prediction.upper()}`")
    st.markdown(f"**Confidence:** `{round(confidence * 100, 2)}%`")

# ---------- Batch CSV Upload ---------- #
st.header("📁 Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV (must contain a 'text' column)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'text' in df.columns:
        df['processed_text'] = df['text'].astype(str).apply(preprocess_text)
        predictions = model.predict(df['processed_text'])
        probs = model.predict_proba(df['processed_text'])gi

        df['prediction'] = [p.upper() for p in predictions]
        df['confidence'] = (probs.max(axis=1) * 100).round(2)

        st.success("Prediction completed!")
        st.dataframe(df[['text', 'prediction', 'confidence']])

        # CSV download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="fake_news_predictions.csv",
            mime="text/csv"
        )
    else:
        st.error("Uploaded CSV does not contain a 'text' column.")
