import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('/content/fake_news_model.pkl')

st.title("Fake News Detector")

# Single headline input
headline = st.text_input("Enter a news headline:")
if headline:
    prediction = model.predict([headline])[0]
    prob = model.predict_proba([headline])[0][0] if prediction == 'fake' else model.predict_proba([headline])[0][1]
    st.write(f"Prediction: **{prediction.upper()}**")
    st.write(f"Confidence: **{round(prob * 100, 2)}%**")

# CSV upload for batch check
uploaded_file = st.file_uploader("Upload CSV for batch prediction (column: 'text')", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'text' in df.columns:
        predictions = model.predict(df['text'])
        probs = model.predict_proba(df['text'])
        df['prediction'] = predictions
        df['confidence'] = [max(p) for p in probs]  # Max probability for confidence
        st.write(df)
        st.download_button("Download Results", df.to_csv(index=False), "results.csv")
    else:
        st.error("CSV must have a 'text' column.")
