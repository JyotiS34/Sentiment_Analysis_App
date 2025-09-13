# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import numpy as np
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax

# -------------------------------
# Setup
# -------------------------------
st.set_page_config(page_title="Sentiment Analysis", layout="wide")
st.title("Sentiment Analysis with VADER & RoBERTa")

# Download NLTK resources
nltk.download("punkt")
nltk.download("vader_lexicon")

# Initialize models (cached to avoid reloading)
@st.cache_resource
def load_models():
    sia = SentimentIntensityAnalyzer()
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    sent_pipeline = pipeline("sentiment-analysis")
    return sia, tokenizer, model, sent_pipeline

sia, tokenizer, model, sent_pipeline = load_models()

# -------------------------------
# Upload Dataset
# -------------------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload twitter_training.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding_errors="replace")
    st.write("### Dataset Preview", df.head())

    if "Unnamed: 1" in df.columns:
        counts = df["Unnamed: 1"].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Set2")
        ax.set_title("Sentiment Distribution in Dataset")
        st.pyplot(fig)

    # Define Roberta polarity scoring function
    def polarity_scores_roberta(text):
        encoded_text = tokenizer(text, return_tensors="pt")
        output = model(**encoded_text)
        scores = softmax(output[0][0].detach().numpy())
        return {
            "roberta_neg": float(scores[0]),
            "roberta_neu": float(scores[1]),
            "roberta_pos": float(scores[2]),
        }

    # Compute VADER + RoBERTa scores for dataset
    res = {}
    for i, row in tqdm(df.head(1000).iterrows(), total=1000):
        try:
            text = row['Text']
            label = row['Unnamed: 1']
            if pd.notna(text):
                vader_result = sia.polarity_scores(text)
                vader_result_rename = {
                    f"vader_{key}": value for key, value in vader_result.items()
                }
                roberta_result = polarity_scores_roberta(text)
                combined = {**vader_result_rename, **roberta_result}
                res[i] = {**combined, "label": label, "Text": text}
            else:
                print(f"Skipping row with None text at index {i}")
        except Exception as e:
            print(f"Error processing row {i}: {e}")

    # Build results DataFrame
    results_df = pd.DataFrame(res).T.reset_index(drop=True)

    # Plot comparison graph
    sns.pairplot(
        data=results_df,
        vars=['vader_neg', 'vader_neu', 'vader_pos',
              'roberta_neg', 'roberta_neu', 'roberta_pos'],
        hue='label',
        palette='tab10'
    )
    st.pyplot(plt)


# -------------------------------
# Text input for analysis
# -------------------------------
st.subheader("Sentiment analysis!")
user_text = st.text_area("Enter text:", "I love sentiment analysis!")

if user_text:
    # VADER
    vader_scores = sia.polarity_scores(user_text)

    # RoBERTa
    encoded_text = tokenizer(user_text, return_tensors="pt")
    output = model(**encoded_text)
    scores = softmax(output[0][0].detach().numpy())
    roberta_scores = {
        "Negative": float(scores[0]),
        "Neutral": float(scores[1]),
        "Positive": float(scores[2]),
    }

# Compute entropy of scores
entropy = -sum(p * np.log(p + 1e-10) for p in scores)

# Threshold-based labeling logic
if  (0.72 > entropy >= 0.2832448): 
    final_label = "Neutral"

elif  (1 > entropy >= 0.24): 
    final_label = "Irrelevant"
    
else:
    max_label = max(roberta_scores, key=roberta_scores.get)
    final_label = max_label

col1, col2 = st.columns(2)
with col1:
    st.write("### VADER Scores")
    st.json(vader_scores)
with col2:
    st.write("### Roberta Scores")
    st.json(roberta_scores)

# HuggingFace pipeline result (default)
pipeline_result = sent_pipeline(user_text)[0]  # [{'label': ..., 'score': ...}]
st.write("### HuggingFace Pipeline (Default Label)")
st.json(pipeline_result)

# Display final sentiment label
st.write("### Final Sentiment Prediction (with Neutral & Irrelevant)")
st.json({
    "Final Label": final_label,
    "Entropy": entropy,
    "Scores": roberta_scores
})
# -------------------------------
# Footer
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.write("Built with Streamlit, NLTK & HuggingFace")
