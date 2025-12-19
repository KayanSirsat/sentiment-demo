import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def build_model():

    texts = ["I love this product", "Great service", "Excellent quality", "Fast delivery", "Best purchase", "Terrible waste", "Bad service", "Hate this", "Slow and rude", "Disappoint", "Terrible experience"]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 1 for positive, 0 for negative

    vec = CountVectorizer()
    X = vec.fit_transform(texts)
    model = LogisticRegression()
    model.fit(X, labels)
    return vec, model

vectorizer, model = build_model()

st.set_page_config(page_title="Sentiment Analysis")
st.title("Customer Sentiment Analysis")
st.write("Enter a review below to test the sentiment model.")

user_input = st.text_area("Enter your review here:", height=150)

if st.button("Predict Sentiment"):
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)[0]
    probability = model.predict_proba(input_vector)[0][1]

    if prediction == 1:
        st.success(f"Postive Sentiment (Confidence: {probability:.2%})")
    else:
        st.error(f"Negative Sentiment (Confidence: {1 - probability:.2%})")

st.info("Note: This UI serves the model logic. The core training pipeline was built in PySpark.")
