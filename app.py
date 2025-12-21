import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def build_model():
       texts = [
        "I love this product", "Great service", "Excellent quality", "Fast delivery", "Best purchase",
        "Amazing experience", "Very happy with this", "Good value", "Super helpful support", "Highly recommend",
        "Perfect condition", "Works beautifully", "Five stars", "Incredible performance", "So easy to use",
        "Nice design", "Prompt shipping", "Exceeded expectations", "Really satisfied", "Top notch",
        
        "Terrible waste of money", "Bad service", "Hate this", "Slow and rude", "Disappointed",
        "Broken on arrival", "Worst purchase ever", "Not worth it", "Poor quality", "Did not work",
        "Avoid this product", "Very angry", "Useless item", "The delivery was late", "Defective",
        "Too expensive for what it is", "Complete failure", "Refund please", "Horrible experience", "Garbage"
    ]
       labels = [1] * 20 + [0] * 20  # 1 for positive, 0 for negative

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
