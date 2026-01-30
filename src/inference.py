import os
import joblib
import numpy as np

class SentimentPredictor:
    """
    Inference-only sentiment prediction class.
    Loads trained artifacts using absolute paths for cloud compatibility.
    """
    def __init__(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, "model.pkl")
        vectorizer_path = os.path.join(project_root, "vectorizer.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
        
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text:str):
        if not text or not text.strip():
            raise ValueError("Input text must be a non-empty string.")
        
        text_vec = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vec)
        probabilities = self.model.predict_proba(text_vec)

        confidence = np.max(probabilities)
        label = "positive" if prediction == 1 else "negative"

        return {
            "label" : label,
            "confidence" : confidence
        }
        