import joblib
import numpy as np

class SentimentPredictor:
    """
    Inference-only sentiment prediction class
    Loads trained artifacts once and performs predictions on new text.
    """
    def __init__(self, model_path = "model.pkl", vectorizer_path = "vectorizer.pkl"):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text: str):
        """
        Predict sentiment for a single text input.
        
        Returns:
            dict:{
                "label": "postive" or "negative",
                confidence: float
            }
        """
        if not text or not text.strip():
            raise ValueError("Input text must be a non-empty string.")
        
        # Transform input text
        text_vec = self.vectorizer.transform([text])

        # Predict class and probability
        prediction = self.model.predict(text_vec)[0]
        probabilities = self.model.predict_proba(text_vec)[0]

        confidence = float(np.max(probabilities))

        label = "positive" if prediction == 1 else "negative"

        return {
            "label": label,
            "confidence": confidence
        }
if __name__ == "__main__":
    # Simple test
    predictor = SentimentPredictor()
    sample_text = "The movie was surprisingly good and well-acted."
    result = predictor.predict(sample_text)
    print(result)