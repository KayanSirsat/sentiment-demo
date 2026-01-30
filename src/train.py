from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def load_imdb_data(sample_size = 20000):
    dataset = load_dataset("imdb")
    texts = dataset["train"]["text"][:sample_size]
    labels = dataset["train"]["label"][:sample_size]
    return texts, labels

def main():
    print("Loading IMDB dataset...")
    texts, labels = load_imdb_data()

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size = 0.2,
        random_state = 30,
        stratify = labels
    )

    vectorizer = TfidfVectorizer(
        max_features = 20000,
        ngram_range = (1, 2),
        stop_words = "english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter = 1000, n_jobs = 1)
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    print(classification_report(y_test, predictions))

    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

if __name__ == "__main__":
    main()