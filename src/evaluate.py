import joblib
from datasets import load_dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def load_imdb_data(sample_size = 20000):
    dataset = load_dataset("imdb")
    texts = dataset["train"]["text"][:sample_size]
    labels = dataset["train"]["label"][:sample_size]

    _, X_test, _, y_test = train_test_split(
        texts, 
        labels,
        test_size = 0.2,
        random_state = 42,
        stratify = labels
    )

def main():
    print("Loading trained artifacts...")
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    
    print("Loading test data...")
    X_test, y_test = load_imdb_data()

    print("Vectorizing test data...")
    X_test_vec = vectorizer.transform(X_test)
    
    print("Evaluating model...")
    predictions = model.predict(X_test_vec)
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    main()