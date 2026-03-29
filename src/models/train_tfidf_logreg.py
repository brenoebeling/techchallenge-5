import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

TRAIN_PATH = "data/processed/train.parquet"
VAL_PATH = "data/processed/val.parquet"
TEST_PATH = "data/processed/test.parquet"

MODEL_PATH = "models_artifacts/tfidf_logreg.joblib"


def main():
    print("Carregando dados...")
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_train = train_df["clean_text"]
    y_train = train_df["sentiment"]

    X_val = val_df["clean_text"]
    y_val = val_df["sentiment"]

    X_test = test_df["clean_text"]
    y_test = test_df["sentiment"]

    print("Vetorizando texto com TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1,2),
        min_df=5
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Treinando modelo...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_tfidf, y_train)

    print("\nAvaliação - VAL:")
    y_val_pred = model.predict(X_val_tfidf)
    print(classification_report(y_val, y_val_pred))

    print("\nAvaliação - TEST:")
    y_test_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, y_test_pred))

    print("\nMatriz de confusão (TEST):")
    print(confusion_matrix(y_test, y_test_pred))

    print("\nSalvando modelo...")
    joblib.dump((vectorizer, model), MODEL_PATH)

    print("Finalizado!")


if __name__ == "__main__":
    main()
