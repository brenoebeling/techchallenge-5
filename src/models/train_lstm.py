import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =========================
# CONFIG
# =========================
TRAIN_PATH = "data/processed/train.parquet"
VAL_PATH = "data/processed/val.parquet"
TEST_PATH = "data/processed/test.parquet"

ARTIFACT_DIR = "models_artifacts/lstm"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "lstm_model.keras")
TOKENIZER_PATH = os.path.join(ARTIFACT_DIR, "tokenizer_config.json")

VOCAB_SIZE = 50000
MAX_LEN = 120
EMBED_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 512
EPOCHS = 8
RANDOM_STATE = 42

os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def load_data():
    print("Carregando dados...")
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    cols = ["clean_text", "sentiment"]
    train_df = train_df[cols].dropna()
    val_df = val_df[cols].dropna()
    test_df = test_df[cols].dropna()

    train_df = train_df[train_df["clean_text"].str.len() > 5]
    val_df = val_df[val_df["clean_text"].str.len() > 5]
    test_df = test_df[test_df["clean_text"].str.len() > 5]

    print(f"Train: {len(train_df):,}")
    print(f"Val:   {len(val_df):,}")
    print(f"Test:  {len(test_df):,}")

    return train_df, val_df, test_df


def prepare_tokenizer(train_texts):
    print("Treinando tokenizer...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    return tokenizer


def texts_to_padded(tokenizer, texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )
    return padded


def build_model():
    print("Construindo modelo LSTM...")
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, input_length=MAX_LEN),
        LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.0),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    return model


def save_tokenizer(tokenizer):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    tokenizer_json = tokenizer.to_json()
    with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
        f.write(tokenizer_json)
    print(f"Tokenizer salvo em: {TOKENIZER_PATH}")


def evaluate_model(model, X, y, split_name="TEST", threshold=0.5):
    print(f"\nAvaliação - {split_name}:")
    y_prob = model.predict(X, batch_size=BATCH_SIZE, verbose=1).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    print(classification_report(y, y_pred, digits=4))
    print("Matriz de confusão:")
    print(confusion_matrix(y, y_pred))

    return y_pred, y_prob


def main():
    train_df, val_df, test_df = load_data()

    X_train_text = train_df["clean_text"].astype(str).tolist()
    y_train = train_df["sentiment"].astype(int).values

    X_val_text = val_df["clean_text"].astype(str).tolist()
    y_val = val_df["sentiment"].astype(int).values

    X_test_text = test_df["clean_text"].astype(str).tolist()
    y_test = test_df["sentiment"].astype(int).values

    tokenizer = prepare_tokenizer(X_train_text)

    print("Convertendo textos em sequências...")
    X_train = texts_to_padded(tokenizer, X_train_text)
    X_val = texts_to_padded(tokenizer, X_val_text)
    X_test = texts_to_padded(tokenizer, X_test_text)

    save_tokenizer(tokenizer)

    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    print("\nTreinando modelo...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\nModelo salvo em: {MODEL_PATH}")

    evaluate_model(model, X_val, y_val, split_name="VAL", threshold=0.5)
    evaluate_model(model, X_test, y_test, split_name="TEST", threshold=0.5)

    history_path = os.path.join(ARTIFACT_DIR, "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f)
    print(f"Histórico salvo em: {history_path}")

    print("\nFinalizado!")


if __name__ == "__main__":
    main()
