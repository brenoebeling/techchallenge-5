"""
Arquivo: src/preprocessing/clean_text.py

Objetivo:
Limpar e normalizar o texto da narrativa das reclamações para uso em vetorização
e modelagem de sentimento.

Fonte de dados:
Arquivo intermediário normalizado do pipeline

Arquivo de entrada:
data/interim/complaints_normalized.parquet

Arquivo gerado:
data/processed/complaints_cleaned.parquet
"""

import pandas as pd
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(ENGLISH_STOP_WORDS)

INPUT_PATH = "data/interim/complaints_selected.parquet"
OUTPUT_PATH = "data/processed/complaints_cleaned.parquet"


def clean_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]

    return " ".join(tokens)


def main():
    print("Lendo dados...")
    df = pd.read_parquet(INPUT_PATH)

    print("Filtrando textos válidos...")
    df = df[df["Consumer complaint narrative"].notna()]
    df = df[df["Consumer complaint narrative"].str.len() > 50]

    print("Aplicando limpeza...")
    tqdm.pandas()
    df["clean_text"] = df["Consumer complaint narrative"].progress_apply(clean_text)

    print("Salvando...")
    df.to_parquet(OUTPUT_PATH, index=False)

    print("Finalizado!")
    print(f"Linhas finais: {len(df):,}")


if __name__ == "__main__":
    main()
