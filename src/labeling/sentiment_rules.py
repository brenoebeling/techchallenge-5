import pandas as pd

INPUT_PATH = "data/processed/complaints_cleaned.parquet"
OUTPUT_PATH = "data/processed/complaints_labeled.parquet"


def create_sentiment(row):
    response = str(row["Company response to consumer"]).lower()
    disputed = str(row["Consumer disputed?"]).lower()

    # NEGATIVO forte
    if "untimely" in response:
        return 0

    if "disputed" in disputed:
        return 0

    if "closed with explanation" in response:
        return 0

    # POSITIVO (relativamente resolvido)
    if "closed with monetary relief" in response:
        return 1

    if "closed with non-monetary relief" in response:
        return 1

    # fallback → negativo (conservador)
    return 0


def main():
    print("Lendo dados...")
    df = pd.read_parquet(INPUT_PATH)

    print("Criando variável de sentimento...")
    df["sentiment"] = df.apply(create_sentiment, axis=1)

    print("Distribuição:")
    print(df["sentiment"].value_counts(normalize=True))

    print("Salvando...")
    df.to_parquet(OUTPUT_PATH, index=False)

    print("Finalizado!")


if __name__ == "__main__":
    main()
