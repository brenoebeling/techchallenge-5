import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = "data/processed/complaints_labeled.parquet"

TRAIN_PATH = "data/processed/train.parquet"
VAL_PATH = "data/processed/val.parquet"
TEST_PATH = "data/processed/test.parquet"


def main():
    print("Lendo base rotulada...")
    df = pd.read_parquet(INPUT_PATH)

    # Mantém apenas o necessário para modelagem inicial
    df = df[["clean_text", "Product", "Issue", "Company response to consumer", "sentiment"]].copy()

    # Remove vazios
    df = df[df["clean_text"].notna()]
    df = df[df["clean_text"].str.len() > 10]

    print(f"Base final para split: {len(df):,}")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["sentiment"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["sentiment"]
    )

    print("Distribuição:")
    print(f"Train: {len(train_df):,}")
    print(f"Val:   {len(val_df):,}")
    print(f"Test:  {len(test_df):,}")

    print("\nProporção sentimento:")
    print("Train")
    print(train_df["sentiment"].value_counts(normalize=True))
    print("Val")
    print(val_df["sentiment"].value_counts(normalize=True))
    print("Test")
    print(test_df["sentiment"].value_counts(normalize=True))

    train_df.to_parquet(TRAIN_PATH, index=False)
    val_df.to_parquet(VAL_PATH, index=False)
    test_df.to_parquet(TEST_PATH, index=False)

    print("\nArquivos salvos com sucesso.")


if __name__ == "__main__":
    main()
