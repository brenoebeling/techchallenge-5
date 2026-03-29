import os
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

INPUT_PATH = "data/processed/complaints_labeled.parquet"
OUTPUT_TABLE_PATH = "reports/tables/top_terms_by_product.csv"
OUTPUT_FIG_DIR = "reports/figures/wordclouds"

# stopwords adicionais específicas do domínio
CUSTOM_STOPWORDS = {
    "xxxx", "xx", "would", "could", "also", "said", "told", "got", "get",
    "one", "two", "back", "called", "call", "email", "sent", "letter",
    "company", "bank", "account", "credit", "loan", "card", "payment",
    "payments", "consumer", "complaint", "complaints", "reported",
    "report", "response", "customer", "customers", "service", "services",
    "day", "days", "time", "months", "month", "year", "years"
}


def tokenize(text: str) -> list[str]:
    if pd.isna(text):
        return []

    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2 and t not in CUSTOM_STOPWORDS]
    return tokens


def get_top_terms(text_series: pd.Series, top_n: int = 30) -> list[tuple[str, int]]:
    counter = Counter()

    for text in text_series.dropna():
        counter.update(tokenize(text))

    return counter.most_common(top_n)


def save_wordcloud(text_series: pd.Series, product_name: str, output_dir: str) -> None:
    full_text = " ".join(text_series.dropna().astype(str).tolist()).strip()

    if not full_text:
        return

    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        max_words=150
    ).generate(full_text)

    plt.figure(figsize=(14, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud - {product_name}")

    safe_name = (
        product_name.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/wordcloud_{safe_name}.png", bbox_inches="tight")
    plt.close()


def main():
    print("Lendo base...")
    df = pd.read_parquet(INPUT_PATH)

    print("Filtrando reclamações negativas...")
    df = df[df["sentiment"] == 0].copy()

    # garante textos válidos
    df = df[df["clean_text"].notna()]
    df = df[df["clean_text"].str.len() > 10]

    print(f"Total de negativas: {len(df):,}")

    products = (
        df["Product"]
        .fillna("Unknown")
        .value_counts()
        .index
        .tolist()
    )

    rows = []

    print("Gerando tabelas e wordclouds por produto...")
    for product in products:
        product_df = df[df["Product"].fillna("Unknown") == product].copy()

        # evita produtos minúsculos demais
        if len(product_df) < 1000:
            continue

        top_terms = get_top_terms(product_df["clean_text"], top_n=30)

        for term, freq in top_terms:
            rows.append({
                "Product": product,
                "term": term,
                "frequency": freq
            })

        save_wordcloud(product_df["clean_text"], product, OUTPUT_FIG_DIR)

        print(f"Produto: {product} | negativas: {len(product_df):,}")

    top_terms_df = pd.DataFrame(rows)
    os.makedirs("reports/tables", exist_ok=True)
    top_terms_df.to_csv(OUTPUT_TABLE_PATH, index=False)

    print(f"Tabela salva em: {OUTPUT_TABLE_PATH}")
    print(f"Imagens salvas em: {OUTPUT_FIG_DIR}")
    print("Finalizado!")


if __name__ == "__main__":
    main()
