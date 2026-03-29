import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = "reports/tables/pain_points_structured.csv"
OUTPUT_DIR = "reports/figures/pain_points"


def main():
    print("Lendo tabela estruturada...")
    df = pd.read_csv(INPUT_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # gráfico 1: total por grupo de produto
    product_totals = (
        df.groupby("product_group", as_index=False)["frequency"]
        .sum()
        .sort_values("frequency", ascending=False)
    )

    plt.figure(figsize=(12, 7))
    plt.bar(product_totals["product_group"], product_totals["frequency"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Total de dores por grupo de produto")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pain_by_product_group.png")
    plt.close()

    # gráfico 2: principal dor por grupo
    top_pains = (
        df.sort_values(["product_group", "frequency"], ascending=[True, False])
        .groupby("product_group", as_index=False)
        .first()
        .sort_values("frequency", ascending=False)
    )

    plt.figure(figsize=(12, 7))
    plt.bar(top_pains["product_group"], top_pains["frequency"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Principal dor por grupo de produto")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/top_pain_by_product_group.png")
    plt.close()

    # gráfico 3: um gráfico por produto com top 5 dores
    for product in df["product_group"].unique():
        temp = (
            df[df["product_group"] == product]
            .sort_values("frequency", ascending=False)
            .head(5)
        )

        plt.figure(figsize=(10, 6))
        plt.bar(temp["pain"], temp["frequency"])
        plt.xticks(rotation=30, ha="right")
        plt.title(f"Top 5 dores - {product}")
        plt.ylabel("Frequência")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/top5_{product}.png")
        plt.close()

    print(f"Gráficos salvos em: {OUTPUT_DIR}")
    print("Finalizado!")


if __name__ == "__main__":
    main()
