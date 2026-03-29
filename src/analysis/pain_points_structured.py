import os
import pandas as pd
from collections import Counter, defaultdict

INPUT_PATH = "data/processed/complaints_labeled.parquet"
OUTPUT_PATH = "reports/tables/pain_points_structured.csv"

# -------------------------
# 1. NORMALIZAÇÃO DE PRODUTOS
# -------------------------

PRODUCT_MAP = {
    "credit_reporting": [
        "Credit reporting",
        "Credit reporting or other personal consumer reports",
        "Credit reporting, credit repair services, or other personal consumer reports"
    ],
    "debt_collection": [
        "Debt collection"
    ],
    "banking": [
        "Checking or savings account",
        "Bank account or service"
    ],
    "credit_card": [
        "Credit card",
        "Credit card or prepaid card"
    ],
    "loans": [
        "Mortgage",
        "Student loan",
        "Vehicle loan or lease",
        "Consumer Loan"
    ],
    "payments": [
        "Money transfer, virtual currency, or money service",
        "Money transfers",
        "Prepaid card"
    ],
    "high_risk_loans": [
        "Payday loan",
        "Payday loan, title loan, or personal loan",
        "Payday loan, title loan, personal loan, or advance loan"
    ]
}


def map_product(product):
    for group, values in PRODUCT_MAP.items():
        if product in values:
            return group
    return "other"


# -------------------------
# 2. DICIONÁRIO DE DORES
# -------------------------

PAIN_DICTIONARY = {
    "fraud": ["fraud", "unauthorized", "scam", "identity"],
    "interest_fees": ["interest", "fee", "fees", "charge", "charged"],
    "delay": ["delay", "late", "pending", "processing"],
    "customer_service": ["service", "support", "call", "representative"],
    "credit_score": ["score", "report", "credit", "history"],
    "payment_issues": ["payment", "paid", "balance", "due"],
    "account_access": ["access", "login", "blocked", "closed"],
    "debt_dispute": ["debt", "collection", "owe", "dispute"]
}


def classify_pain(tokens):
    pain_counts = defaultdict(int)

    for token in tokens:
        for pain, keywords in PAIN_DICTIONARY.items():
            if token in keywords:
                pain_counts[pain] += 1

    if not pain_counts:
        return "other"

    return max(pain_counts, key=pain_counts.get)


# -------------------------
# 3. TOKENIZAÇÃO
# -------------------------

def tokenize(text):
    if pd.isna(text):
        return []
    return text.split()


# -------------------------
# 4. PIPELINE PRINCIPAL
# -------------------------

def main():
    print("Lendo dados...")
    df = pd.read_parquet(INPUT_PATH)

    print("Filtrando negativos...")
    df = df[df["sentiment"] == 0].copy()

    df["product_group"] = df["Product"].fillna("Unknown").apply(map_product)

    print("Classificando dores...")
    pain_counter = defaultdict(Counter)

    for _, row in df.iterrows():
        tokens = tokenize(row["clean_text"])
        pain = classify_pain(tokens)
        product = row["product_group"]

        pain_counter[product][pain] += 1

    print("Gerando tabela final...")
    rows = []

    for product, counter in pain_counter.items():
        total = sum(counter.values())

        for pain, freq in counter.most_common():
            rows.append({
                "product_group": product,
                "pain": pain,
                "frequency": freq,
                "percentage": round(freq / total, 4)
            })

    result_df = pd.DataFrame(rows)
    os.makedirs("reports/tables", exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Salvo em: {OUTPUT_PATH}")
    print("Finalizado!")


if __name__ == "__main__":
    main()
