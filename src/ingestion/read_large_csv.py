"""
Arquivo: src/ingestion/read_large_csv.py

Objetivo:
Ler o CSV bruto de reclamações financeiras em chunks, selecionar colunas relevantes
e salvar uma versão intermediária mais leve para o pipeline.

Fonte de dados:
CSV local do projeto (base principal do desafio)

Arquivo de entrada:
data/raw/complaints.csv

Arquivo gerado:
data/interim/complaints_selected.parquet
"""

from pathlib import Path
import sys
import pandas as pd


# =========================
# CONFIGURAÇÃO DO PIPELINE
# =========================
INPUT_CSV = Path("data/raw/complaints.csv")
OUTPUT_PARQUET = Path("data/interim/complaints_selected.parquet")

CHUNK_SIZE = 100_000

# Colunas escolhidas para o projeto.
# Mantemos apenas o que é útil para:
# - sentimento
# - dores por produto
# - análises complementares
RAW_COLUMNS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company",
    "State",
    "Submitted via",
    "Company response to consumer",
    "Timely response?",
    "Consumer disputed?",
    "Complaint ID",
]

# Alguns CSVs podem vir com pequenas variações de nome.
# Aqui centralizamos aliases para tornar a leitura mais robusta.
COLUMN_ALIASES = {
    "Consumer Complaint Narrative": "Consumer complaint narrative",
    "Consumer complaint narrative": "Consumer complaint narrative",
    "Complaint ID": "Complaint ID",
    "Product": "Product",
    "Sub-product": "Sub-product",
    "Issue": "Issue",
    "Sub-issue": "Sub-issue",
    "Company": "Company",
    "Company response to consumer": "Company response to consumer",
    "State": "State",
    "Submitted via": "Submitted via",
    "Timely response?": "Timely response?",
    "Consumer disputed?": "Consumer disputed?",
    "Date received": "Date received",
}


def log(message: str) -> None:
    print(f"[read_large_csv] {message}")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_header_names(columns: list[str]) -> list[str]:
    """
    Padroniza nomes de colunas quando houver pequenas variações.
    """
    normalized = []
    for col in columns:
        normalized.append(COLUMN_ALIASES.get(col, col))
    return normalized


def validate_input_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de entrada não encontrado: {path}\n"
            f"Coloque seu CSV em: {INPUT_CSV}"
        )


def discover_available_columns(sample_path: Path) -> list[str]:
    """
    Lê apenas o cabeçalho para descobrir colunas disponíveis.
    """
    sample = pd.read_csv(sample_path, nrows=0)
    sample.columns = normalize_header_names(sample.columns.tolist())
    return sample.columns.tolist()


def select_existing_columns(available_columns: list[str]) -> list[str]:
    selected = [col for col in RAW_COLUMNS if col in available_columns]

    if "Consumer complaint narrative" not in selected:
        raise ValueError(
            "A coluna 'Consumer complaint narrative' não foi encontrada no CSV. "
            "Sem ela, o pipeline de NLP não consegue seguir."
        )

    if "Product" not in selected:
        raise ValueError(
            "A coluna 'Product' não foi encontrada no CSV. "
            "Sem ela, a análise por categoria de produto fica comprometida."
        )

    return selected


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Limpezas mínimas da etapa de ingestão.
    Não faz pré-processamento textual profundo aqui.
    """
    # Remove espaços extras de nomes de coluna
    chunk.columns = [c.strip() for c in chunk.columns]
    chunk.columns = normalize_header_names(chunk.columns.tolist())

    # Mantém somente linhas com narrativa útil
    text_col = "Consumer complaint narrative"
    chunk[text_col] = chunk[text_col].astype("string").str.strip()

    chunk = chunk[chunk[text_col].notna()]
    chunk = chunk[chunk[text_col] != ""]
    chunk = chunk[chunk[text_col].str.lower() != "nan"]

    # Remove duplicidade por complaint id, se existir
    if "Complaint ID" in chunk.columns:
        chunk = chunk.drop_duplicates(subset=["Complaint ID"])

    return chunk


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduz memória nos textos categóricos mais comuns.
    """
    categorical_candidates = [
        "Product",
        "Sub-product",
        "Issue",
        "Sub-issue",
        "Company",
        "State",
        "Submitted via",
        "Company response to consumer",
        "Timely response?",
        "Consumer disputed?",
    ]

    for col in categorical_candidates:
        if col in df.columns:
            df[col] = df[col].astype("category")

    if "Complaint ID" in df.columns:
        # Usa string por segurança; alguns datasets misturam formatos
        df["Complaint ID"] = df["Complaint ID"].astype("string")

    if "Date received" in df.columns:
        df["Date received"] = pd.to_datetime(
            df["Date received"], errors="coerce"
        )

    return df


def read_and_process_csv() -> pd.DataFrame:
    validate_input_file(INPUT_CSV)

    log(f"Lendo cabeçalho do arquivo: {INPUT_CSV}")
    available_columns = discover_available_columns(INPUT_CSV)
    selected_columns = select_existing_columns(available_columns)

    log(f"Colunas disponíveis no CSV: {len(available_columns)}")
    log(f"Colunas selecionadas para ingestão: {selected_columns}")

    chunks = []
    total_rows_read = 0
    total_rows_kept = 0

    chunk_iterator = pd.read_csv(
        INPUT_CSV,
        usecols=selected_columns,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for i, chunk in enumerate(chunk_iterator, start=1):
        rows_before = len(chunk)
        total_rows_read += rows_before

        chunk = clean_chunk(chunk)
        rows_after = len(chunk)
        total_rows_kept += rows_after

        chunks.append(chunk)

        log(
            f"Chunk {i}: lidas={rows_before:,} | mantidas={rows_after:,} | "
            f"acumulado_mantidas={total_rows_kept:,}"
        )

    if not chunks:
        raise ValueError(
            "Nenhum dado válido foi mantido após a ingestão. "
            "Verifique o CSV e a coluna 'Consumer complaint narrative'."
        )

    log("Concatenando chunks em um único DataFrame...")
    df = pd.concat(chunks, ignore_index=True)

    # Remove duplicados novamente no consolidado final
    if "Complaint ID" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["Complaint ID"])
        after = len(df)
        log(f"Duplicados removidos no consolidado final: {before - after:,}")

    df = optimize_dtypes(df)

    log(f"Total de linhas lidas: {total_rows_read:,}")
    log(f"Total de linhas mantidas: {len(df):,}")

    return df


def save_output(df: pd.DataFrame) -> None:
    ensure_parent_dir(OUTPUT_PARQUET)
    log(f"Salvando parquet em: {OUTPUT_PARQUET}")
    df.to_parquet(OUTPUT_PARQUET, index=False)
    log("Arquivo parquet salvo com sucesso.")


def main() -> None:
    try:
        df = read_and_process_csv()
        save_output(df)

        log("Resumo final:")
        log(f"Linhas finais: {len(df):,}")
        log(f"Colunas finais: {list(df.columns)}")
        log("Etapa de ingestão concluída.")
    except Exception as exc:
        log(f"ERRO: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
