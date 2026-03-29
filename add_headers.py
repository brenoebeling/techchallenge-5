from pathlib import Path

HEADERS = {
    "src/ingestion/read_large_csv.py": '''"""
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

''',

    "src/preprocessing/normalize_columns.py": '''"""
Arquivo: src/preprocessing/normalize_columns.py

Objetivo:
Padronizar nomes de colunas, tipos e estrutura do dataset para uso consistente
nas próximas etapas do pipeline.

Fonte de dados:
Arquivo intermediário gerado a partir do CSV local

Arquivo de entrada:
data/interim/complaints_selected.parquet

Arquivo gerado:
data/interim/complaints_normalized.parquet
"""

''',

    "src/preprocessing/clean_text.py": '''"""
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

''',
}


def already_has_docstring(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith('"""') or stripped.startswith("'''")


def main() -> None:
    for file_path_str, header in HEADERS.items():
        path = Path(file_path_str)

        if not path.exists():
            print(f"[AVISO] Arquivo não encontrado: {file_path_str}")
            continue

        original = path.read_text(encoding="utf-8")

        if already_has_docstring(original):
            print(f"[OK] Já possui cabeçalho: {file_path_str}")
            continue

        path.write_text(header + original, encoding="utf-8")
        print(f"[ADICIONADO] {file_path_str}")


if __name__ == "__main__":
    main()
