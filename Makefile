install:
	python -m pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt

spacy-model:
	python -m spacy download en_core_web_sm

run-ingestion:
	python -m src.ingestion.read_large_csv

run-normalize:
	python -m src.preprocessing.normalize_columns

run-clean:
	python -m src.preprocessing.clean_text

run-label:
	python -m src.labeling.sentiment_rules

run-split:
	python -m src.preprocessing.split_data

test:
	pytest tests/
