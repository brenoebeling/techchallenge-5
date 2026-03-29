# Financial Complaints NLP — Sentiment Analysis and Customer Pain Points

## 1. Overview

This project delivers an end-to-end NLP pipeline to classify sentiment in financial complaints and identify the main customer pain points by product category.

The solution was designed around a real-world large-scale complaint dataset, combining data engineering, text preprocessing, supervised learning, deep learning, and business-oriented analysis.

This repository was developed as the Phase 5 Tech Challenge of the Pós Tech in Data Analytics program, focused on building a Deep Learning model for sentiment classification in financial complaints and 
extracting the main negative themes by financial product category.

---

## 2. Business Problem

Financial institutions receive massive volumes of unstructured customer complaints.  
The challenge is not only to read these texts, but to transform them into structured analytical signals.

This project addresses two questions:

1. Can we automatically classify complaint sentiment as positive or negative?
2. What are the most common pain points reported by customers in each financial product category?

The goal is to support better decision-making for customer experience, product improvement, operational prioritization, and risk monitoring.

---

## 3. Challenge Requirements

According to the Phase 5 challenge, the deliverable should include:

- text preprocessing;
- text vectorization techniques;
- creation of the target sentiment variable;
- training of a Deep Learning model;
- analysis of the main customer pain points by product;
- visualizations to support interpretation;
- GitHub repository with the NLP pipeline and classification model.  

This repository was structured to meet all those requirements.

---

## 4. Dataset

### Source
- CFPB public complaint database

### Main fields used
- `Product`
- `Sub-product`
- `Issue`
- `Sub-issue`
- `Consumer complaint narrative`
- `Company`
- `State`
- `Submitted via`
- `Company response to consumer`
- `Timely response?`
- `Consumer disputed?`
- `Complaint ID`

The dataset includes complaint narratives and contextual fields that allow both sentiment modeling and pain-point analysis by product category.

---

## 5. Data Scale

Pipeline execution summary:

- Total rows read: ~14.1 million
- Filtered relevant rows: ~3.75 million
- Cleaned complaint texts: ~3.73 million
- Negative complaints analyzed for pain points: ~2.48 million

Because of the scale, the ingestion step was implemented using chunked reading and Parquet conversion for better performance and reproducibility.

---

## 6. Project Pipeline

### 6.1 Data Ingestion
- large CSV ingestion in chunks
- relevant column selection
- duplicate removal
- export to Parquet

### 6.2 Text Preprocessing
- lowercase normalization
- regex cleaning
- punctuation removal
- stopword removal
- invalid/short narrative filtering

### 6.3 Target Creation
A target column named `sentiment` was created using a weak supervision strategy based on operational signals available in the dataset, especially:

- `Company response to consumer`
- `Consumer disputed?`

This approach allowed scalable labeling for millions of complaints.

### 6.4 Train / Validation / Test Split
- stratified split
- train / validation / test separation
- preserved class distribution
- avoided leakage

### 6.5 Baseline Model
- TF-IDF
- Logistic Regression

### 6.6 Deep Learning Model
- Tokenizer + padded sequences
- Embedding layer
- LSTM
- Dense output with sigmoid activation

### 6.7 Pain Point Analysis
- filtering negative complaints
- grouping by product
- term-frequency analysis
- structured pain classification
- executive visualizations by product group

---

## 7. Modeling Results

## 7.1 Baseline — TF-IDF + Logistic Regression

### Test metrics
- Accuracy: **0.71**
- Precision (class 1): **0.59**
- Recall (class 1): **0.45**
- F1-score (class 1): **0.51**

This baseline was stable between validation and test and served as a strong interpretability benchmark.

---

## 7.2 Deep Learning — LSTM

### Test metrics
- Accuracy: **0.7147**
- Precision (class 1): **0.5915**
- Recall (class 1): **0.4855**
- F1-score (class 1): **0.5333**

### Key takeaway
The LSTM did not dramatically increase overall accuracy, but it improved the identification of the minority class, especially by reducing false negatives. This is consistent with the role of recurrent 
networks in capturing sequential context better than sparse lexical representations.

---

## 8. Main Business Insights

After analyzing negative complaints, the following pattern emerged:

### Product groups with highest concentration of negative pain points
- `credit_reporting`
- `debt_collection`
- `loans`
- `credit_card`
- `banking`
- `payments`

### Main pain points by group
- **credit_reporting** → `credit_score`
- **debt_collection** → `debt_dispute`
- **loans** → `payment_issues`
- **credit_card** → `credit_score`
- **banking** → `account_access`
- **payments** → `customer_service`

### Executive interpretation
The most relevant finding is that the largest concentration of customer pain is not simply related to service quality, but to structural issues involving credit information and debt disputes.

Operational categories such as banking and payments showed more distributed pain patterns, linked to access, support, and customer experience.

---

## 9. Repository Structure

data/
  raw/
  interim/
  processed/

src/
  ingestion/
  preprocessing/
  labeling/
  features/
  models/
  analysis/
  aws_pipeline/

reports/
  figures/
  tables/

models_artifacts/
notebooks/
tests/

---

## 10. Main Scripts
### Ingestion
src.ingestion.read_large_csv

### Preprocessing
src.preprocessing.clean_text
src.preprocessing.split_data

### Labeling
src.labeling.sentiment_rules

### Baseline model

src.models.train_tfidf_logreg

### Deep learning model

src.models.train_lstm

### Pain-point analysis

src.analysis.pain_points
src.analysis.pain_points_structured
src.analysis.visualize_pain_points

## 11. How to Run

### 1. Create and activate virtual environment

python3.12 -m venv .venv
source .venv/bin/activate

### 2. Install dependencies

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

### 3. Run pipeline

python -m src.ingestion.read_large_csv
python -m src.preprocessing.clean_text
python -m src.labeling.sentiment_rules
python -m src.preprocessing.split_data
python -m src.models.train_tfidf_logreg
python -m src.models.train_lstm
python -m src.analysis.pain_points
python -m src.analysis.pain_points_structured
python -m src.analysis.visualize_pain_points

## 12. Tech Stack

Python
Pandas
NumPy
scikit-learn
TensorFlow / Keras
Matplotlib
WordCloud
PyArrow

## 13. Limitations

The target variable was created using weak supervision, not manual ground truth labeling.
Complaint narratives are noisy and naturally ambiguous.
The baseline and LSTM operate only on text, while the labeling strategy uses structured proxy signals from operational fields.
More advanced contextual models such as BERT may further improve semantic capture.

## 14. Future Improvements

improve class-imbalance treatment
evaluate threshold tuning
test pretrained embeddings
fine-tune BERT on a representative sample
build a dashboard for continuous complaint monitoring
integrate storage and querying with AWS S3 and Athena

## 15. Conclusion

This project demonstrates how to transform large-scale unstructured financial complaints into analytical intelligence.
Beyond sentiment classification, it organizes customer narratives into actionable pain-point structures, connecting NLP techniques with practical business value in the financial sector.

## 16. Author

Breno Leoni Ebeling
Economist | Analytics | NLP | Financial Services


