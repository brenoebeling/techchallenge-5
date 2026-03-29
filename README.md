# Reclamações Financeiras NLP — Análise de Sentimento e Pontos de Dor do Cliente :contentReference[oaicite:0]{index=0}

## 1. Visão Geral

Este projeto entrega um pipeline completo de NLP para classificar o sentimento em reclamações financeiras e identificar os principais pontos de dor dos clientes por categoria de produto.

A solução foi desenhada com base em um dataset real de grande escala, combinando engenharia de dados, pré-processamento de texto, aprendizado supervisionado, deep learning e análise orientada a negócios.

Este repositório foi desenvolvido como o Tech Challenge da Fase 5 da Pós Tech em Data Analytics, com foco na construção de um modelo de Deep Learning para classificação de sentimento em reclamações financeiras e extração dos principais temas negativos por categoria de produto financeiro.

## 2. Problema de Negócio

Instituições financeiras recebem volumes massivos de reclamações não estruturadas.

O desafio não é apenas ler esses textos, mas transformá-los em sinais analíticos estruturados.

Este projeto responde duas perguntas:

Podemos classificar automaticamente o sentimento das reclamações como positivo ou negativo?  
Quais são os principais pontos de dor relatados pelos clientes em cada categoria de produto financeiro?

O objetivo é apoiar decisões melhores em experiência do cliente, melhoria de produtos, priorização operacional e monitoramento de riscos.

## 3. Requisitos do Desafio

De acordo com o desafio da Fase 5, a entrega deve incluir:

- pré-processamento de texto  
- técnicas de vetorização  
- criação da variável alvo de sentimento  
- treinamento de modelo de Deep Learning  
- análise dos principais pontos de dor por produto  
- visualizações para interpretação  
- repositório no GitHub com pipeline NLP e modelo  

Este repositório foi estruturado para atender todos esses requisitos.

## 4. Dataset

### Fonte

Base pública de reclamações do CFPB

### Principais campos utilizados

- Produto  
- Subproduto  
- Problema  
- Subproblema  
- Narrativa da reclamação  
- Empresa  
- Estado  
- Canal de envio  
- Resposta da empresa ao consumidor  
- Resposta no prazo?  
- Consumidor contestou?  
- ID da reclamação  

O dataset inclui narrativas e dados contextuais que permitem tanto modelagem de sentimento quanto análise de pontos de dor por categoria.

## 5. Escala dos Dados

Resumo da execução:

- Linhas totais: ~14,1 milhões  
- Linhas relevantes: ~3,75 milhões  
- Textos limpos: ~3,73 milhões  
- Reclamações negativas analisadas: ~2,48 milhões  

Devido à escala, a ingestão foi feita com leitura em chunks e conversão para Parquet para melhor performance.

## 6. Pipeline do Projeto

### 6.1 Ingestão

- leitura de CSV grande em chunks  
- seleção de colunas relevantes  
- remoção de duplicados  
- exportação para Parquet  

### 6.2 Pré-processamento

- normalização para minúsculas  
- limpeza com regex  
- remoção de pontuação  
- remoção de stopwords  
- filtragem de textos inválidos  

### 6.3 Criação do Target

Foi criada a coluna **sentiment** usando weak supervision baseada em:

- resposta da empresa  
- disputa do consumidor  

### 6.4 Split

- divisão estratificada  
- treino / validação / teste  
- preservação de distribuição  
- prevenção de vazamento  

### 6.5 Modelo Base

- TF-IDF  
- Regressão Logística  

### 6.6 Deep Learning

- Tokenização  
- Padding  
- Embedding  
- LSTM  
- Saída com sigmoid  

### 6.7 Análise de Pontos de Dor

- filtro de reclamações negativas  
- agrupamento por produto  
- análise de frequência de termos  
- classificação estruturada  
- visualizações executivas  

## 7. Resultados

### 7.1 Modelo Base

- Accuracy: 0.71  
- Precision: 0.59  
- Recall: 0.45  
- F1: 0.51  

### 7.2 LSTM

- Accuracy: 0.7147  
- Precision: 0.5915  
- Recall: 0.4855  
- F1: 0.5333  

**Insight:**  
O LSTM melhorou a identificação da classe minoritária, reduzindo falsos negativos.

## 8. Principais Insights

Produtos com maior dor:

- credit_reporting  
- debt_collection  
- loans  
- credit_card  
- banking  
- payments  

Pontos de dor:

- credit_reporting → score de crédito  
- debt_collection → disputa de dívida  
- loans → problemas de pagamento  
- credit_card → score  
- banking → acesso à conta  
- payments → atendimento  

**Insight executivo:**  
Os principais problemas não são apenas operacionais, mas estruturais (crédito e dívida).

## 9. Estrutura do Repositório

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

## 10. Scripts Principais

### Ingestão

src.ingestion.read_large_csv

### Pré-processamento

src.preprocessing.clean_text  
src.preprocessing.split_data

### Rotulagem

src.labeling.sentiment_rules

### Modelo Base

src.models.train_tfidf_logreg

### Deep Learning

src.models.train_lstm

### Análise de Pontos de Dor

src.analysis.pain_points  
src.analysis.pain_points_structured  
src.analysis.visualize_pain_points

## 11. Como Executar

### 1. Criar ambiente virtual

python3.12 -m venv .venv  
source .venv/bin/activate  

### 2. Instalar dependências

python -m pip install --upgrade pip setuptools wheel  
pip install -r requirements.txt  

### 3. Rodar pipeline

python -m src.ingestion.read_large_csv  
python -m src.preprocessing.clean_text  
python -m src.labeling.sentiment_rules  
python -m src.preprocessing.split_data  
python -m src.models.train_tfidf_logreg  
python -m src.models.train_lstm  
python -m src.analysis.pain_points  
python -m src.analysis.pain_points_structured  
python -m src.analysis.visualize_pain_points  

## 12. Stack Tecnológico

- Python  
- Pandas  
- NumPy  
- scikit-learn  
- TensorFlow / Keras  
- Matplotlib  
- WordCloud  
- PyArrow  

## 13. Limitações

- Target criado via weak supervision  
- Dados ruidosos  
- Modelos baseados apenas em texto  
- Possível melhoria com BERT  

## 14. Melhorias Futuras

- tratar desbalanceamento  
- ajustar thresholds  
- embeddings pré-treinados  
- fine-tuning com BERT  
- dashboard  
- integração com AWS  

## 15. Conclusão

O projeto mostra como transformar dados não estruturados em inteligência analítica, conectando NLP com valor de negócio.

## 16. Autor

Breno Leoni Ebeling  
Economista | Analytics | NLP | Serviços Financeiros
