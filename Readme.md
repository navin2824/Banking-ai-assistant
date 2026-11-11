# Banking AI Assistant
### Domain-Aligned AI using Nugen Intelligence and Retrieval-Augmented Generation

An intelligent banking question-answering application built using **Nugen Domain-Aligned AI**, Retrieval-Augmented Generation (RAG), TF-IDF document retrieval and Streamlit.

The application answers banking-related questions using a curated knowledge base covering Indian banking fundamentals, RBI regulatory concepts, asset quality and financial risk. It also identifies certain time-sensitive questions and abstains when its static knowledge base cannot verify current information.

---

## 1. Project Overview

The Banking AI Assistant demonstrates how domain-specific knowledge can be combined with a domain-aligned language model to build a focused question-answering application.

The system integrates:

- Banking knowledge documents.
- Document processing and text chunking.
- TF-IDF-based information retrieval.
- Nugen domain alignment and model deployment.
- Context-grounded inference.
- Unsupported and time-sensitive question detection.
- Streamlit-based conversational interface.

## 2. Objectives

- Build a banking-focused AI assistant.
- Prepare and organize domain-specific banking knowledge.
- Retrieve relevant information before generating answers.
- Integrate a deployed Nugen domain-aligned model.
- Reduce unsupported responses through retrieval checks and abstention.
- Evaluate retrieval quality using information-retrieval metrics.

## 3. System Architecture

**Workflow:**

1. Banking documents are collected and organized.
2. Documents are processed and divided into text chunks.
3. TF-IDF vectorization converts the chunks into numerical representations.
4. A user submits a banking question.
5. The retrieval engine identifies relevant chunks using cosine similarity.
6. A support assessment determines whether sufficient evidence is available.
7. Retrieved context is supplied to the deployed Nugen domain-aligned model.
8. The model generates a response.
9. The Streamlit application displays the answer and retrieved sources.

Time-sensitive questions requiring current information are rejected when no live-data source is available.

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| AI model platform | Nugen Intelligence |
| Base model | Meta Llama 3.2 3B Instruct |
| Retrieval method | TF-IDF |
| Similarity measurement | Cosine similarity |
| Machine learning | Scikit-learn |
| Numerical processing | NumPy, SciPy |
| User interface | Streamlit |
| API integration | Python Requests |
| Configuration | Python-dotenv |

## 5. Knowledge Base

The project contains three domain-specific documents:

| Document | Coverage |
|---|---|
| `01_banking_fundamentals.txt` | Banking fundamentals, deposits, lending, EMI, NPAs and customer protection |
| `02_rbi_regulatory_and_asset_quality.txt` | RBI responsibilities, prudential regulation, asset classification and provisioning |
| `03_financial_risk_and_reliability.txt` | Credit risk, liquidity risk, operational risk and risk governance |

The current knowledge base contains **31 indexed chunks**.

## 6. Nugen Domain Alignment

The project uses Nugen Intelligence to create and deploy a banking-focused aligned model.

Alignment workflow:

1. Access the Nugen platform.
2. Create a banking and finance domain-alignment project.
3. Upload the prepared banking knowledge documents.
4. Run the alignment process.
5. Deploy the aligned model.
6. Retrieve the deployed model ID and inference endpoint.
7. Integrate the deployed model into the Python application.

**Model configuration:**

- Base model: `meta-llama/Llama-3.2-3B-Instruct`
- Deployed model ID: Configured through environment variables.
- Inference interface: Nugen Chat Completions API.

The application combines the aligned model with a separate retrieval pipeline. The retrieved evidence is supplied as context during inference.

## 7. Retrieval-Augmented Generation

The retrieval pipeline uses TF-IDF to identify relevant banking information.

**Current retrieval configuration:**

| Parameter | Value |
|---|---:|
| Chunk size | 1,200 characters |
| Chunk overlap | 150 characters |
| Top K | 3 |
| Minimum similarity | 0.01 |
| Confidence threshold | 0.05 |
| Indexed chunks | 31 |

The retrieval process uses cosine similarity to rank candidate chunks.

A support-assessment layer checks whether the question requires live information and whether the retrieved evidence meets the configured threshold.

Note: The similarity threshold is a heuristic, not a calibrated probability of answer correctness.

## 8. Application Features

- Banking-focused conversational question answering.
- Retrieval of relevant document sections.
- Display of retrieved banking sources.
- Nugen API integration.
- Domain-aligned model inference.
- Time-sensitive question detection.
- Abstention when static documents cannot verify current information.
- Streamlit-based user interface.

## 9. Evaluation Results

The initial automated evaluation contains six answerable banking questions and five unsupported or time-sensitive questions.

### Answerable Question Evaluation

| Metric | Result |
|---|---:|
| Questions tested | 6 |
| Precision@3 | 0.444 |
| Hit Rate@3 | 1.000 |
| Mean Reciprocal Rank | 0.750 |

### Unsupported Question Evaluation

| Metric | Result |
|---|---:|
| Questions tested | 5 |
| Rejected questions | 5 |
| Rejection rate | 1.000 |

### Interpretation

- Hit Rate@3 of 1.000 indicates that at least one relevant chunk was retrieved for each of the six answerable test questions.
- Precision@3 of 0.444 indicates that some irrelevant chunks are also appearing in the top-three results.
- MRR of 0.750 measures the ranking position of the first relevant result.
- The rejection rate of 1.000 indicates that all five selected time-sensitive questions were rejected by the current support-assessment logic.

**Evaluation limitations:** These are preliminary results from a small, manually prepared test set. The rejection rate does not establish general hallucination prevention. Retrieval relevance labels are manually defined, and further testing is required to assess generalization and answer faithfulness.

## 10. Project Structure

```text
banking-ai-assistant/
│
├── data/
│   ├── raw/
│   │   ├── 01_banking_fundamentals.txt
│   │   ├── 02_rbi_regulatory_and_asset_quality.txt
│   │   └── 03_financial_risk_and_reliability.txt
│   │
│   └── processed/
│       └── chunks.json
│
├── src/
│   ├── __init__.py
│   ├── rag.py
│   ├── evaluation.py
│   └── test_rag.py
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 11. Installation and Setup

### Prerequisites

- Python 3.10 or later.
- A Nugen Intelligence account.
- A valid Nugen API key.
- A deployed domain-aligned model.

### Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd banking-ai-assistant
```

### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

### Configure environment variables

Create a `.env` file in the project root:

```env
NUGEN_API_KEY=your_api_key
NUGEN_ENDPOINT=https://api.nugen.in/api/v3/inference/chat/completions
NUGEN_MODEL_ID=your_deployed_model_id
```

Never commit `.env` or expose API credentials publicly.

### Run the application

```powershell
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## 12. Run Evaluation

Execute the automated retrieval evaluation:

```powershell
python -m src.evaluation
```

The evaluation reports:

- Precision@3
- Hit Rate@3
- Mean Reciprocal Rank
- Unsupported question rejection rate

## 13. Future Improvements

- Implement hybrid retrieval using lexical and semantic search.
- Improve ranking of domain-specific document sections.
- Expand the evaluation dataset.
- Add answer faithfulness and citation correctness evaluation.
- Integrate verified live-data sources for time-sensitive banking queries.
- Add more banking and financial domain documents.
- Improve retrieval confidence calibration.
- Introduce automated regression testing.

## 14. Responsible AI Considerations

The application is designed to avoid presenting unsupported or outdated information as verified fact.

Its current safeguards include:

- Evidence retrieval before inference.
- Retrieval support assessment.
- Detection of selected time-sensitive question patterns.
- Abstention when current information cannot be verified.

This implementation is an educational prototype and is not intended to provide financial, investment, credit approval or regulatory advice.

## 15. Acknowledgements

- Nugen Intelligence — Domain-Aligned AI platform.
- Scikit-learn — TF-IDF vectorization and similarity-based retrieval.
- Streamlit — Interactive application interface.

---

**Project:** Banking AI Assistant  
**Domain:** Banking and Financial Services  
**AI Platform:** Nugen Intelligence  
**Interface:** Streamlit  
**Language:** Python