# Policy RAG (Modular LLM Pipeline for Policy Document Analysis)

A modular **Retrieval-Augmented Generation (RAG)** system for extracting, retrieving, querying, and analyzing policy documents using **Python, Google Gemini, and FAISS**.

This project was developed to explore how LLM-based workflows can be structured beyond a monolithic notebook into a **reproducible, modular pipeline** that separates data ingestion, semantic retrieval, LLM inference, and downstream text analysis.

The system is designed around a simple objective:

> **Turn unstructured policy documents into a searchable knowledge base and use an LLM to generate context-aware answers from retrieved evidence.**

---

## Project Overview

Policy documents are often long, heterogeneous, and difficult to analyze manually. This project implements an end-to-end workflow that combines traditional NLP techniques with LLM-based retrieval and generation.

```text
PDF / CSV Documents
        │
        ▼
┌─────────────────┐
│   ingest.py     │
│ Extract + Clean │
│ + Chunk Text    │
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ build_vectorstore.py │
│ Embedding + FAISS    │
└──────────┬───────────┘
           │
           ▼
┌─────────────────┐
│   rag_query.py  │
│ Retrieval + LLM │
│     Inference   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   analysis.py   │
│ NLP Analysis    │
│ Sentiment /     │
│ Frequency /     │
│ Co-occurrence   │
└─────────────────┘
```

The pipeline separates each stage so that individual components can be tested, reused, and improved independently.

---

## Key Capabilities

### 1. Document Ingestion

`ingest.py` handles the transformation of raw documents into structured text suitable for downstream retrieval.

Supported inputs include:

* PDF documents
* CSV datasets
* Text cleaning and normalization
* Document chunking
* Metadata preparation

The objective is to create consistent text units that can be embedded and retrieved efficiently.

---

### 2. Semantic Retrieval with FAISS

`build_vectorstore.py` transforms document chunks into vector representations and builds a **FAISS-based vector index**.

The retrieval process enables semantic matching rather than relying only on exact keyword overlap.

For example, a query about:

> "government strategies for reducing poverty"

can retrieve passages discussing related concepts even when the exact wording differs.

This creates the retrieval layer required by the RAG architecture.

---

### 3. LLM-Powered Question Answering

`rag_query.py` implements the inference layer using **Google Gemini**.

The workflow is:

```text
User Query
    ↓
Query Embedding
    ↓
Semantic Retrieval
    ↓
Relevant Document Chunks
    ↓
Context Construction
    ↓
Gemini LLM
    ↓
Context-Aware Response
```

Instead of asking the LLM to answer solely from its pretrained knowledge, the system provides retrieved document context as part of the generation process.

This makes the system more suitable for domain-specific document analysis.

---

### 4. Downstream NLP Analysis

`analysis.py` provides additional analytical capabilities around the document corpus.

Current components include:

* Sentiment analysis
* Word frequency analysis
* Wordcloud generation
* Co-occurrence analysis
* Network-based visualization of term relationships

These components complement the RAG workflow by providing exploratory analysis of the underlying corpus.

---

## Modular Architecture

The original implementation was developed in an experimental notebook environment.

As the project grew, repeated functions and analysis cells made the workflow difficult to maintain. The code was therefore refactored into reusable Python modules.

| Module                             | Responsibility                                              |
| ---------------------------------- | ----------------------------------------------------------- |
| `ingest.py`                        | Document extraction, cleaning, and chunking                 |
| `build_vectorstore.py`             | Embedding generation and FAISS index management             |
| `rag_query.py`                     | Retrieval, Gemini integration, and RAG inference            |
| `analysis.py`                      | Sentiment, frequency, wordcloud, and co-occurrence analysis |
| `LLM_PREANALISIS_REFACTORED.ipynb` | Interactive demo and experimentation                        |

This separation makes it possible to modify one component without rewriting the entire pipeline.

---

## Why This Project

The project focuses on a practical LLM engineering problem:

**How can an LLM be connected to domain-specific documents in a way that is modular, reproducible, and easier to maintain?**

Rather than treating an LLM as an isolated chatbot, the project explores an end-to-end workflow involving:

* Data preparation
* Text representation
* Vector retrieval
* Context construction
* LLM inference
* NLP-based analysis
* Modular Python architecture
* Credential management

This reflects an interest in building **usable LLM workflows rather than only experimenting with prompts inside a notebook.**

---

## Evaluation & Reliability

The current project focuses primarily on the **retrieval and inference pipeline** rather than a full production evaluation framework.

Potential evaluation dimensions for further development include:

### Retrieval evaluation

Measure whether the retriever returns relevant document passages for a given query.

Possible metrics:

* Recall@K
* Precision@K
* MRR

### Generation evaluation

Evaluate whether generated answers are:

* Relevant to the question
* Grounded in retrieved context
* Consistent with source documents
* Free from unsupported claims

A future evaluation layer could introduce a curated question–answer benchmark and compare retrieval and generation performance across different configurations.

---

## Security & Credential Management

API credentials are intentionally separated from source code.

The system uses environment variables rather than hard-coded API keys.

Create a local `.env` file:

```bash
GOOGLE_API_KEY=your_gemini_api_key
```

The `.env` file should **never be committed to version control**.

A `.env.example` template is provided so that the required configuration can be reproduced without exposing credentials.

> Any previously exposed API key should be revoked and replaced before using the project.

---

## Tech Stack

**Programming**

* Python

**LLM**

* Google Gemini

**Retrieval**

* FAISS
* Text embeddings

**NLP**

* Sentiment analysis
* Word frequency analysis
* Wordcloud
* Co-occurrence analysis

**Development**

* Jupyter Notebook
* Modular Python scripts
* Environment-based configuration

---

## Project Structure

```text
Policy-RAG/
│
├── ingest.py
├── build_vectorstore.py
├── rag_query.py
├── analysis.py
│
├── LLM_PREANALISIS_REFACTORED.ipynb
│
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
└── data/
    └── [local policy documents]
```

Sensitive or private datasets should remain outside version control.

---

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the API key

Create `.env` from `.env.example`:

```bash
GOOGLE_API_KEY=your_gemini_api_key
```

### 3. Prepare the document corpus

Place PDF or CSV files in the configured data directory.

### 4. Build the vector store

Run the ingestion and vector-store pipeline:

```bash
python ingest.py
python build_vectorstore.py
```

### 5. Run RAG queries

Use:

```bash
python rag_query.py
```

or run the interactive notebook:

```text
LLM_PREANALISIS_REFACTORED.ipynb
```

### 6. Run downstream analysis

Use the functions provided in:

```text
analysis.py
```

for sentiment, frequency, wordcloud, and co-occurrence analysis.

---

## What I Learned

This project strengthened my understanding of building LLM applications as a **pipeline rather than a single model call**.

Key areas explored:

1. Preparing unstructured documents for LLM applications
2. Designing semantic retrieval using vector embeddings
3. Connecting retrieval results to an LLM generation step
4. Structuring RAG components into reusable Python modules
5. Separating exploratory analysis from reusable application logic
6. Managing API credentials securely
7. Thinking about retrieval and generation as separate components that can be evaluated independently

The project also provides a foundation for further exploration of **LLM evaluation, prompt optimization, model serving, and production-oriented deployment**.

---

## Current Scope & Future Improvements

This project is intentionally positioned as a **learning and engineering prototype**, not as a claim of production deployment experience.

Potential next steps include:

* Building a formal retrieval evaluation dataset
* Adding automated offline RAG evaluation
* Comparing different chunking and retrieval strategies
* Experimenting with prompt optimization
* Adding experiment tracking
* Testing latency and throughput
* Exploring local LLM serving with vLLM
* Exploring parameter-efficient fine-tuning approaches such as LoRA/QLoRA
* Adding monitoring for production deployment

These extensions would allow the current RAG prototype to evolve toward a more complete **production-oriented LLM workflow**.

---

## Relevance to LLM / AI Engineering

This project demonstrates hands-on experience with several foundations relevant to LLM engineering:

| Role Requirement             | Project Evidence                                                                 |
| ---------------------------- | -------------------------------------------------------------------------------- |
| Strong Python fundamentals   | Modular Python implementation and reusable functions                             |
| Build something with LLMs    | Gemini-powered RAG application                                                   |
| Understand LLM concepts      | Retrieval → context → generation workflow                                        |
| Work with real-world data    | PDF/CSV ingestion and policy-document analysis                                   |
| Data science techniques      | NLP, embeddings, semantic retrieval, sentiment and co-occurrence analysis        |
| Reproducible workflows       | Modular scripts and environment-based configuration                              |
| Production-oriented thinking | Separation of pipeline components, credential management, and planned evaluation |

The project does **not** currently claim hands-on production deployment, vLLM, LoRA/QLoRA fine-tuning, or A/B testing experience. These are identified as natural areas for further development.
