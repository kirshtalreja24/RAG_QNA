# RAG_Qna

A retrieval-augmented generation (RAG) QA system that combines dense retrieval via FAISS, sparse retrieval via BM25, hybrid fusion, and an LLM-based answer generation pipeline.

## Project Overview

This repository contains a RAG pipeline built around Natural Questions data and a small Wikipedia-derived knowledge base. It supports:

- Data ingestion and preprocessing
- Knowledge base creation from text chunks
- Dense retrieval with FAISS
- Sparse retrieval with BM25
- Hybrid retrieval and reranking
- Question answering via a generation module
- Flask web API and interactive CLI interface

## Repository Structure

- `app/`
  - `app.py` — CLI and Flask web server entrypoint
  - `static/` — frontend assets for the web UI
- `data/`
  - `raw/` — raw knowledge/data sources
  - `processed/` — prepared indexes and metadata
- `src/`
  - `data/` — dataset loading, preprocessing, and embedding/index utilities
  - `generation/` — prompt creation and answer generation logic
  - `retrieval/` — retrieval pipeline and search logic
- `pipeline.MD` — overview of the pipeline design
- `requirements.txt` — Python dependencies

## Installation

1. Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Quick Start

### Run the web server

```powershell
python app\app.py --serve
```

Open `http://localhost:5000` in your browser to access the frontend.

### Run a single query

```powershell
python app\app.py --query "What is the capital of Australia?"
```

### Start interactive mode

```powershell
python app\app.py
```

## Configuration

Common optional flags:

- `--k` — number of final returned documents (default: 5)
- `--candidate-k` — number of retrieval candidates before reranking (default: 50)
- `--model` — optional model selection for generation
- `--debug` — enable Flask debug mode when serving

## Data and Indexes

The `data/processed/` folder contains prepared dataset files and indexes, including:

- `kb_index.faiss` — FAISS vector index file
- `kb_meta.json` — metadata for knowledge base passages
- `knowledge_base*.json` / `.csv` — question and knowledge data

## Notes

- Flask and Flask-CORS are required for the web API.
- The retrieval pipeline is implemented in `src/retrieval/rag_pipeline.py`.
- The generation and prompt assembly logic lives in `src/generation/prompt.py`.

## License

No license is specified in this repository.
