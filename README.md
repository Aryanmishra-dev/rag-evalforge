<div align="center">

  <h1>🚀 RAG EvalForge</h1>
  <p><i>Benchmark chunking strategies and retrieval pipelines for RAG — fully local with Ollama + ChromaDB.</i></p>

  <p>
    <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+"></a>
    <img src="https://img.shields.io/badge/tests-116%20passed-brightgreen.svg?style=for-the-badge&logo=pytest&logoColor=white" alt="116 tests passing">
    <img src="https://img.shields.io/badge/coverage-89%25-brightgreen.svg?style=for-the-badge" alt="89% coverage">
    <img src="https://img.shields.io/badge/pylint-10.00%2F10-brightgreen.svg?style=for-the-badge" alt="Pylint 10.00/10">
    <img src="https://img.shields.io/badge/pyright-0%20errors-brightgreen.svg?style=for-the-badge" alt="Pyright clean">
    <img src="https://img.shields.io/badge/CI-GitHub%20Actions-blue.svg?style=for-the-badge&logo=githubactions&logoColor=white" alt="GitHub Actions CI">
  </p>
</div>

---

## What It Does

RAG EvalForge indexes documents under **four chunking strategies** (Fixed, Recursive, Sentence, Semantic) into separate ChromaDB collections and evaluates retrieval quality using `hit_rate@k` and MRR against a hand-labeled QA dataset. It optionally scores end-to-end RAG answer quality via an LLM-as-a-judge.

Everything runs locally — no cloud APIs required.

---

## Key Features

| Feature | Description |
|:---|:---|
| **Chunking Benchmark** | Fixed, Recursive, Sentence, and Semantic strategies compared side-by-side |
| **Multi-Embedding Evaluation** | Same strategies under multiple embedding models with per-model collection namespacing |
| **Advanced Retrieval** | Dense, BM25, hybrid fusion (RRF), and cross-encoder re-ranking |
| **Generation Quality** | LLM-as-a-judge scoring of faithfulness, correctness, and relevancy |
| **Experiment Tracking** | SQLite registry with git commit, config, dataset hash, and per-query metrics |
| **Streamlit UI** | Interactive dashboard for ingest, query, evaluate, and history |
| **Docker Support** | One-command deployment via `docker compose` |
| **MCP Server** | Expose pipeline tools to LLM clients (Claude Desktop, etc.) via Model Context Protocol |

---

## Tech Stack

**Python 3.11+** · **ChromaDB** · **Ollama** (`nomic-embed-text`, `qwen2.5:7b`) · **BM25 + RRF** · **PyMuPDF** · **Streamlit** · **SQLite** · **Docker**

---

## Quick Start

### Prerequisites

- Python 3.11+ and [Ollama](https://ollama.com/) running locally

```sh
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

### Install & Run

```sh
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python smoke_test.py
```

### Docker (alternative)

```sh
docker compose up -d --build    # App at http://localhost:8501
```

---

## Usage

```sh
# Ingest
python src/ingestion/ingest.py                          # bundled sample PDF
python src/ingestion/ingest.py path/to/other.pdf        # custom PDF

# Benchmark
python src/evaluation/run_eval.py --k 5                 # retrieval-only
python src/evaluation/run_eval.py --k 5 --hybrid        # + hybrid fusion & re-ranking
python src/evaluation/run_eval.py --k 5 --hybrid --rag  # + LLM-as-a-judge

# UI
streamlit run app/streamlit_app.py

# MCP Server
make mcp-server
```

---

## Benchmark Results (k = 5)

| Strategy | Chunks | Hit Rate @ 5 | MRR |
|:---|---:|---:|---:|
| `fixed` | 125 | 0.950 | 0.778 |
| **`recursive`** | 126 | **1.000** | **0.858** |
| `sentence` | 167 | 0.950 | 0.850 |
| `semantic` (t=0.3) | 168 | 0.950 | 0.850 |

---

## Project Structure

```
rag-evalforge/
├── app/                   # Streamlit UI
├── src/
│   ├── config.py          # Configuration & env vars
│   ├── mcp_server.py      # MCP server for LLM clients
│   ├── embeddings/        # Ollama embeddings & ChromaDB layer
│   ├── ingestion/         # PDF parsing & chunkers
│   ├── retrieval/         # Dense, BM25, hybrid, re-ranking
│   ├── generation/        # Grounded answers & LLM-as-a-judge
│   ├── evaluation/        # Benchmarking & metrics
│   └── experiment/        # SQLite experiment registry
├── tests/                 # 108 offline unit tests
├── data/                  # PDFs, ChromaDB storage, eval results
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

---

## QA & Code Quality

All checks pass on every push via GitHub Actions (Python 3.11 / 3.12 / 3.13):

**pytest** 116/116 · **coverage** 89% · **pylint** 10.00/10 · **ruff** 0 errors · **pyright** 0 errors · **bandit** 0 issues · **radon** complexity A (2.83)

```sh
make lint      # pylint + ruff
make qa        # full gate: lint + pyright + bandit + tests + coverage
make profile   # scalene CPU profile
```
