# RAG EvalForge

Build a Retrieval-Augmented Generation (RAG) pipeline over a technical-writing
textbook and **benchmark chunking strategies** on retrieval quality — all running
locally with Ollama embeddings and ChromaDB.

The core question this project answers: *how you chunk a document changes how
well retrieval works.* Four chunking strategies are indexed into separate Chroma
collections, then scored against a hand-labeled QA dataset using `hit_rate@k`
and mean reciprocal rank (MRR).

## Pipeline

```
PDF ──parse──▶ pages ──chunk──▶ chunks ──embed (Ollama)──▶ ChromaDB
                                                              │
                                                              ▼
                          eval QA pairs ◀──retrieve──  query ──┘
                                                              │
                                                              ▼
                                              hit_rate@k, MRR report
```

- **Ingest**: `parse_pdf` extracts page text with PyMuPDF; each strategy chunks
  the pages into `{chunk_id, text, page_number, strategy}` records.
- **Embed**: `nomic-embed-text` via Ollama (768-dim) produces the vectors.
- **Store**: one Chroma collection per strategy (`rag_fixed`, `rag_recursive`,
  `rag_sentence`, `rag_semantic`), keyed by a content-hash `doc_id` so
  re-ingesting the same file is idempotent.
- **Evaluate**: 20 hand-labeled question/page pairs
  (`src/evaluation/test_qa_pairs.json`) are queried against each collection and
  scored on page-level `hit_rate@k` and `MRR`.

## Chunking strategies

| Strategy | How it splits text | Chunk keying |
|---|---|---|
| `fixed` | Hard character window (500 chars, 50 overlap) | `docid_fixed_page_start_end` |
| `recursive` | Recursive split on `\n\n` → `\n` → `. ` → ` ` | `docid_recursive_page_i` |
| `sentence` | Sentence-boundary split, overlap carries sentence tail | `docid_sentence_page_i` |
| `semantic` | Sentence embeddings; new chunk when consecutive-sentence cosine similarity drops below a threshold (0.7), or the size cap is hit | `docid_semantic_page_i` |

## Project structure

```
app/streamlit_app.py          Streamlit UI: ingest, ask, evaluate, history
src/
  config.py                   Loads LLM_MODEL, EMBED_MODEL, CHROMA_DB_PATH, OLLAMA_HOST from .env
  embeddings/
    embedder.py               OllamaEmbeddingFunction + embed() helper
    chroma_client.py          Chroma get_collection / reset_collection / add_chunks / query
  ingestion/
    pdf_parser.py             PyMuPDF page extraction -> [{page_number, text}]
    chunkers.py               The four chunking strategies
    ingest.py                 CLI: parse + chunk + embed + store a PDF
  evaluation/
    metrics.py                hit_rate_at_k, reciprocal_rank, extract_pages
    run_eval.py               CLI/API: benchmark all strategies against the QA pairs
    sweep_threshold.py        CLI: sweep semantic threshold to match a target chunk count
    test_qa_pairs.json        20 QA pairs with expected page numbers
data/
  raw_pdfs/                   Input PDFs
  chroma_db/                  Persistent Chroma store (gitignored)
  eval_results/               Saved benchmark runs as JSON (gitignored)
tests/test_chunkers.py        Chunker tests
```

## Prerequisites

- Python 3.11
- [Ollama](https://ollama.com) running locally with the embed + LLM models pulled:

  ```sh
  ollama pull nomic-embed-text
  ollama pull qwen2.5:7b
  ```

## Setup

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env` requires these variables:

```ini
LLM_MODEL=qwen2.5:7b
EMBED_MODEL=nomic-embed-text
CHROMA_DB_PATH=./data/chroma_db
OLLAMA_HOST=http://localhost:11434
```

Verify the stack is reachable:

```sh
python smoke_test.py
```

## Usage

### 1. Ingest a PDF

```sh
python src/ingestion/ingest.py                      # uses the bundled sample PDF
python src/ingestion/ingest.py path/to/other.pdf    # or a specific file
```

This chunks the document with all four strategies and embeds them into the four
collections.

### 2. Run the retrieval benchmark

```sh
python src/evaluation/run_eval.py --k 5
```

Prints a strategy-vs-metrics table and saves
`data/eval_results/eval_results_<timestamp>.json`.

### 3. Tune the semantic threshold

```sh
python src/evaluation/sweep_threshold.py
```

Sweeps the semantic chunker's similarity threshold, reports chunk count / chunk
length / hit_rate / MRR per threshold, and restores the collection at the
threshold closest to the target chunk count.

### 4. Streamlit UI

```sh
streamlit run app/streamlit_app.py
```

| Page | What it does |
|---|---|
| **Ingest** | Upload a PDF (or use the sample), chunk + embed it with all four strategies, watch progress |
| **Ask** | Retrieve top-k chunks from any strategy; optionally generate an LLM answer grounded in the context |
| **Evaluate** | Run the benchmark from the UI and save the result |
| **History** | Compare all saved eval runs side by side; inspect/download any run |

## Evaluation metrics

Scoring is page-level against the ground-truth `page_number` in each QA pair:

- **hit_rate@k** — 1 if any retrieved chunk is from the expected page, else 0.
  Averaged across the dataset.
- **MRR (mean reciprocal rank)** — `1/rank` of the first chunk from the expected
  page, or 0 if it is not in the top-k. Penalizes correct-but-low-ranked
  retrievals.

## Latest benchmark results

Latest run: `data/eval_results/eval_results_20260802_204421.json` (k=5).

| strategy | chunks | hit_rate@5 | MRR |
|---|---|---|---|
| fixed | 125 | 0.950 | 0.778 |
| recursive | 126 | **1.000** | **0.858** |
| sentence | 167 | 0.950 | 0.850 |
| semantic (t=0.3) | 168 | 0.950 | 0.850 |

**Granularity confound (fixed):** the default semantic threshold (0.7)
over-segments this document into 317 chunks vs ~125–167 for the other
strategies, which biases raw comparisons. The threshold was swept on this exact
document and set to 0.3 (168 chunks) to match sentence/recursive granularity
before scoring — under that matched comparison, semantic is on par with
sentence instead of appearing worse.

> Caveat: the test set is small (20 QA pairs), so a single miss moves hit_rate
> by 0.05 and differences under ~0.05 MRR are within noise. MRR is also capped
> at 1.0 with a single page-level ground truth per question, so near-ceiling
> strategies are hard to separate — treat the top of this table as a tie.

> Note: retrieval quality is measured here — generation quality is not yet
> scored. `src/retrieval/`, `src/generation/`, and `src/vectorstore/` are
> reserved for the next stages of the pipeline.
