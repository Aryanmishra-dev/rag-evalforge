"""Streamlit UI for ingesting PDFs, retrieving chunks, and benchmarking strategies."""
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ollama
import pandas as pd
import streamlit as st

from src.config import LLM_MODEL
from src.embeddings.chroma_client import add_chunks, get_collection
from src.evaluation.run_eval import RESULTS_DIR, TEST_PAIRS_PATH, run_eval
from src.ingestion.chunkers import (
    chunk_fixed,
    chunk_recursive,
    chunk_semantic,
    chunk_sentence,
)
from src.ingestion.ingest import DEFAULT_PDF, generate_doc_id
from src.ingestion.pdf_parser import parse_pdf

CHUNKERS = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "sentence": chunk_sentence,
    "semantic": chunk_semantic,
}
STRATEGIES = list(CHUNKERS)

st.set_page_config(page_title="RAG EvalForge", layout="wide")


def collection_counts() -> dict[str, int]:
    """Return the chunk count for each strategy collection."""
    return {s: get_collection(f"rag_{s}").count() for s in STRATEGIES}


def ingest_pdf(pdf_path: Path) -> dict:
    """Chunk and embed `pdf_path` into all four strategy collections."""
    pages = parse_pdf(str(pdf_path))
    doc_id = generate_doc_id(pages)
    counts = {}
    progress = st.progress(0.0, text="Parsed pages, embedding chunks...")
    for i, (strategy, chunker) in enumerate(CHUNKERS.items()):
        chunks = chunker(pages, doc_id)
        collection = get_collection(f"rag_{strategy}")
        add_chunks(collection, chunks)
        counts[strategy] = len(chunks)
        progress.progress((i + 1) / len(CHUNKERS), text=f"{strategy}: {len(chunks)} chunks")
    progress.empty()
    return {"doc_id": doc_id, "n_pages": len(pages), "chunk_counts": counts}


def retrieve(strategy: str, question: str, k: int) -> dict | None:
    """Return the top-k Chroma query result for `question`, or None if empty."""
    collection = get_collection(f"rag_{strategy}")
    if collection.count() == 0:
        return None
    return collection.query(query_texts=[question], n_results=k)


def generate_answer(question: str, results: dict) -> str:
    """Generate an LLM answer grounded in the retrieved chunks."""
    context = "\n\n".join(
        f"[source: page {meta['page_number']}] {doc}"
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    )
    prompt = (
        "Answer the question using only the provided context. "
        "If the context does not contain the answer, say you could not find it.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def render_retrieved(docs: list, metas: list, dists: list) -> None:
    """Display the retrieved chunks as expandable entries."""
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        label = f"#{rank}  |  page {meta['page_number']}  |  distance {dist:.3f}"
        with st.expander(label):
            st.write(doc)


def render_ingest() -> None:
    """Render the Ingest page: parse, chunk, and embed a PDF."""
    st.header("Ingest")
    st.write(
        "Parse a PDF, chunk it with all four strategies, and embed the chunks "
        "into one Chroma collection per strategy (`rag_fixed`, `rag_recursive`, "
        "`rag_sentence`, `rag_semantic`)."
    )

    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    use_default = st.checkbox("Use the bundled sample PDF", value=uploaded is None)

    pdf_source = None
    if uploaded is not None:
        tmp_dir = Path(tempfile.mkdtemp())
        pdf_source = tmp_dir / uploaded.name
        pdf_source.write_bytes(uploaded.getvalue())
        st.caption(f"Uploaded: {uploaded.name}")
    elif use_default:
        pdf_source = DEFAULT_PDF
        st.caption(f"Sample: {DEFAULT_PDF}")

    if pdf_source is None or not pdf_source.exists():
        st.warning("Choose a PDF to ingest.")
        return

    if st.button("Ingest", type="primary"):
        try:
            with st.spinner("Ingesting (semantic strategy embeds each sentence)..."):
                result = ingest_pdf(pdf_source)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.error(f"Ingestion failed: {exc}")
            return
        st.session_state["last_ingest"] = result
        st.session_state["last_ingest_time"] = datetime.now().strftime("%H:%M:%S")

    if "last_ingest" in st.session_state:
        result = st.session_state["last_ingest"]
        st.success(
            f"Ingested {result['n_pages']} pages (doc_id={result['doc_id']}) "
            f"at {st.session_state.get('last_ingest_time')}."
        )
        counts = pd.DataFrame(
            [{"strategy": s, "chunks": c} for s, c in result["chunk_counts"].items()]
        )
        st.dataframe(counts, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Current collection sizes")
    sizes = pd.DataFrame(
        [{"strategy": s, "chunks": c} for s, c in collection_counts().items()]
    )
    st.dataframe(sizes, width="stretch", hide_index=True)


def render_ask() -> None:
    """Render the Ask page: retrieve top-k chunks and optionally generate an answer."""
    st.header("Ask")
    st.write(
        "Retrieve the most relevant chunks from a strategy's collection and, "
        "optionally, generate an answer grounded in them."
    )

    col_left, col_right = st.columns([1, 3])
    with col_left:
        strategy = st.selectbox("Strategy", STRATEGIES)
        k = st.slider("Top-k chunks", 1, 10, 5)
    with col_right:
        question = st.text_input("Question", placeholder="e.g. What is a phrase?")
        with_answer = st.checkbox("Generate an answer with the LLM", value=True)

    if not question:
        return

    results = retrieve(strategy, question, k)
    if results is None:
        st.warning(f"Collection `rag_{strategy}` is empty. Ingest a PDF first.")
        return

    with st.spinner("Retrieving..."):
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

    st.subheader("Retrieved chunks")
    render_retrieved(docs, metas, dists)

    if with_answer:
        st.subheader("Answer")
        try:
            with st.spinner(f"Generating with {LLM_MODEL}..."):
                answer = generate_answer(question, results)
            st.markdown(answer)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.error(f"Generation failed: {exc}")


def render_evaluate() -> None:
    """Render the Evaluate page: run the benchmark and display results."""
    st.header("Evaluate")
    n_pairs = len(json.loads(TEST_PAIRS_PATH.read_text(encoding="utf-8")))
    st.write(
        f"Run the retrieval benchmark against the {n_pairs} hand-labeled QA pairs "
        "in `src/evaluation/test_qa_pairs.json` and compare strategies on "
        "hit_rate@k and mean reciprocal rank (MRR)."
    )

    k = st.slider("k (retrieved chunks per query)", 1, 10, 5)
    if st.button("Run evaluation", type="primary"):
        try:
            with st.spinner(f"Evaluating {len(STRATEGIES)} strategies x {n_pairs} queries..."):
                results = run_eval(k=k)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.error(f"Evaluation failed: {exc}")
            return

        out_path = RESULTS_DIR / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        st.session_state["last_eval"] = results
        st.session_state["last_eval_path"] = str(out_path)

    if "last_eval" in st.session_state:
        st.success(f"Saved to `{st.session_state.get('last_eval_path')}`")
        st.dataframe(eval_frame(st.session_state["last_eval"]), width="stretch")


def eval_frame(results: dict) -> pd.DataFrame:
    """Convert eval results into a rounded DataFrame for display."""
    frame = pd.DataFrame(results).T
    frame = frame.rename(columns={"avg_hit_rate": "hit_rate@k", "avg_mrr": "MRR"})
    frame.index.name = "strategy"
    return frame.round(3)


def render_history() -> None:
    """Render the History page: compare and download saved evaluation runs."""
    st.header("History")
    files = sorted(RESULTS_DIR.glob("eval_results_*.json"))
    if not files:
        st.info("No evaluation runs saved yet.")
        return

    rows = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {"run": path.stem.removeprefix("eval_results_")}
        for strategy, metrics in data.items():
            row[f"{strategy} hit@k"] = metrics["avg_hit_rate"]
            row[f"{strategy} MRR"] = metrics["avg_mrr"]
        rows.append(row)

    st.subheader("All runs")
    runs = pd.DataFrame(rows).sort_values("run", ascending=False).round(3)
    st.dataframe(runs, width="stretch")

    st.subheader("Inspect a run")
    selected = st.selectbox("Run", [r["run"] for r in rows])
    path = next(p for p in files if p.stem == f"eval_results_{selected}")
    st.dataframe(eval_frame(json.loads(path.read_text(encoding="utf-8"))), width="stretch")
    st.download_button(
        "Download JSON",
        data=path.read_text(encoding="utf-8"),
        file_name=path.name,
        mime="application/json",
    )


def render_sidebar() -> str:
    """Render the sidebar navigation and return the selected page name."""
    with st.sidebar:
        st.title("RAG EvalForge")
        page = st.radio(
            "Navigation",
            ["Ingest", "Ask", "Evaluate", "History"],
            index=0,
        )
        st.divider()
        st.caption("Collection sizes")
        for strategy, count in collection_counts().items():
            st.caption(f"rag_{strategy}: {count} chunks")
    return page


def main() -> None:
    """Route to the page selected in the sidebar."""
    page = render_sidebar()
    if page == "Ingest":
        render_ingest()
    elif page == "Ask":
        render_ask()
    elif page == "Evaluate":
        render_evaluate()
    else:
        render_history()


if __name__ == "__main__":
    main()
