"""Streamlit UI for ingesting PDFs, retrieving chunks, and benchmarking strategies."""

import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.config import EMBED_MODEL, LLM_MODEL
from src.embeddings.chroma_client import (
    add_chunks,
    collection_name,
    get_collection,
    get_collection_for_model,
)
from src.evaluation.run_eval import RESULTS_DIR, TEST_PAIRS_PATH, run_eval
from src.generation.answer import generate_answer
from src.ingestion.chunkers import (
    chunk_fixed,
    chunk_recursive,
    chunk_semantic,
    chunk_sentence,
)
from src.ingestion.ingest import DEFAULT_PDF, generate_doc_id
from src.ingestion.pdf_parser import parse_pdf
from src.retrieval.dense import dense_retrieve
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.rerank import build_reranker

CHUNKERS = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "sentence": chunk_sentence,
    "semantic": chunk_semantic,
}
STRATEGIES = list(CHUNKERS)

STEPS = ["Ingest", "Ask", "Evaluate", "History"]
STEP_TIPS = {
    "Ingest": (
        "**Step 1 of 4 — Build the index.** Upload a PDF below to chunk and "
        "embed it with all four strategies. When ingestion completes, click "
        "**Next → Ask** to query your document."
    ),
    "Ask": (
        "**Step 2 of 4 — Query your document.** Ask a question, inspect the "
        "retrieved chunks, and optionally get a generated answer. When you are "
        "ready, click **Next → Evaluate** to benchmark the strategies."
    ),
    "Evaluate": (
        "**Step 3 of 4 — Benchmark.** Run the retrieval benchmark against the "
        "labeled QA pairs. When it finishes, click **Next → History** to "
        "compare runs."
    ),
    "History": (
        "**Step 4 of 4 — Compare results.** You are done! Review past runs and "
        "download their JSON. Upload another PDF to repeat the cycle."
    ),
}

st.set_page_config(page_title="RAG EvalForge", layout="wide")


def _step_pills(current: str) -> str:
    """HTML stepper showing all four workflow steps with the current one active."""
    pills = []
    for step in STEPS:
        state = (
            "active"
            if step == current
            else "done"
            if STEPS.index(step) < STEPS.index(current)
            else "todo"
        )
        pills.append(f'<span class="steppill {state}">{step}</span>')
    return (
        "<style>"
        ".stepper{display:flex;gap:.5rem;margin:.25rem 0 1rem}"
        ".steppill{flex:1;text-align:center;padding:.4rem 0;border-radius:.4rem;"
        "border:1px solid #555;color:#999;background:#222;font-weight:600}"
        ".steppill.active{background:#4CAF50;border-color:#4CAF50;color:#fff}"
        ".steppill.done{border-color:#4CAF50;color:#4CAF50}"
        "</style>"
        f'<div class="stepper">{"".join(pills)}</div>'
    )


def _apply_nav_request() -> None:
    """Apply a pending "Next →" navigation before the sidebar radio is created.

    Streamlit forbids writing to a widget's key after the widget is
    instantiated, so the workflow buttons stage the target page here and it is
    applied at the top of the next run, before ``render_sidebar`` builds the
    ``nav`` radio.
    """
    request = st.session_state.pop("_nav_request", None)
    if request is not None:
        st.session_state["nav"] = request


def render_workflow(current: str) -> None:
    """Render the 4-step guide for ``current`` plus a link to the next step."""
    st.markdown(_step_pills(current), unsafe_allow_html=True)
    st.info(STEP_TIPS[current], icon=":material/flag:")
    next_step = STEPS[STEPS.index(current) + 1] if current != STEPS[-1] else None
    if next_step is not None and st.button(f"Next → {next_step}", type="primary"):
        st.session_state["_nav_request"] = next_step
        st.rerun()
    st.divider()


def collection_counts() -> dict[str, int]:
    """Return the chunk count for each strategy collection."""
    return {s: get_collection(f"rag_{s}").count() for s in STRATEGIES}


def ingest_pdf(pages: list) -> dict:
    """Chunk and embed `pages` into all four strategy collections."""
    start_time = time.time()
    doc_id = generate_doc_id(pages)
    counts = {}

    status_container = st.empty()
    progress_bar = st.progress(0.0)

    for i, (strategy, chunker) in enumerate(CHUNKERS.items()):
        status = f"Processing...\n\n**{strategy.capitalize()} Chunking**\n\nCreating embeddings..."
        status_container.markdown(status)
        chunks = chunker(pages, doc_id)
        collection = get_collection(f"rag_{strategy}")
        add_chunks(collection, chunks)
        counts[strategy] = len(chunks)
        progress_bar.progress((i + 1) / len(CHUNKERS))
        status = (
            f"Processing...\n\n**{strategy.capitalize()} Chunking**\n\n{len(chunks)} chunks created"
        )
        status_container.markdown(status)

    progress_bar.empty()
    status_container.empty()
    elapsed = time.time() - start_time
    return {
        "doc_id": doc_id,
        "n_pages": len(pages),
        "chunk_counts": counts,
        "elapsed": elapsed,
    }


def retrieve(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    strategy: str,
    question: str,
    k: int,
    hybrid: bool = False,
    rerank: bool = False,
    embed_model: str = EMBED_MODEL,
) -> list | None:
    """Return the top-k retrieved chunks for `question`.

    Uses dense Chroma search by default; with `hybrid` it fuses BM25 + dense
    via RRF, and with `rerank` it additionally re-ranks the candidates. Returns
    a list of RetrievedChunk, or None if the collection is empty.
    """
    collection = get_collection_for_model(strategy, embed_model)
    if collection.count() == 0:
        return None

    if hybrid:
        retriever = HybridRetriever(collection).retrieve
        if rerank:
            return build_reranker().rerank(question, retriever(question, k * 3), k)
        return retriever(question, k)
    return dense_retrieve(collection, question, k)


def render_retrieved(chunks: list) -> None:
    """Display the retrieved chunks as expandable entries."""
    for rank, chunk in enumerate(chunks, start=1):
        label = f"#{rank}  |  page {chunk.page_number}  |  score {chunk.score:.3f}"
        with st.expander(label):
            st.write(chunk.text)


def render_ingest() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Render the Ingest page: parse, chunk, and embed a PDF."""
    render_workflow("Ingest")
    st.header("Ingest")
    st.write(
        "Upload a PDF (up to **200 pages**) to generate embeddings using four "
        "chunking strategies. The document will be processed and stored for "
        "later evaluation."
    )
    st.divider()

    st.markdown(
        """
        <style>
        [data-testid="stFileUploadDropzoneInstructions"] {
            display: none !important;
        }
        [data-testid="stFileUploadDropzone"] > section::after {
            content: "upload your PDF" !important;
            margin-left: 1rem !important;
            align-self: center !important;
            color: rgba(250, 250, 250, 0.6) !important;
            font-size: 0.8rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.4rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload a PDF", type="pdf")

    st.markdown(
        """
        <div style='padding: 1rem; border: 1px solid #333;
                    border-radius: 0.5rem; margin-bottom: 1rem;'>
            <h4 style='margin-top: 0;'>Quick Start</h4>
            <p>Use the bundled sample PDF for testing the application.</p>
        </div>""",
        unsafe_allow_html=True,
    )
    if st.button("Use Sample PDF"):
        st.session_state["use_sample"] = True

    pdf_source = None
    file_name = ""
    file_size_mb = 0.0

    if uploaded is not None:
        st.session_state["use_sample"] = False
        tmp_dir = Path(tempfile.mkdtemp())
        pdf_source = tmp_dir / uploaded.name
        pdf_source.write_bytes(uploaded.getvalue())
        file_name = uploaded.name
        file_size_mb = uploaded.size / (1024 * 1024)
    elif st.session_state.get("use_sample"):
        pdf_source = DEFAULT_PDF
        file_name = DEFAULT_PDF.name
        if pdf_source.exists():
            file_size_mb = pdf_source.stat().st_size / (1024 * 1024)

    if pdf_source is not None and pdf_source.exists():
        st.markdown("### Document Information")
        try:
            pages = parse_pdf(str(pdf_source))
            num_pages = len(pages)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("File Name", file_name)
            col2.metric("Pages", f"{num_pages} / 200")
            col3.metric("Size", f"{file_size_mb:.1f} ")

            if num_pages > 200:
                col4.metric("Status", "Too Large")
                st.error(
                    f"**Maximum page limit exceeded.**\n\nThis document contains "
                    f"{num_pages} pages.\n\nPlease upload a smaller document."
                )
                st.button(
                    "Start Ingestion",
                    type="primary",
                    use_container_width=True,
                    disabled=True,
                )
            else:
                col4.metric("Status", "Ready to ingest")
                st.divider()
                if st.button("Start Ingestion", type="primary", use_container_width=True):
                    try:
                        result = ingest_pdf(pages)
                        st.session_state["last_ingest"] = result
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        st.error(f"**Embedding failed.**\n\nPlease try again. Error: {exc}")

        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.error(f"**Unsupported file.**\n\nOnly PDF documents are allowed. Error: {exc}")

    if "last_ingest" in st.session_state:
        result = st.session_state["last_ingest"]
        st.success("### Ingestion Completed Successfully\n\nDocument processed successfully.")

        counts = result["chunk_counts"]
        st.markdown(
            f"✔ **Fixed:** {counts.get('fixed', 0)} chunks\n\n"
            f"✔ **Recursive:** {counts.get('recursive', 0)} chunks\n\n"
            f"✔ **Sentence:** {counts.get('sentence', 0)} chunks\n\n"
            f"✔ **Semantic:** {counts.get('semantic', 0)} chunks\n\n"
            f"**Total processing time:** {int(result.get('elapsed', 0))} seconds"
        )

    st.divider()
    st.markdown("### Collection Statistics")

    current_counts = collection_counts()
    cols = st.columns(4)
    cols[0].metric("Fixed", f"{current_counts.get('fixed', 0)} Chunks")
    cols[1].metric("Recursive", f"{current_counts.get('recursive', 0)} Chunks")
    cols[2].metric("Sentence", f"{current_counts.get('sentence', 0)} Chunks")
    cols[3].metric("Semantic", f"{current_counts.get('semantic', 0)} Chunks")

    st.divider()
    with st.expander("Advanced Details ▼"):
        st.markdown(
            "**Collections Created**\n\n"
            "• rag_fixed\n"
            "• rag_recursive\n"
            "• rag_sentence\n"
            "• rag_semantic\n\n"
            f"**Embedding Model**\n\n`{LLM_MODEL}`\n\n"
            "**Chunking Strategies**\n\nFixed, Recursive, Sentence, Semantic\n\n"
            "**Storage Location**\n\nLocal ChromaDB"
        )


def render_ask() -> None:
    """Render the Ask page: retrieve top-k chunks and optionally generate an answer."""
    render_workflow("Ask")
    st.header("Ask")
    st.write(
        "Retrieve the most relevant chunks from a strategy's collection and, "
        "optionally, generate an answer grounded in them."
    )

    col_left, col_right = st.columns([1, 3])
    with col_left:
        strategy = st.selectbox("Strategy", STRATEGIES)
        k = st.slider("Top-k chunks", 1, 10, 5)
        hybrid = st.checkbox("Hybrid (BM25 + dense)", value=False)
        rerank = st.checkbox("Re-rank candidates", value=False)
        embed_model = st.text_input("Embedding model", value=EMBED_MODEL)
    with col_right:
        question = st.text_input("Question", placeholder="e.g. What is a phrase?")
        with_answer = st.checkbox("Generate an answer with the LLM", value=True)

    if not question:
        return

    results = retrieve(strategy, question, k, hybrid=hybrid, rerank=rerank, embed_model=embed_model)
    if results is None:
        st.warning(
            f"Collection `{collection_name(strategy, embed_model)}` is empty. Ingest a PDF first."
        )
        return

    with st.spinner("Retrieving..."):
        chunks = results

    st.subheader("Retrieved chunks")
    render_retrieved(chunks)

    if with_answer:
        st.subheader("Answer")
        try:
            with st.spinner(f"Generating with {LLM_MODEL}..."):
                answer = generate_answer(question, chunks)
            st.markdown(answer)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.error(f"Generation failed: {exc}")


def render_evaluate() -> None:
    """Render the Evaluate page: run the benchmark and display results."""
    render_workflow("Evaluate")
    st.header("Evaluate")
    n_pairs = len(json.loads(TEST_PAIRS_PATH.read_text(encoding="utf-8")))
    st.write(
        f"Benchmark retrieval against the {n_pairs} hand-labeled QA pairs "
        "in `src/evaluation/test_qa_pairs.json`. Retrieval is scored on "
        "hit_rate@k and mean reciprocal rank (MRR); the full RAG mode adds "
        "LLM-as-a-judge faithfulness, answer correctness, and answer relevancy."
    )

    k = st.slider("k (retrieved chunks per query)", 1, 10, 5)
    include_hybrid = st.checkbox("Add hybrid (BM25 + dense) and re-ranked retrievers", value=False)
    full_rag = st.checkbox("Full RAG evaluation (LLM-as-a-judge)", value=False)
    if full_rag:
        st.warning(
            f"Full RAG mode makes ~{n_pairs * 8} local LLM calls (generate + 3 judges "
            "per query) and can take **10+ minutes**. Prefer retrieval-only for a "
            "quick check."
        )
    if include_hybrid:
        st.caption(
            "The re-ranked retriever embeds every candidate per query, which "
            "roughly triples the runtime."
        )
    embed_models = st.text_input("Embedding models (comma-separated)", value=EMBED_MODEL)
    models = [m.strip() for m in embed_models.split(",") if m.strip()]

    if st.button("Run evaluation", type="primary"):
        start = time.time()
        progress_bar = st.progress(0.0)
        status = st.empty()
        results = {}

        def _on_progress(key: str, done: int, total: int) -> None:
            progress_bar.progress(done / total)
            status.markdown(f"Evaluating **`{key}`** ... ({done}/{total} retrievers)")

        try:
            with st.spinner("Evaluating..."):
                results = run_eval(
                    k=k,
                    include_hybrid=include_hybrid,
                    full_rag=full_rag,
                    embed_models=models,
                    progress_cb=_on_progress,
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.error(f"Evaluation failed: {exc}")
            return
        finally:
            progress_bar.empty()
            status.empty()

        elapsed = time.time() - start
        out_path = RESULTS_DIR / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        st.session_state["last_eval"] = results
        st.session_state["last_eval_path"] = str(out_path)
        st.session_state["last_eval_elapsed"] = elapsed

    if "last_eval" in st.session_state:
        elapsed = st.session_state.get("last_eval_elapsed", 0)
        st.success(
            f"Done in {int(elapsed // 60)}m {int(elapsed % 60)}s — saved to "
            f"`{st.session_state.get('last_eval_path')}`"
        )
        st.dataframe(eval_frame(st.session_state["last_eval"]), width="stretch")


def eval_frame(results: dict) -> pd.DataFrame:
    """Convert eval results into a rounded DataFrame for display."""
    frame = pd.DataFrame(results).T
    frame = frame.rename(
        columns={
            "avg_hit_rate": "hit_rate@k",
            "avg_mrr": "MRR",
            "avg_faithfulness": "faithfulness",
            "avg_answer_correctness": "answer_correct",
            "avg_answer_relevancy": "answer_relevancy",
        }
    )
    frame.index.name = "strategy"
    return frame.round(3)


def render_history() -> None:
    """Render the History page: compare and download saved evaluation runs."""
    render_workflow("History")
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
            STEPS,
            key="nav",
            help="Workflow: Ingest → Ask → Evaluate → History",
        )
        st.divider()
        st.caption("Collection sizes")
        for strategy, count in collection_counts().items():
            st.caption(f"rag_{strategy}: {count} chunks")
    return page


def main() -> None:
    """Route to the page selected in the sidebar."""
    _apply_nav_request()
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
