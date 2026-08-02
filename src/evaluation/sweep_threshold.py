import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.embeddings.chroma_client import (
    add_chunks,
    query_collection,
    reset_collection,
)
from src.evaluation.metrics import extract_pages, hit_rate_at_k, reciprocal_rank
from src.ingestion.chunkers import chunk_fixed, chunk_semantic
from src.ingestion.ingest import generate_doc_id
from src.ingestion.pdf_parser import parse_pdf

DEFAULT_PDF = Path("data/raw_pdfs/9788498803488_L33_23.pdf")
TEST_PAIRS_PATH = Path("src/evaluation/test_qa_pairs.json")
THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]
K = 5
TARGET_CHUNKS = 125


def avg_len(chunks: list[dict]) -> float:
    return sum(len(c["text"]) for c in chunks) / len(chunks)


def eval_collection(collection, pairs: list[dict], k: int) -> dict:
    hits = []
    rrs = []
    for pair in pairs:
        result = query_collection(collection, pair["question"], n_results=k)
        pages = extract_pages(result)
        hits.append(hit_rate_at_k(pages, pair["page_number"]))
        rrs.append(reciprocal_rank(pages, pair["page_number"]))
    return {
        "hit_rate": sum(hits) / len(hits),
        "mrr": sum(rrs) / len(rrs),
    }


def main() -> None:
    pages = parse_pdf(str(DEFAULT_PDF))
    doc_id = generate_doc_id(pages)
    pairs = json.loads(TEST_PAIRS_PATH.read_text())

    fixed = chunk_fixed(pages, doc_id)
    print(f"reference fixed: n={len(fixed)} avg_len={avg_len(fixed):.0f}")

    rows = []
    for threshold in THRESHOLDS:
        chunks = chunk_semantic(pages, doc_id, threshold=threshold)
        collection = reset_collection("rag_semantic")
        add_chunks(collection, chunks)
        metrics = eval_collection(collection, pairs, K)
        row = {
            "threshold": threshold,
            "n_chunks": len(chunks),
            "avg_chunk_len": round(avg_len(chunks), 1),
            "hit_rate@5": round(metrics["hit_rate"], 3),
            "MRR": round(metrics["mrr"], 3),
        }
        rows.append(row)
        print(f"semantic t={threshold}: n={len(chunks)} avg_len={avg_len(chunks):.0f} "
              f"hit={metrics['hit_rate']:.3f} mrr={metrics['mrr']:.3f}")

    closest = min(rows, key=lambda r: abs(r["n_chunks"] - TARGET_CHUNKS))
    print(f"\nclosest to {TARGET_CHUNKS} chunks: threshold={closest['threshold']} "
          f"({closest['n_chunks']} chunks, hit={closest['hit_rate@5']}, MRR={closest['MRR']})")

    print("\nrestoring rag_semantic at closest threshold...")
    chunks = chunk_semantic(pages, doc_id, threshold=closest["threshold"])
    add_chunks(reset_collection("rag_semantic"), chunks)
    print("done")


if __name__ == "__main__":
    main()
