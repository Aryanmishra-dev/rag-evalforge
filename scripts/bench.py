"""Representative offline RAG workload, used as a scalene profiling target.

Exercises chunking (all four strategies), dense retrieval, BM25 hybrid
retrieval, reciprocal rank fusion, and the retrieval metrics against in-memory
fakes so it runs with no Ollama/network. Run with::

    scalene run --cpu-only scripts/bench.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.ingestion.chunkers as chunkers
from src.evaluation.metrics import hit_rate_at_k, reciprocal_rank
from src.ingestion.chunkers import (
    chunk_fixed,
    chunk_recursive,
    chunk_semantic,
    chunk_sentence,
)
from src.retrieval.dense import dense_retrieve
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.rrf import reciprocal_rank_fusion
from tests.fakes import FakeCollection

DOC = (
    "Machine learning retrieval systems combine dense and sparse signals. "
    "Embeddings capture semantic similarity while BM25 matches lexical terms. "
    "Hybrid retrieval fuses both rankings with reciprocal rank fusion. "
    "Chunking splits documents into fixed, recursive, sentence, and semantic units. "
    "Evaluation measures hit rate at k and mean reciprocal rank against labeled pages. "
    "Rerankers like cross encoders improve precision on the fused shortlist. "
    "These ingredients form a modern RAG pipeline used in production systems. "
) * 20

QUERIES = [
    "how do embeddings capture semantics",
    "what is BM25 lexical matching",
    "how does reciprocal rank fusion work",
    "what chunking strategies exist",
    "how is hit rate measured",
    "cross encoder reranker precision",
    "modern rag pipeline production",
    "dense sparse signal fusion",
    "labeled pages mean reciprocal rank",
    "semantic similarity shortlist",
]


def _fake_embed(text: str) -> list[float]:
    v = [0.0] * 64
    for i, ch in enumerate(text):
        v[hash((ch, i)) % 64] += 1.0
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v


chunkers.embed = _fake_embed


def run() -> None:
    docs = [{"page_number": i + 1, "text": DOC} for i in range(30)]
    doc_id = "bench"

    all_chunks = []
    all_chunks += chunk_fixed(docs, doc_id=doc_id, chunk_size=200, overlap=20)
    all_chunks += chunk_recursive(docs, doc_id=doc_id, chunk_size=200)
    all_chunks += chunk_sentence(docs, doc_id=doc_id, chunk_size=200, overlap=0)
    all_chunks += chunk_semantic(docs, doc_id=doc_id, chunk_size=200, threshold=0.8)

    collection = FakeCollection(
        name="bench",
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"page_number": c["page_number"]} for c in all_chunks],
        ids=[c["chunk_id"] for c in all_chunks],
    )

    hybrid = HybridRetriever(collection)
    page_of = {c["chunk_id"]: c["page_number"] for c in all_chunks}

    total_hits = 0.0
    total_mrr = 0.0
    n = 0
    for i, query in enumerate(QUERIES):
        dense = dense_retrieve(collection, query, k=10)
        fused = reciprocal_rank_fusion(
            [
                [c.chunk_id for c in dense],
                [c.chunk_id for c in hybrid.retrieve(query, k=10)],
            ],
            k=10,
        )
        pages = [page_of[cid] for cid in fused]
        expected = i % 20 + 1
        total_hits += hit_rate_at_k(pages, expected)
        total_mrr += reciprocal_rank(pages, expected)
        n += 1
    print(f"chunks={len(all_chunks)} avg_hit={total_hits / n:.3f} avg_mrr={total_mrr / n:.3f}")


if __name__ == "__main__":
    run()
