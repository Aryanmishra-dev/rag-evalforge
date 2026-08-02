import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.embeddings.chroma_client import get_collection, query_collection
from src.evaluation.metrics import extract_pages, hit_rate_at_k, reciprocal_rank

STRATEGIES = ["fixed", "recursive", "sentence", "semantic"]
TEST_PAIRS_PATH = Path("src/evaluation/test_qa_pairs.json")
RESULTS_DIR = Path("data/eval_results")


def run_eval(k: int = 5) -> dict:
    pairs = json.loads(TEST_PAIRS_PATH.read_text())
    results = {}
    for strategy in STRATEGIES:
        collection = get_collection(f"rag_{strategy}")
        count = collection.count()
        if count == 0:
            print(f"WARNING: collection 'rag_{strategy}' is empty ({count} chunks). "
                  f"Run ingestion first.")
        hits = []
        rrs = []
        for pair in pairs:
            result = query_collection(collection, pair["question"], n_results=k)
            pages = extract_pages(result)
            hits.append(hit_rate_at_k(pages, pair["page_number"]))
            rrs.append(reciprocal_rank(pages, pair["page_number"]))
        results[strategy] = {
            "avg_hit_rate": sum(hits) / len(hits),
            "avg_mrr": sum(rrs) / len(rrs),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG chunking-strategy benchmark.")
    parser.add_argument("--k", type=int, default=5, help="Retrieved chunks per query.")
    args = parser.parse_args()

    results = run_eval(k=args.k)

    print(f"\n{'strategy':<12}{'hit_rate@k':>12}{'MRR':>10}")
    print("-" * 34)
    for strategy, metrics in results.items():
        print(f"{strategy:<12}{metrics['avg_hit_rate']:>12.3f}{metrics['avg_mrr']:>10.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_results_{timestamp}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
