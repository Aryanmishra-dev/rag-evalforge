"""Reciprocal Rank Fusion: combine multiple ranked result lists into one."""

from collections import defaultdict
from collections.abc import Sequence


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[str]],
    k: int = 60,
) -> list[str]:
    """Fuse ranked doc-id lists via the RRF formula ``1 / (k + rank)``.

    Each result list contributes a decaying score per document based on its
    1-indexed rank; documents are emitted in descending fused-score order.

    Complexity
    ----------
    * Time/space: ``O(sum(len(result_list)))``.

    Parameters
    ----------
    result_lists : sequence of sequences
        Ranked doc-id lists from independent retrieval systems (e.g. dense and
        BM25). Lists are assumed pre-ranked best-first.
    k : int
        RRF smoothing constant (default ``60``); larger ``k`` flattens rank
        disparities between systems.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked_ids in result_lists:
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
