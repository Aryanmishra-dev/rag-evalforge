"""A memory-efficient BM25Okapi index backed by numpy inverted postings.

Complexity
----------
* **Build**: ``O(T)`` time and ``O(T)`` space, where ``T`` is the total number
  of tokens in the corpus. Each term maps to a flat ``(doc_index, term_freq)``
  posting array, so no per-document token bags are retained after indexing.
* **Query**: ``O(sum(df(t)) over query terms)`` time via vectorized numpy
  accumulation — sub-linear in the corpus size for typical queries.

The implementation is self-contained (stdlib ``re`` + ``numpy``) and exposes a
drop-in ``fit``/``search``/``top`` API so callers never touch internals.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase and split ``text`` into tokens on non-alphanumeric runs."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """BM25Okapi index over an in-memory corpus.

    Parameters
    ----------
    k1 : float
        Term-frequency saturation parameter (default ``1.5``).
    b : float
        Document-length normalization parameter in ``[0, 1]`` (default ``0.75``).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = []
        self._doc_lengths: np.ndarray = np.empty(0, dtype=np.int64)
        self._avg_doc_length: float = 0.0
        self._postings: dict[str, np.ndarray] = {}
        self._idf: dict[str, float] = {}
        self._n_docs: int = 0

    def fit(
        self,
        documents: Sequence[str],
        doc_ids: Sequence[str] | None = None,
    ) -> BM25Index:
        """Index ``documents``; ``doc_ids`` aligns positionally with ``documents``.

        Idempotent: calling ``fit`` again rebuilds the index from scratch,
        discarding any previously ingested corpus.
        """
        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(documents))]

        self._doc_ids = list(doc_ids)
        self._n_docs = len(self._doc_ids)
        if self._n_docs != len(documents):
            raise ValueError("doc_ids and documents must have equal length")

        self._doc_lengths = np.empty(self._n_docs, dtype=np.int64)
        term_doc_freq: Counter[str] = Counter()
        raw_postings: dict[str, list[tuple[int, int]]] = {}

        for doc_idx, document in enumerate(documents):
            terms = tokenize(document)
            self._doc_lengths[doc_idx] = len(terms)
            for term, freq in Counter(terms).items():
                raw_postings.setdefault(term, []).append((doc_idx, freq))
                term_doc_freq[term] += 1

        self._avg_doc_length = (
            float(self._doc_lengths.sum()) / self._n_docs if self._n_docs else 0.0
        )

        self._postings = {}
        self._idf = {}
        for term, postings in raw_postings.items():
            self._postings[term] = np.asarray(postings, dtype=np.int32)
            self._idf[term] = self._inverse_document_frequency(term_doc_freq[term])

        return self

    @property
    def n_docs(self) -> int:
        """Number of indexed documents."""
        return self._n_docs

    def _inverse_document_frequency(self, df: int) -> float:
        """Standard BM25 IDF with smoothing to keep IDF non-negative."""
        return float(np.log(1.0 + (self._n_docs - df + 0.5) / (df + 0.5)))

    def scores(self, query: str) -> np.ndarray:
        """Return a dense ``(n_docs,)`` score vector for ``query``."""
        if self._n_docs == 0:
            return np.zeros(0, dtype=np.float64)

        scores = np.zeros(self._n_docs, dtype=np.float64)
        for term in tokenize(query):
            postings = self._postings.get(term)
            if postings is None:
                continue
            doc_idxs = postings[:, 0]
            term_freq = postings[:, 1].astype(np.float64)
            doc_lengths = self._doc_lengths[doc_idxs].astype(np.float64)
            denominator = (
                self.k1 * (1.0 - self.b + self.b * doc_lengths / self._avg_doc_length) + term_freq
            )
            scores[doc_idxs] += self._idf[term] * term_freq / denominator
        return scores

    def search(self, query: str) -> list[tuple[str, float]]:
        """Return ``(doc_id, score)`` pairs for every corpus document, ranked."""
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")
        return [(self._doc_ids[idx], float(scores[idx])) for idx in order if scores[idx] > 0.0]

    def top(self, query: str, k: int) -> list[str]:
        """Return the top-``k`` doc ids for ``query`` (empty list if none match)."""
        if k <= 0:
            return []
        return [doc_id for doc_id, _ in self.search(query)[:k]]
