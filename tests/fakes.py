"""In-memory stand-ins for Chroma collections so retrieval code is testable offline."""


class FakeCollection:
    """Minimal Chroma collection contract: name, count, get, query.

    ``query`` returns a Chroma-shaped result dict: each top-level key wraps a
    list of per-query lists (one query here). ``get`` returns flat lists.
    """

    def __init__(self, name, documents, metadatas=None, ids=None):
        self.name = name
        self._ids = list(ids or [f"{name}:{i}" for i in range(len(documents))])
        self._documents = list(documents)
        self._metadatas = list(
            metadatas or [{"page_number": i % 20 + 1} for i in range(len(documents))]
        )
        # Dense ordering: documents whose text overlaps the query come first.
        self._query_rank = self._ids

    def count(self):
        return len(self._ids)

    def get(self, include=None):
        return {
            "ids": list(self._ids),
            "documents": list(self._documents),
            "metadatas": [dict(m) for m in self._metadatas],
        }

    def query(self, query_texts, n_results):
        query = query_texts[0]
        terms = query.lower().split()
        ranked = sorted(
            zip(self._ids, self._documents, self._metadatas, strict=True),
            key=lambda t: -sum(w in t[1].lower() for w in terms),
        )
        ranked = ranked[:n_results]
        return {
            "ids": [[t[0] for t in ranked]],
            "documents": [[t[1] for t in ranked]],
            "metadatas": [[t[2] for t in ranked]],
            "distances": [[float(i + 1) for i in range(len(ranked))]],
        }
