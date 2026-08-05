"""Four chunking strategies: fixed-size, recursive, sentence, and semantic."""

import re

from src.embeddings.embedder import embed


def chunk_fixed(
    pages: list[dict],
    doc_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    Input: [{"page_number": int, "text": str}, ...]
    Output: [{"chunk_id": str, "text": str, "page_number": int, "strategy": str}, ...]
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    for page in pages:
        text = page["text"]
        page_number = page["page_number"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            if len(chunk_text.strip()) < 20:
                start += chunk_size - overlap
                continue
            chunk_id = f"{doc_id}_fixed_{page_number}_{start}_{end}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "page_number": page_number,
                    "strategy": "fixed",
                }
            )
            start += chunk_size - overlap
    return chunks


def _split_into_pieces(text: str, separators: list[str], chunk_size: int) -> list[str]:
    pieces = []

    def recurse(s: str, seps: list[str]) -> None:
        if len(s) <= chunk_size or not seps:
            if s.strip():
                pieces.append(s)
            return
        for part in s.split(seps[0]):
            recurse(part, seps[1:])

    recurse(text, separators)
    return pieces


def _merge_pieces(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    buffer = ""
    for piece in pieces:
        if len(piece) > chunk_size:
            if buffer.strip():
                chunks.append(buffer)
            for start in range(0, len(piece), chunk_size):
                chunk = piece[start : start + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            buffer = ""
            continue
        if buffer and len(buffer) + len(piece) + 1 > chunk_size:
            chunks.append(buffer)
            tail = buffer[-overlap:] if overlap else ""
            buffer = tail + piece
        else:
            buffer = buffer + " " + piece if buffer else piece
    if buffer.strip():
        chunks.append(buffer)
    return chunks


def chunk_recursive(
    pages: list[dict],
    doc_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " "),
) -> list[dict]:
    """
    Input: [{"page_number": int, "text": str}, ...]
    Output: [{"chunk_id": str, "text": str, "page_number": int, "strategy": str}, ...]
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    for page in pages:
        text = page["text"]
        page_number = page["page_number"]
        pieces = _split_into_pieces(text, list(separators), chunk_size)
        merged = _merge_pieces(pieces, chunk_size, overlap)
        for i, chunk_text in enumerate(merged):
            if len(chunk_text.strip()) < 20:
                continue
            chunk_id = f"{doc_id}_recursive_{page_number}_{i}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "page_number": page_number,
                    "strategy": "recursive",
                }
            )
    return chunks


_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "mt",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "cf",
    "al",
    "no",
    "dept",
    "univ",
    "fig",
    "figs",
    "vol",
    "pp",
    "sec",
    "ch",
    "ltd",
    "inc",
}
_ABBREV_RE = re.compile(
    r"\b(" + "|".join(sorted(_ABBREVIATIONS, key=len, reverse=True)) + r")\.|"
    r"(?<=\b[A-Z])\.(?=\s+[A-Z])",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_ABBREV_PLACEHOLDER = "\x00{0}\x00"


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, keeping abbreviations intact.

    Periods inside known abbreviations (``Dr.``, ``e.g.``) and single-letter
    initials are temporarily masked before splitting, so they are never treated
    as sentence boundaries.
    """
    protected: list[str] = []

    def _protect(match: re.Match) -> str:
        protected.append(match.group(0))
        return _ABBREV_PLACEHOLDER.format(len(protected) - 1)

    masked = _ABBREV_RE.sub(_protect, text)
    raw_parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(masked)]

    parts: list[str] = []
    for part in raw_parts:
        for i, token in enumerate(protected):
            part = part.replace(_ABBREV_PLACEHOLDER.format(i), token)
        if part.strip():
            parts.append(part)
    return parts


def _overlap_tail(sentences: list[str], overlap: int) -> list[str]:
    """Return trailing sentences to carry as overlap.

    Carries up to `overlap` characters, but capped at 2 sentences — so this
    is NOT char-exact like chunk_fixed/chunk_recursive; it only guarantees
    at least 1 sentence when overlap > 0.
    """
    if overlap <= 0:
        return []
    tail = []
    total = 0
    for sent in reversed(sentences):
        if len(tail) >= 2:
            break
        cost = len(sent) + (1 if tail else 0)
        if tail and total + cost > overlap:
            break
        tail.insert(0, sent)
        total += cost
    return tail


def _sentence_chunk_page(text: str, chunk_size: int, overlap: int) -> list[str]:
    sentences = _split_sentences(text)
    page_chunks: list[str] = []
    buffer: list[str] = []
    for sent in sentences:
        if len(sent) > chunk_size:
            if buffer:
                page_chunks.append(" ".join(buffer))
                buffer = []
            for start in range(0, len(sent), chunk_size):
                piece = sent[start : start + chunk_size]
                if piece.strip():
                    page_chunks.append(piece)
            continue
        if buffer and len(" ".join(buffer)) + len(sent) + 1 > chunk_size:
            tail = _overlap_tail(buffer, overlap)
            page_chunks.append(" ".join(buffer))
            buffer = list(tail)
        buffer.append(sent)
    if buffer:
        page_chunks.append(" ".join(buffer))
    return page_chunks


def chunk_sentence(
    pages: list[dict],
    doc_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    Input: [{"page_number": int, "text": str}, ...]
    Output: [{"chunk_id": str, "text": str, "page_number": int, "strategy": str}, ...]
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    for page in pages:
        page_number = page["page_number"]
        for i, chunk_text in enumerate(_sentence_chunk_page(page["text"], chunk_size, overlap)):
            if len(chunk_text.strip()) < 20:
                continue
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_sentence_{page_number}_{i}",
                    "text": chunk_text,
                    "page_number": page_number,
                    "strategy": "sentence",
                }
            )
    return chunks


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _semantic_chunk_page(
    text: str,
    chunk_size: int,
    overlap: int,
    threshold: float,
) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    vectors = [embed(s) for s in sentences]
    similarities = [_cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    page_chunks: list[str] = []
    buffer = [sentences[0]]
    for i in range(1, len(sentences)):
        boundary = similarities[i - 1] < threshold
        over_size = len(" ".join(buffer)) + len(sentences[i]) + 1 > chunk_size
        if boundary or over_size:
            page_chunks.append(" ".join(buffer))
            buffer = [buffer[-1]] if overlap > 0 else []
        buffer.append(sentences[i])
    if buffer:
        page_chunks.append(" ".join(buffer))
    return page_chunks


def chunk_semantic(
    pages: list[dict],
    doc_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
    threshold: float = 0.7,
) -> list[dict]:
    """
    Input: [{"page_number": int, "text": str}, ...]
    Output: [{"chunk_id": str, "text": str, "page_number": int, "strategy": str}, ...]

    Splits at sentence boundaries where consecutive-sentence cosine similarity
    drops below `threshold` (topic shift). `chunk_size` acts only as a safety
    cap; overlap carries the boundary sentence into the next chunk.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    for page in pages:
        page_number = page["page_number"]
        for i, chunk_text in enumerate(
            _semantic_chunk_page(page["text"], chunk_size, overlap, threshold)
        ):
            if len(chunk_text.strip()) < 20:
                continue
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_semantic_{page_number}_{i}",
                    "text": chunk_text,
                    "page_number": page_number,
                    "strategy": "semantic",
                }
            )
    return chunks
