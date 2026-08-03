"""Four chunking strategies: fixed-size, recursive, sentence, and semantic."""
from typing import List, Tuple

import re

from src.embeddings.embedder import embed


def chunk_fixed(
    pages: List[dict],
    doc_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[dict]:
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
                start += (chunk_size - overlap)
                continue
            chunk_id = f"{doc_id}_fixed_{page_number}_{start}_{end}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "page_number": page_number,
                "strategy": "fixed"
            })
            start += (chunk_size - overlap)
    return chunks


def _split_into_pieces(text: str, separators: List[str], chunk_size: int) -> List[str]:
    pieces = []

    def recurse(s: str, seps: List[str]) -> None:
        if len(s) <= chunk_size or not seps:
            if s.strip():
                pieces.append(s)
            return
        for part in s.split(seps[0]):
            recurse(part, seps[1:])

    recurse(text, separators)
    return pieces


def _merge_pieces(pieces: List[str], chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    buffer = ""
    for piece in pieces:
        if len(piece) > chunk_size:
            if buffer.strip():
                chunks.append(buffer)
            for start in range(0, len(piece), chunk_size):
                chunk = piece[start:start + chunk_size]
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
    pages: List[dict],
    doc_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separators: Tuple[str, ...] = ("\n\n", "\n", ". ", " "),
) -> List[dict]:
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
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "page_number": page_number,
                "strategy": "recursive"
            })
    return chunks


def _split_sentences(text: str) -> List[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)]
    return [s for s in parts if s]


def _overlap_tail(sentences: List[str], overlap: int) -> List[str]:
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


def _sentence_chunk_page(text: str, chunk_size: int, overlap: int) -> List[str]:
    sentences = _split_sentences(text)
    page_chunks: List[str] = []
    buffer: List[str] = []
    for sent in sentences:
        if len(sent) > chunk_size:
            if buffer:
                page_chunks.append(" ".join(buffer))
                buffer = []
            for start in range(0, len(sent), chunk_size):
                piece = sent[start:start + chunk_size]
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
    pages: List[dict],
    doc_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[dict]:
    """
    Input: [{"page_number": int, "text": str}, ...]
    Output: [{"chunk_id": str, "text": str, "page_number": int, "strategy": str}, ...]
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    for page in pages:
        page_number = page["page_number"]
        for i, chunk_text in enumerate(
            _sentence_chunk_page(page["text"], chunk_size, overlap)
        ):
            if len(chunk_text.strip()) < 20:
                continue
            chunks.append({
                "chunk_id": f"{doc_id}_sentence_{page_number}_{i}",
                "text": chunk_text,
                "page_number": page_number,
                "strategy": "sentence"
            })
    return chunks


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
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
) -> List[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    vectors = [embed(s) for s in sentences]
    similarities = [
        _cosine(vectors[i], vectors[i + 1])
        for i in range(len(vectors) - 1)
    ]
    page_chunks: List[str] = []
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
    pages: List[dict],
    doc_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
    threshold: float = 0.7,
) -> List[dict]:
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
            chunks.append({
                "chunk_id": f"{doc_id}_semantic_{page_number}_{i}",
                "text": chunk_text,
                "page_number": page_number,
                "strategy": "semantic"
            })
    return chunks
