"""CLI to parse, chunk, embed, and store a PDF into all four strategy collections."""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import EMBED_MODEL
from src.embeddings.chroma_client import (
    add_chunks,
    collection_name,
    get_collection_for_model,
)
from src.ingestion.chunkers import (
    chunk_fixed,
    chunk_recursive,
    chunk_semantic,
    chunk_sentence,
)
from src.ingestion.pdf_parser import parse_pdf

DEFAULT_PDF = Path("data/raw_pdfs/9788498803488_L33_23.pdf")

STRATEGIES = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "sentence": chunk_sentence,
    "semantic": chunk_semantic,
}


def generate_doc_id(pages: list[dict]) -> str:
    """Content-hash of the parsed text; stable across re-ingests of the same file."""
    digest = hashlib.sha256("".join(p["text"] for p in pages).encode()).hexdigest()
    return digest[:16]


def ingest(pdf_path: Path, embed_model: str = EMBED_MODEL) -> None:
    """Parse `pdf_path` and index one chunk per strategy into ChromaDB.

    Chunks are stored in the collection namespace for ``embed_model`` so the
    same document can be benchmarked under multiple embedding models.
    """
    pages = parse_pdf(str(pdf_path))
    doc_id = generate_doc_id(pages)
    print(f"Parsed {len(pages)} pages | doc_id={doc_id} | embed_model={embed_model}")

    for strategy, chunker in STRATEGIES.items():
        chunks = chunker(pages, doc_id)
        collection = get_collection_for_model(strategy, embed_model)
        add_chunks(collection, chunks)
        print(f"  {collection_name(strategy, embed_model)}: {len(chunks)} chunks")


def main() -> None:
    """Run the ingest CLI."""
    parser = argparse.ArgumentParser(description="Ingest a PDF into all 4 strategy collections.")
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=DEFAULT_PDF,
        help=f"PDF to ingest (default: {DEFAULT_PDF})",
    )
    parser.add_argument(
        "--embed-model",
        default=EMBED_MODEL,
        help=f"Embedding model to use (default: {EMBED_MODEL}).",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")
    ingest(args.pdf, embed_model=args.embed_model)


if __name__ == "__main__":
    main()
