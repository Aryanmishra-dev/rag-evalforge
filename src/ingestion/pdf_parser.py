"""PDF text extraction using PyMuPDF."""

import fitz


def parse_pdf(pdf_path: str) -> list[dict]:
    """Extract each PDF page into {"page_number": int, "text": str}.

    page_number is the 1-based PDF page index, matching the page numbers used
    in the eval test set (src/evaluation/test_qa_pairs.json).
    """
    pages = []
    with fitz.open(pdf_path) as doc:
        for i in range(doc.page_count):
            pages.append({"page_number": i + 1, "text": doc.load_page(i).get_text()})
    return pages
