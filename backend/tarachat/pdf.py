"""PDF processing: text extraction, validation, page serving, and highlighting."""

import io
import logging
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)


def validate(pdf_bytes: bytes) -> bool:
    """Check whether *pdf_bytes* represent a valid PDF.

    >>> validate(b"not a pdf")
    False
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        _ = doc.page_count
        doc.close()
    except Exception as e:
        logger.warning("PDF validation failed: %s", e)
        return False
    else:
        return True


def extract_text(pdf_bytes: bytes) -> tuple[str, dict]:
    """Extract text and metadata from PDF bytes.

    Args:
        pdf_bytes: PDF file content as bytes.

    Returns:
        Tuple of (extracted_text, metadata).

    Raises:
        ValueError: If *pdf_bytes* cannot be read or the PDF is empty.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Failed to process PDF: {e!s}") from e

    metadata: dict = {
        "num_pages": doc.page_count,
        "file_type": "pdf",
    }
    if doc.metadata:
        for key in ("title", "author", "subject", "creator"):
            value = doc.metadata.get(key)
            if value:
                metadata[key] = value

    text_content: list[str] = []
    for page_num, page in enumerate(doc, start=1):
        try:
            page_text = page.get_text()
            if page_text and page_text.strip():
                text_content.append(f"[Page {page_num}]\n{page_text}")
        except Exception as e:
            logger.warning("Failed to extract text from page %d: %s", page_num, e)

    doc.close()

    if not text_content:
        raise ValueError("No text content could be extracted from the PDF")

    full_text = "\n\n".join(text_content)
    logger.info(
        "Successfully extracted %d characters from %d pages",
        len(full_text),
        metadata["num_pages"],
    )
    return full_text, metadata


def serve(
    pdf_path: Path,
    *,
    page: int | None = None,
    num_pages: int = 1,
    highlights: list[str] | None = None,
) -> bytes:
    """Read a PDF, optionally select pages around *page* and highlight text.

    Args:
        pdf_path: Path to the PDF file on disk.
        page: 1-based page number to centre the selection on.
        num_pages: How many pages to include around *page*.
        highlights: Text snippets to highlight with yellow annotations.

    Returns:
        Modified PDF content as bytes.
    """
    doc = fitz.open(str(pdf_path))

    if page is not None and doc.page_count > num_pages:
        min_page = max(page - num_pages // 2, 1)
        pages_before = page - min_page + 1
        max_page = min(page + num_pages - pages_before, doc.page_count)
        doc.select([p - 1 for p in range(min_page, max_page + 1)])
        logger.info("Selected pages %d-%d of %s", min_page, max_page, pdf_path.name)

    if highlights:
        for pdf_page in doc:
            for text in highlights:
                areas = pdf_page.search_for(text)
                if areas:
                    annot = pdf_page.add_highlight_annot(areas)
                    annot.update()

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()
