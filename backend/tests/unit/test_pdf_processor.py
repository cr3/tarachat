import io

import pytest
from PyPDF2 import PdfReader, PdfWriter

from tarachat.pdf_processor import PDFProcessor


@pytest.fixture
def processor():
    return PDFProcessor()


def _make_pdf(text: str = "Hello world") -> bytes:
    """Build a minimal in-memory PDF with the given text."""
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    # Annotate won't add extractable text, but the page exists.
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _make_pdf_bytes() -> bytes:
    """Create a real PDF via PyPDF2."""
    return _make_pdf()


class TestValidatePdf:
    def test_invalid_bytes(self, processor):
        assert processor.validate_pdf(b"not a pdf") is False

    def test_empty_bytes(self, processor):
        assert processor.validate_pdf(b"") is False

    def test_valid_pdf(self, processor):
        assert processor.validate_pdf(_make_pdf_bytes()) is True


class TestExtractTextFromPdf:
    def test_invalid_pdf_raises(self, processor):
        with pytest.raises(ValueError, match="Failed to process PDF"):
            processor.extract_text_from_pdf(b"not a pdf")

    def test_empty_bytes_raises(self, processor):
        with pytest.raises(ValueError):
            processor.extract_text_from_pdf(b"")

    def test_blank_pdf_raises_no_content(self, processor):
        """A valid PDF with a blank page has no extractable text."""
        with pytest.raises(ValueError, match="No text content"):
            processor.extract_text_from_pdf(_make_pdf_bytes())


class TestExtractMetadata:
    def test_returns_page_count(self, processor):
        reader = PdfReader(io.BytesIO(_make_pdf_bytes()))
        meta = processor._extract_metadata(reader)
        assert meta["num_pages"] == 1
        assert meta["file_type"] == "pdf"


class TestExtractPages:
    def test_blank_page_returns_empty(self, processor):
        reader = PdfReader(io.BytesIO(_make_pdf_bytes()))
        pages = processor._extract_pages(reader)
        assert pages == []
