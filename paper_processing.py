from __future__ import annotations

import io
import re
from typing import Any

import fitz  # PyMuPDF


def extract_text_from_pdf(file_obj: Any) -> str:
    """
    Extract text from a PDF-like object using PyMuPDF.

    `file_obj` can be a Streamlit UploadedFile or any file-like object supporting .read().
    """
    # Streamlit's UploadedFile supports .read() and .getvalue()
    if hasattr(file_obj, "getvalue"):
        data = file_obj.getvalue()
    else:
        data = file_obj.read()

    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("Expected PDF bytes for extraction.")

    text_parts = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())

    if not text_parts:
        raise ValueError("No extractable text found in PDF.")

    raw_text = "\n".join(text_parts)
    return clean_paper_text(raw_text)


def clean_paper_text(text: str) -> str:
    """
    Perform light cleaning on extracted paper text.

    - Normalize line breaks and whitespace.
    - Remove obvious standalone page numbers.
    """
    # Replace Windows/Mac newlines with standard
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove isolated page numbers on their own lines (simple heuristic)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # Collapse multiple blank lines into at most two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse repeated spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def prepare_paper_context(raw_text: str, max_chars: int = 30000) -> str:
    """
    Clean and, if necessary, truncate the paper text for use with LLMs.

    For v1 we simply truncate by character count and append a note when truncated.
    """
    cleaned = clean_paper_text(raw_text)
    if len(cleaned) <= max_chars:
        return cleaned
    truncated = cleaned[:max_chars]
    return truncated.rstrip() + "\n\n[Truncated for length; content after this point was omitted.]"


