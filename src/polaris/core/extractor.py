"""Extract text and chunk documents for AI search, chat, and summarization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class DocumentChunk:
    doc_path: str
    page: int
    text: str
    chunk_index: int


def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Splits text into overlapping character windows."""
    if not text or not text.strip():
        return []
    chunks = []
    start = 0
    text_len = len(text)
    step = max(1, chunk_chars - overlap)
    while start < text_len:
        end = min(start + chunk_chars, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start += step
    return chunks


def extract_text_from_file(file_path: Path | str) -> list[tuple[int, str]]:
    """
    Extracts text from a file by page/section.
    Returns list of (page_number, text).
    """
    path = Path(file_path)
    if not path.is_file():
        return []

    ext = path.suffix.lower()

    # PDF via PyMuPDF
    if ext == ".pdf":
        try:
            import pymupdf as fitz  # noqa: F401 — pymupdf is the modern API
            doc = fitz.open(str(path))
            pages = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text").strip()
                if text:
                    pages.append((page_num + 1, text))
            return pages
        except Exception:
            return []

    # DOCX via python-docx
    if ext in (".docx", ".doc"):
        try:
            import docx
            doc = docx.Document(str(path))
            full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            return [(1, full_text)] if full_text else []
        except Exception:
            return []

    # Text, Markdown, Code, CSV, JSON
    if ext in (
        ".txt", ".md", ".py", ".js", ".ts", ".html", ".css", ".json",
        ".csv", ".xml", ".yaml", ".yml", ".sql", ".sh", ".bat", ".ps1"
    ):
        try:
            content = path.read_text(encoding="utf-8", errors="replace").strip()
            return [(1, content)] if content else []
        except Exception:
            return []

    return []


def chunk_file(file_path: Path | str, chunk_chars: int = 1200, overlap: int = 150) -> list[DocumentChunk]:
    """Extracts and chunks a file into searchable pieces with page references."""
    pages = extract_text_from_file(file_path)
    chunks = []
    idx = 0
    for page_num, text in pages:
        page_chunks = chunk_text(text, chunk_chars, overlap)
        for piece in page_chunks:
            chunks.append(DocumentChunk(
                doc_path=str(file_path),
                page=page_num,
                text=piece,
                chunk_index=idx
            ))
            idx += 1
    return chunks
