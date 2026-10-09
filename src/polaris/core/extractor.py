"""Extract text and chunk documents for AI search, chat, and summarization with section and code awareness."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import List, Tuple


@dataclass
class DocumentChunk:
    doc_path: str
    page: int
    text: str
    chunk_index: int
    section: str = ""

    @property
    def file_name(self) -> str:
        return Path(self.doc_path).name

    @property
    def extension(self) -> str:
        return Path(self.doc_path).suffix.lower()

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def preview(self, max_chars: int = 120) -> str:
        # Strip internal section/symbol metadata tags for clean text preview
        clean = re.sub(r"^\[(?:Section|Symbol):[^\]]+\]\s*", "", self.text)
        clean = " ".join(clean.split())
        if len(clean) <= max_chars:
            return clean
        return clean[:max_chars].rstrip() + "..."

    def to_dict(self) -> dict:
        return {
            "doc_path": self.doc_path,
            "page": self.page,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "section": self.section,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DocumentChunk:
        return cls(
            doc_path=data["doc_path"],
            page=data["page"],
            text=data["text"],
            chunk_index=data["chunk_index"],
            section=data.get("section", ""),
        )


def compute_file_hash(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash of a file for cache verification and incremental indexing."""
    p = Path(file_path)
    if not p.is_file():
        return ""
    hasher = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 150) -> list[str]:
    """
    Splits text into overlapping windows with boundary awareness.
    Prefers paragraph breaks, sentence boundaries, or line breaks to preserve
    semantic coherence while strictly enforcing chunk_chars size limits.
    """
    if not text or not text.strip():
        return []

    text_len = len(text)
    if text_len <= chunk_chars:
        return [text.strip()]

    chunks: list[str] = []
    start = 0
    overlap = max(0, min(overlap, chunk_chars // 2))

    while start < text_len:
        max_end = min(start + chunk_chars, text_len)
        if max_end == text_len:
            chunk = text[start:max_end].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Look for natural split point within search window
        min_split = start + max(overlap + 10, int(chunk_chars * 0.6))
        window = text[min_split:max_end]

        best_split = -1
        # 1. Paragraph breaks
        p_idx = window.rfind("\n\n")
        if p_idx != -1:
            best_split = min_split + p_idx + 2
        else:
            # 2. Line breaks
            nl_idx = window.rfind("\n")
            if nl_idx != -1:
                best_split = min_split + nl_idx + 1
            else:
                # 3. Sentence boundaries (. ! ?)
                sentence_match = list(re.finditer(r"[.!?]\s+", window))
                if sentence_match:
                    best_split = min_split + sentence_match[-1].end()
                else:
                    # 4. Word boundary
                    space_idx = window.rfind(" ")
                    if space_idx != -1:
                        best_split = min_split + space_idx + 1

        end = best_split if best_split != -1 else max_end
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        step = max(1, (end - start) - overlap)
        start += step

    return chunks


def chunk_markdown(
    text: str,
    file_path: str = "",
    chunk_chars: int = 1200,
    overlap: int = 150,
) -> list[DocumentChunk]:
    """
    Splits Markdown documents with Heading Hierarchy & Code Block awareness.
    Maintains a breadcrumb header stack (e.g. [Section: Title > Subsection])
    and prevents splitting fenced code blocks mid-construct whenever possible.
    """
    if not text or not text.strip():
        return []

    lines = text.splitlines(keepends=True)
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

    sections: list[tuple[str, str]] = []  # (breadcrumb, content)
    current_breadcrumbs: list[tuple[int, str]] = []  # (level, title)
    current_content: list[str] = []
    in_code_fence = False

    def flush_section():
        nonlocal current_content
        has_content = any(not header_pattern.match(l.strip()) for l in current_content if l.strip())
        if has_content:
            body = "".join(current_content).strip()
            breadcrumb_str = " > ".join(t for _, t in current_breadcrumbs)
            sections.append((breadcrumb_str, body))
            current_content = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence

        if not in_code_fence:
            m = header_pattern.match(stripped)
            if m:
                flush_section()
                level = len(m.group(1))
                title = m.group(2).strip()
                while current_breadcrumbs and current_breadcrumbs[-1][0] >= level:
                    current_breadcrumbs.pop()
                current_breadcrumbs.append((level, title))
                current_content.append(line)
                continue

        current_content.append(line)

    body = "".join(current_content).strip()
    if body:
        breadcrumb_str = " > ".join(t for _, t in current_breadcrumbs)
        sections.append((breadcrumb_str, body))

    if not sections:
        sections = [("", text.strip())]

    chunks: list[DocumentChunk] = []
    idx = 0

    for breadcrumb, sec_content in sections:
        header_prefix = f"[Section: {breadcrumb}]\n" if breadcrumb else ""
        avail_chars = max(300, chunk_chars - len(header_prefix))

        sub_chunks = chunk_text(sec_content, chunk_chars=avail_chars, overlap=overlap)
        for piece in sub_chunks:
            full_text = f"{header_prefix}{piece}" if header_prefix else piece
            chunks.append(DocumentChunk(
                doc_path=file_path,
                page=1,
                text=full_text,
                chunk_index=idx,
                section=breadcrumb,
            ))
            idx += 1

    return chunks


def chunk_code(
    text: str,
    file_path: str = "",
    extension: str = "",
    chunk_chars: int = 1200,
    overlap: int = 150,
) -> list[DocumentChunk]:
    """
    Splits Source Code files preserving function & class block boundaries
    and prepending Symbol Breadcrumbs (e.g. [Symbol: def calculate_budget]).
    """
    if not text or not text.strip():
        return []

    lines = text.splitlines(keepends=True)
    ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"

    # Symbol patterns for common languages
    if ext in (".py", ".pyw"):
        symbol_pattern = re.compile(r"^(?:async\s+def|def|class)\s+([A-Za-z0-9_]+)")
    elif ext in (".js", ".jsx", ".ts", ".tsx"):
        symbol_pattern = re.compile(r"^(?:export\s+)?(?:async\s+)?(?:function|class|const|let|var)\s+([A-Za-z0-9_]+)")
    elif ext in (".rs",):
        symbol_pattern = re.compile(r"^(?:pub\s+)?(?:async\s+)?(?:fn|struct|enum|impl|trait)\s+([A-Za-z0-9_]+)")
    elif ext in (".go",):
        symbol_pattern = re.compile(r"^(?:func|type)\s+([A-Za-z0-9_]+)")
    elif ext in (".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".cs", ".java"):
        symbol_pattern = re.compile(r"^(?:public|private|protected|static|\s)*(?:class|interface|void|int|float|double|bool|string|[A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*\(")
    else:
        symbol_pattern = re.compile(r"^(?:def|class|function|func|fn)\s+([A-Za-z0-9_]+)")

    blocks: list[tuple[str, str]] = []  # (symbol_name, code_content)
    current_symbol = ""
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        m = symbol_pattern.match(stripped)
        if m and current_lines:
            block_code = "".join(current_lines).rstrip()
            if block_code:
                blocks.append((current_symbol, block_code))
            current_symbol = m.group(0).strip()
            current_lines = [line]
        else:
            if m and not current_symbol:
                current_symbol = m.group(0).strip()
            current_lines.append(line)

    if current_lines:
        block_code = "".join(current_lines).rstrip()
        if block_code:
            blocks.append((current_symbol, block_code))

    if not blocks:
        blocks = [("", text.strip())]

    chunks: list[DocumentChunk] = []
    idx = 0

    for symbol, block_content in blocks:
        prefix = f"[Symbol: {symbol}]\n" if symbol else ""
        avail_chars = max(300, chunk_chars - len(prefix))

        sub_chunks = chunk_text(block_content, chunk_chars=avail_chars, overlap=overlap)
        for piece in sub_chunks:
            full_text = f"{prefix}{piece}" if prefix else piece
            chunks.append(DocumentChunk(
                doc_path=file_path,
                page=1,
                text=full_text,
                chunk_index=idx,
                section=symbol,
            ))
            idx += 1

    return chunks


def read_text_safe(path: Path) -> str:
    """Reads a text file handling UTF-8, BOMs, UTF-16, and Windows-1252 fallbacks."""
    raw = path.read_bytes()
    if not raw:
        return ""

    # UTF-16 LE / BE detection
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        try:
            return raw.decode("utf-16", errors="replace").strip()
        except Exception:
            pass

    # UTF-8 BOM
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            return raw.decode("utf-8-sig", errors="replace").strip()
        except Exception:
            pass

    # Standard UTF-8
    try:
        return raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        try:
            return raw.decode("cp1252", errors="replace").strip()
        except Exception:
            return raw.decode("utf-8", errors="replace").strip()


def extract_text_from_file(file_path: Path | str) -> list[tuple[int, str]]:
    """
    Extracts text from a file by page/section.
    Returns list of (page_number, text).
    Supports PDF, DOCX, TXT, MD, RST, Code, Configs, and Data files.
    """
    path = Path(file_path)
    if not path.is_file():
        return []

    ext = path.suffix.lower()

    # PDF via PyMuPDF
    if ext == ".pdf":
        try:
            import fitz
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

    # DOCX via python-docx (paragraphs + tables)
    if ext in (".docx", ".doc"):
        try:
            import docx
            doc = docx.Document(str(path))
            parts: list[str] = []
            for p in doc.paragraphs:
                p_text = p.text.strip()
                if p_text:
                    parts.append(p_text)

            for table in doc.tables:
                for row in table.rows:
                    row_cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if row_cells:
                        parts.append(" | ".join(row_cells))

            full_text = "\n".join(parts)
            return [(1, full_text)] if full_text else []
        except Exception:
            return []

    # Text, Markdown, Documentation, Code & Data files
    supported_text_exts = {
        # Documents & notes
        ".txt", ".md", ".markdown", ".rst", ".rtf", ".log", ".tex",
        # Web & Scripts
        ".html", ".htm", ".css", ".scss", ".sass", ".js", ".jsx", ".ts", ".tsx",
        # General programming
        ".py", ".pyw", ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".cs", ".java",
        ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".kts", ".dart", ".lua",
        ".r", ".scala", ".pl", ".pm", ".sh", ".bash", ".zsh", ".bat", ".cmd", ".ps1",
        # Data & Config formats
        ".json", ".jsonl", ".csv", ".tsv", ".xml", ".yaml", ".yml",
        ".toml", ".ini", ".cfg", ".conf", ".sql", ".env"
    }

    if ext in supported_text_exts:
        try:
            content = read_text_safe(path)
            return [(1, content)] if content else []
        except Exception:
            return []

    return []


def chunk_file(file_path: Path | str, chunk_chars: int = 1200, overlap: int = 150) -> list[DocumentChunk]:
    """
    Extracts and chunks a file into searchable pieces.
    Automatically applies Header-Aware chunking for Markdown,
    Symbol & block-preserving chunking for Code,
    and page-aligned boundary chunking for PDFs and documents.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    # 1. Specialized Markdown header-aware chunking
    if ext in (".md", ".markdown"):
        content = read_text_safe(path)
        if content:
            return chunk_markdown(content, file_path=str(path), chunk_chars=chunk_chars, overlap=overlap)
        return []

    # 2. Specialized Code symbol-aware chunking
    code_extensions = {
        ".py", ".pyw", ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".cs", ".java",
        ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".kts", ".dart", ".lua",
        ".js", ".jsx", ".ts", ".tsx", ".sh", ".bash", ".ps1", ".sql"
    }
    if ext in code_extensions:
        content = read_text_safe(path)
        if content:
            return chunk_code(content, file_path=str(path), extension=ext, chunk_chars=chunk_chars, overlap=overlap)
        return []

    # 3. Standard document chunking (PDF, DOCX, TXT, CSV, etc.)
    pages = extract_text_from_file(file_path)
    chunks: list[DocumentChunk] = []
    idx = 0
    for page_num, text in pages:
        page_chunks = chunk_text(text, chunk_chars=overlap, overlap=overlap) if chunk_chars <= overlap else chunk_text(text, chunk_chars, overlap)
        for piece in page_chunks:
            chunks.append(DocumentChunk(
                doc_path=str(file_path),
                page=page_num,
                text=piece,
                chunk_index=idx,
                section="",
            ))
            idx += 1
    return chunks
