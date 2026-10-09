"""Dedicated test suite for Role 2: Document Extraction, FAISS Indexing, Hybrid RAG, and Chat UI."""
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication

from polaris.core.extractor import (
    DocumentChunk,
    chunk_text,
    chunk_markdown,
    chunk_code,
    extract_text_from_file,
    chunk_file,
    read_text_safe,
    compute_file_hash,
)
from polaris.ai.rag import (
    BM25Index,
    RagEngine,
    format_chat_export,
    reciprocal_rank_fusion,
    expand_query_terms,
    compress_history,
    STOPWORDS,
)
from polaris.ui.chat_tab import ChatTab, SourceViewerDialog, StreamQueryWorker


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["--platform", "offscreen"])
    yield app
    app.processEvents()


# ---------------------------------------------------------------------------
# 1. Extractor, Markdown & Code Chunking Tests
# ---------------------------------------------------------------------------

def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_splitting():
    text = "A" * 500 + " " + "B" * 500 + " " + "C" * 500
    chunks = chunk_text(text, chunk_chars=600, overlap=100)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 600


def test_chunk_text_boundary_awareness():
    text = (
        "First sentence here. Second sentence continues the explanation. "
        "Third sentence introduces key concepts. Fourth sentence concludes this paragraph.\n\n"
        "Fifth sentence begins a completely new topic. Sixth sentence expands on it."
    )
    chunks = chunk_text(text, chunk_chars=120, overlap=30)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 120


def test_chunk_markdown_hierarchy_and_breadcrumbs():
    md = """# Polaris System Architecture
## Core Ingestion Pipeline
The ingestion pipeline streams incoming events into Kafka topics.
### Kafka Consumer Group
Consumer group polaris-workers processes 50,000 events per second.
## Retrieval Engine
Retrieval combines BM25 and vector embeddings."""

    chunks = chunk_markdown(md, file_path="arch.md", chunk_chars=500, overlap=50)
    assert len(chunks) >= 3

    consumer_chunk = next(c for c in chunks if "polaris-workers" in c.text)
    assert "Polaris System Architecture > Core Ingestion Pipeline > Kafka Consumer Group" in consumer_chunk.section
    assert "[Section: Polaris System Architecture > Core Ingestion Pipeline > Kafka Consumer Group]" in consumer_chunk.text

    retrieval_chunk = next(c for c in chunks if "Retrieval combines" in c.text)
    assert "Polaris System Architecture > Retrieval Engine" in retrieval_chunk.section


def test_chunk_markdown_code_fence_preservation():
    md = """# Developer Guide
## Python Example
Here is the code snippet:
```python
# This is a comment inside code fence, not a markdown header
def process():
    return 42
```
Concluding remarks."""

    chunks = chunk_markdown(md, file_path="guide.md", chunk_chars=500, overlap=50)
    assert len(chunks) >= 1
    chunk = chunks[0]
    assert "def process():" in chunk.text
    assert "```python" in chunk.text
    assert "This is a comment inside code fence" not in chunk.section


def test_chunk_code_symbols():
    code = """import os
import sys

def calculate_budget(total, tax_rate):
    \"\"\"Calculates net budget after taxes.\"\"\"
    return total * (1.0 - tax_rate)

class ReportGenerator:
    def __init__(self, title):
        self.title = title

    def generate(self):
        return f"Report: {self.title}"
"""
    chunks = chunk_code(code, file_path="budget.py", extension=".py", chunk_chars=400, overlap=50)
    assert len(chunks) >= 2

    budget_chunk = next(c for c in chunks if "def calculate_budget" in c.text)
    assert "calculate_budget" in budget_chunk.section
    assert "[Symbol: def calculate_budget]" in budget_chunk.text

    class_chunk = next(c for c in chunks if "class ReportGenerator" in c.text)
    assert "ReportGenerator" in class_chunk.section
    assert "[Symbol: class ReportGenerator]" in class_chunk.text


def test_chunk_file_routes_by_extension(tmp_path):
    md_file = tmp_path / "spec.md"
    md_file.write_text("# Main Spec\n## Detail\nSpecification text here.", encoding="utf-8")
    md_chunks = chunk_file(md_file)
    assert len(md_chunks) >= 1
    assert "Main Spec > Detail" in md_chunks[0].section

    py_file = tmp_path / "app.py"
    py_file.write_text("def run():\n    return 'running'\n", encoding="utf-8")
    py_chunks = chunk_file(py_file)
    assert len(py_chunks) >= 1
    assert "run" in py_chunks[0].section


def test_document_chunk_properties(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("# Overview\n\nPolaris is a local assistant.", encoding="utf-8")

    chunks = chunk_file(p, chunk_chars=200, overlap=50)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.page == 1
    assert chunk.file_name == "notes.md"
    assert chunk.extension == ".md"
    assert "Polaris is a local assistant" in chunk.text
    assert chunk.char_count > 0
    assert chunk.word_count > 0
    preview = chunk.preview(max_chars=25)
    assert len(preview) <= 28
    assert "Polaris" in preview

    d = chunk.to_dict()
    assert d["doc_path"] == str(p)
    assert d["page"] == 1
    assert "section" in d
    restored = DocumentChunk.from_dict(d)
    assert restored.doc_path == chunk.doc_path
    assert restored.text == chunk.text
    assert restored.section == chunk.section


def test_compute_file_hash(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("Unique content 123", encoding="utf-8")
    h1 = compute_file_hash(f)
    assert len(h1) == 64

    f.write_text("Unique content 456", encoding="utf-8")
    h2 = compute_file_hash(f)
    assert h1 != h2

    assert compute_file_hash(tmp_path / "nonexistent.txt") == ""


def test_read_text_safe_encodings(tmp_path):
    bom_file = tmp_path / "utf8_bom.txt"
    bom_file.write_bytes(b"\xef\xbb\xbfHello with BOM")
    assert read_text_safe(bom_file) == "Hello with BOM"

    utf16_file = tmp_path / "utf16.txt"
    utf16_file.write_bytes("Hello UTF16".encode("utf-16"))
    assert "Hello UTF16" in read_text_safe(utf16_file)


def test_extract_code_file(tmp_path):
    p = tmp_path / "script.py"
    p.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    pages = extract_text_from_file(p)
    assert len(pages) == 1
    assert pages[0][0] == 1
    assert "def hello():" in pages[0][1]


def test_extract_expanded_formats(tmp_path):
    sql = tmp_path / "query.sql"
    sql.write_text("SELECT id, name FROM users WHERE active = 1;", encoding="utf-8")
    assert len(extract_text_from_file(sql)) == 1

    rs = tmp_path / "main.rs"
    rs.write_text("fn main() { println!(\"Hello\"); }", encoding="utf-8")
    assert len(extract_text_from_file(rs)) == 1

    yml = tmp_path / "config.yaml"
    yml.write_text("app:\n  name: polaris\n  port: 8080\n", encoding="utf-8")
    assert len(extract_text_from_file(yml)) == 1


def test_extract_docx_paragraphs_and_tables(tmp_path):
    docx_file = tmp_path / "test.docx"
    docx_file.touch()

    mock_p1 = MagicMock()
    mock_p1.text = "Introduction to Polaris"
    mock_p2 = MagicMock()
    mock_p2.text = "Second paragraph"

    mock_cell1 = MagicMock()
    mock_cell1.text = "Feature"
    mock_cell2 = MagicMock()
    mock_cell2.text = "Status"
    mock_row = MagicMock()
    mock_row.cells = [mock_cell1, mock_cell2]

    mock_table = MagicMock()
    mock_table.rows = [mock_row]

    mock_doc = MagicMock()
    mock_doc.paragraphs = [mock_p1, mock_p2]
    mock_doc.tables = [mock_table]

    with patch("docx.Document", return_value=mock_doc):
        pages = extract_text_from_file(docx_file)
        assert len(pages) == 1
        text = pages[0][1]
        assert "Introduction to Polaris" in text
        assert "Feature | Status" in text


def test_extract_pdf_pages(tmp_path):
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.touch()

    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Page 1 Content"
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Page 2 Content"

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 2
    mock_doc.load_page.side_effect = [mock_page1, mock_page2]

    with patch("fitz.open", return_value=mock_doc):
        pages = extract_text_from_file(pdf_file)
        assert len(pages) == 2
        assert pages[0] == (1, "Page 1 Content")
        assert pages[1] == (2, "Page 2 Content")


def test_extract_missing_file_or_dir(tmp_path):
    assert extract_text_from_file(tmp_path / "nonexistent.txt") == []
    assert extract_text_from_file(tmp_path) == []


# ---------------------------------------------------------------------------
# 2. BM25 & Reciprocal Rank Fusion (RRF) Tests
# ---------------------------------------------------------------------------

def test_bm25_empty():
    bm25 = BM25Index()
    bm25.build([])
    assert bm25.search("test") == []


def test_bm25_tokenization_and_stopwords():
    tokens = BM25Index.tokenize("The quick brown fox is jumping over the lazy dog!")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "over" not in tokens
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens


def test_bm25_ranking_accuracy():
    chunks = [
        DocumentChunk("doc1.txt", 1, "The quarterly financial earnings report shows high revenue.", 0),
        DocumentChunk("doc2.txt", 1, "Cooking pasta requires boiling water and adding salt.", 1),
        DocumentChunk("doc3.txt", 1, "Quarterly financial predictions for next fiscal quarter.", 2),
    ]
    bm25 = BM25Index()
    bm25.build(chunks)

    results = bm25.search("quarterly financial earnings")
    assert len(results) >= 2
    top_doc_idx = results[0][0]
    assert top_doc_idx == 0
    matched_indices = [idx for idx, _ in results]
    assert 1 not in matched_indices


def test_bm25_candidate_indices():
    chunks = [
        DocumentChunk("file1.txt", 1, "System architecture and vector database design.", 0),
        DocumentChunk("file2.py", 1, "System architecture implementation in python.", 1),
    ]
    bm25 = BM25Index()
    bm25.build(chunks)

    results = bm25.search("architecture", candidate_indices={1})
    assert len(results) == 1
    assert results[0][0] == 1


def test_bm25_filename_boost():
    chunks = [
        DocumentChunk("budget_report.txt", 1, "Total expenditures and numbers.", 0),
        DocumentChunk("other.txt", 1, "Mentions budget once in the text.", 1),
    ]
    bm25 = BM25Index()
    bm25.build(chunks)

    results = bm25.search("budget_report")
    assert len(results) >= 1
    assert results[0][0] == 0


def test_reciprocal_rank_fusion_math():
    ranked_lists = [[10, 20], [20, 10]]
    rrf = reciprocal_rank_fusion(ranked_lists, k=60)
    assert len(rrf) == 2
    expected_score = (1.0 / 61) + (1.0 / 62)
    assert pytest.approx(rrf[0][1], rel=1e-4) == expected_score
    assert pytest.approx(rrf[1][1], rel=1e-4) == expected_score


def test_reciprocal_rank_fusion_disjoint():
    ranked_lists = [[100], [200]]
    rrf = reciprocal_rank_fusion(ranked_lists, k=60)
    assert len(rrf) == 2
    assert rrf[0][1] == pytest.approx(1.0 / 61)
    assert rrf[1][1] == pytest.approx(1.0 / 61)


def test_reciprocal_rank_fusion_empty():
    assert reciprocal_rank_fusion([]) == []


# ---------------------------------------------------------------------------
# 3. Query Expansion & HyDE Tests
# ---------------------------------------------------------------------------

def test_expand_query_terms():
    q = "budget forecast"
    hyp = "The fiscal budget forecast shows revenue growth and expenditure decrease."
    expanded = expand_query_terms(q, hyp, max_terms=3)
    assert "budget forecast" in expanded
    assert "fiscal" in expanded or "revenue" in expanded or "expenditure" in expanded


def test_expand_query_terms_empty():
    assert expand_query_terms("test", "") == "test"


def test_rag_hyde_search(tmp_path):
    (tmp_path / "apollo.md").write_text("# Project Apollo\nHardware constraint specifies maximum 8GB system RAM.", encoding="utf-8")

    engine = RagEngine()

    def mock_hyde_chat(settings, messages, options=None, schema=None):
        return "Apollo hardware constraints limit RAM usage to eight gigabytes."

    with patch("polaris.ai.rag.embed", side_effect=_mock_embed), \
         patch("polaris.ai.rag.chat", side_effect=mock_hyde_chat):

        engine.index_folder(str(tmp_path))
        results = engine.search("RAM limitation?", top_k=1, use_hyde=True)
        assert len(results) == 1
        assert "8GB system RAM" in results[0][0].text


# ---------------------------------------------------------------------------
# 4. Persistent Caching & Incremental Indexing Tests
# ---------------------------------------------------------------------------

def _mock_embed(settings, texts):
    """Produces deterministic 16-dim normalized embeddings based on character sum."""
    vecs = []
    for t in texts:
        v = np.zeros(16, dtype=np.float32)
        idx = hash(t[:10]) % 16
        v[idx] = 1.0
        vecs.append(v.tolist())
    return vecs


def test_persistent_cache_save_and_load(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text("Comprehensive architecture guide.", encoding="utf-8")

    cache_root = tmp_path / "cache_store"
    from polaris.config import Settings
    custom_settings = Settings(data_dir=cache_root)

    engine = RagEngine(custom_settings)

    with patch("polaris.ai.rag.embed", side_effect=_mock_embed):
        count = engine.index_folder(str(docs_dir))
        assert count == 1
        assert engine.has_cache(str(docs_dir)) is True

    engine_reloaded = RagEngine(custom_settings)
    assert engine_reloaded.chunks == []
    assert engine_reloaded.load_cache(str(docs_dir)) is True
    assert len(engine_reloaded.chunks) == 1
    assert engine_reloaded.chunks[0].file_name == "guide.md"
    assert engine_reloaded.index is not None

    results = engine_reloaded.search("architecture", top_k=1)
    assert len(results) == 1
    assert "architecture" in results[0][0].text


def test_incremental_indexing_efficiency(tmp_path):
    docs_dir = tmp_path / "my_docs"
    docs_dir.mkdir()
    f1 = docs_dir / "doc1.txt"
    f1.write_text("First document content.", encoding="utf-8")
    f2 = docs_dir / "doc2.txt"
    f2.write_text("Second document content.", encoding="utf-8")

    from polaris.config import Settings
    custom_settings = Settings(data_dir=tmp_path / "rag_cache")
    engine = RagEngine(custom_settings)

    embed_call_count = 0
    def counting_mock_embed(settings, texts):
        nonlocal embed_call_count
        embed_call_count += len(texts)
        return _mock_embed(settings, texts)

    with patch("polaris.ai.rag.embed", side_effect=counting_mock_embed):
        c1 = engine.index_folder(str(docs_dir))
        assert c1 == 2
        initial_calls = embed_call_count
        assert initial_calls == 2

        # Second index run with 0 changes -> zero new embeddings needed!
        c2 = engine.index_folder(str(docs_dir))
        assert c2 == 2
        assert embed_call_count == initial_calls

        # Add 3rd document -> only the new document gets embedded!
        f3 = docs_dir / "doc3.txt"
        f3.write_text("Third document content.", encoding="utf-8")
        c3 = engine.index_folder(str(docs_dir))
        assert c3 == 3
        assert embed_call_count == initial_calls + 1


# ---------------------------------------------------------------------------
# 5. Top-K Tuning & Score Thresholding Tests
# ---------------------------------------------------------------------------

def test_rag_top_k_and_score_threshold(tmp_path):
    (tmp_path / "item.txt").write_text("Neural networks deep learning model.", encoding="utf-8")

    engine = RagEngine()

    with patch("polaris.ai.rag.embed", side_effect=_mock_embed):
        engine.index_folder(str(tmp_path))

        res_normal = engine.search("neural networks", top_k=1, score_threshold=0.0)
        assert len(res_normal) == 1

        res_filtered = engine.search("neural networks", top_k=1, score_threshold=999.0)
        assert len(res_filtered) == 0


def test_rag_pure_vector_search(tmp_path):
    (tmp_path / "item.txt").write_text("Machine learning neural networks.", encoding="utf-8")

    engine = RagEngine()

    with patch("polaris.ai.rag.embed", side_effect=_mock_embed):
        engine.index_folder(str(tmp_path))

        results = engine.search("neural networks", top_k=1, hybrid=False)
        assert len(results) == 1
        assert "neural networks" in results[0][0].text


def test_rag_file_type_filtering(tmp_path):
    (tmp_path / "doc1.txt").write_text("Text content here.", encoding="utf-8")
    (tmp_path / "code.py").write_text("def run(): pass", encoding="utf-8")

    engine = RagEngine()

    with patch("polaris.ai.rag.embed", side_effect=_mock_embed):
        engine.index_folder(str(tmp_path))

        py_results = engine.search("run", top_k=5, file_types=[".py"])
        assert len(py_results) == 1
        assert py_results[0][0].extension == ".py"

        py_results_nodot = engine.search("run", top_k=5, file_types=["py"])
        assert len(py_results_nodot) == 1
        assert py_results_nodot[0][0].extension == ".py"

        pdf_results = engine.search("run", top_k=5, file_types=[".pdf"])
        assert len(pdf_results) == 0


def test_rag_hybrid_search_keyword_boost(tmp_path):
    (tmp_path / "invoice.txt").write_text("Invoice reference ID: INV-998877 from ACME Corp.", encoding="utf-8")
    (tmp_path / "other.txt").write_text("General company memo regarding equipment.", encoding="utf-8")

    engine = RagEngine()

    with patch("polaris.ai.rag.embed", side_effect=_mock_embed):
        engine.index_folder(str(tmp_path))

        results = engine.search("INV-998877 ACME", top_k=2, hybrid=True)
        assert len(results) >= 1
        assert "INV-998877" in results[0][0].text


# ---------------------------------------------------------------------------
# 6. Multi-Turn Context Compression Tests
# ---------------------------------------------------------------------------

def test_compress_history_short():
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    compressed = compress_history(history, max_recent_turns=2)
    assert compressed == history


def test_compress_history_long():
    history = [
        {"role": "user", "content": "Turn 1: What is the company budget?"},
        {"role": "assistant", "content": "Turn 1 answer: 10 million dollars."},
        {"role": "user", "content": "Turn 2: How much allocated for R&D?"},
        {"role": "assistant", "content": "Turn 2 answer: 3 million dollars."},
        {"role": "user", "content": "Turn 3: What about marketing?"},
        {"role": "assistant", "content": "Turn 3 answer: 2 million dollars."},
    ]
    compressed = compress_history(history, max_recent_turns=1)
    assert len(compressed) == 3
    assert compressed[0]["role"] == "system"
    assert "Summary of earlier discussion:" in compressed[0]["content"]
    assert "Turn 1" in compressed[0]["content"]
    assert compressed[1] == history[-2]
    assert compressed[2] == history[-1]


# ---------------------------------------------------------------------------
# 7. Chat Streaming & Export Tests
# ---------------------------------------------------------------------------

def test_chat_with_docs_streaming(tmp_path):
    (tmp_path / "report.md").write_text("Total profit was 1.2 million.", encoding="utf-8")

    engine = RagEngine()

    def mock_chat_stream(settings, messages, options=None, schema=None):
        yield "The "
        yield "profit was "
        yield "1.2M."

    with patch("polaris.ai.rag.embed", side_effect=_mock_embed), \
         patch("polaris.ai.rag.chat_stream", side_effect=mock_chat_stream):

        engine.index_folder(str(tmp_path))
        answer, cited = engine.chat_with_docs("What was the profit?")
        assert answer == "The profit was 1.2M."
        assert len(cited) >= 1
        assert cited[0].file_name == "report.md"


def test_chat_with_docs_empty_index():
    engine = RagEngine()

    def mock_chat_stream(settings, messages, options=None, schema=None):
        yield "General answer."

    with patch("polaris.ai.rag.chat_stream", side_effect=mock_chat_stream):
        answer, cited = engine.chat_with_docs("Hello?")
        assert answer == "General answer."
        assert cited == []


def test_format_chat_export():
    history = [
        {"role": "user", "content": "What is Polaris?"},
        {
            "role": "assistant",
            "content": "Polaris is a local AI desktop assistant.",
            "cited": [DocumentChunk(doc_path="C:/docs/spec.md", page=1, text="Polaris spec", chunk_index=0, section="Overview")]
        }
    ]
    md = format_chat_export(history, title="Test Transcript")
    assert "# Test Transcript" in md
    assert "### 👤 User" in md
    assert "What is Polaris?" in md
    assert "### 🤖 Polaris Assistant" in md
    assert "#### 📄 Sources & Citations" in md
    assert "spec.md" in md
    assert "Overview" in md
    assert "> Polaris spec" in md


# ---------------------------------------------------------------------------
# 8. UI Component Tests (Controls, Stop Streaming, Dialog)
# ---------------------------------------------------------------------------

def test_chat_tab_ui_controls(qapp):
    tab = ChatTab()
    try:
        assert tab.file_type_combo.count() == 4
        assert tab.inspect_sources_btn.isEnabled() is False
        assert tab.top_k_spin.value() == 3
        assert tab.threshold_combo.count() == 3
        assert tab.hyde_check is not None
        assert tab.is_streaming_active is False

        tab.file_type_combo.setCurrentIndex(1)
        assert tab._get_selected_file_types() == [".pdf"]

        tab.file_type_combo.setCurrentIndex(0)
        assert tab._get_selected_file_types() is None

        tab.threshold_combo.setCurrentIndex(0)
        assert tab._get_selected_threshold() == 0.0
        tab.threshold_combo.setCurrentIndex(1)
        assert tab._get_selected_threshold() == 0.012

        tab.last_cited_chunks = [DocumentChunk("test.txt", 1, "sample", 0)]
        tab.inspect_sources_btn.setEnabled(True)
        tab.clear_chat()
        assert len(tab.last_cited_chunks) == 0
        assert tab.inspect_sources_btn.isEnabled() is False
    finally:
        tab.deleteLater()
        qapp.processEvents()


def test_stream_worker_stop(qapp):
    engine = RagEngine()
    worker = StreamQueryWorker(engine, "test question", [])
    assert worker.is_stopped is False
    worker.stop()
    assert worker.is_stopped is True


def test_chat_tab_stop_generation(qapp):
    tab = ChatTab()
    try:
        tab.is_streaming_active = True
        tab.send_btn.setText("⏹ Stop")
        tab.stop_generation()
        assert tab.is_streaming_active is False
        assert tab.send_btn.text() == "Ask AI"
    finally:
        tab.deleteLater()
        qapp.processEvents()


def test_chat_tab_anchor_click(qapp):
    tab = ChatTab()
    try:
        tab.last_cited_chunks = [
            DocumentChunk("c:/sample/test.pdf", 1, "Snippet content here.", 0, section="Intro")
        ]
        with patch.object(tab, "show_source_viewer") as mock_show:
            tab._on_anchor_clicked(QUrl("citation:0"))
            mock_show.assert_called_once_with(initial_index=0)
    finally:
        tab.deleteLater()
        qapp.processEvents()


def test_source_viewer_dialog(qapp):
    chunks = [
        DocumentChunk("c:/sample/report.pdf", 1, "Page 1 revenue was high.", 0, section="Revenue"),
        DocumentChunk("c:/sample/report.pdf", 2, "Page 2 expenses were low.", 1, section="Expenses"),
    ]
    dialog = SourceViewerDialog(chunks, initial_index=0)
    try:
        assert dialog.chunk_list.count() == 2

        assert "report.pdf" in dialog.meta_label.text()
        assert "Revenue" in dialog.meta_label.text()
        assert "Page 1 revenue was high." in dialog.text_preview.toPlainText()

        dialog.chunk_list.setCurrentRow(1)
        assert "Expenses" in dialog.meta_label.text()
        assert "Page 2 expenses were low." in dialog.text_preview.toPlainText()

        dialog.search_edit.setText("expenses")
        assert "expenses" in dialog.text_preview.toHtml()
    finally:
        dialog.deleteLater()
        qapp.processEvents()
