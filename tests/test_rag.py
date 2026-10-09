"""Dedicated test suite for Role 2: Document Extraction, FAISS Indexing, Hybrid RAG, and Chat UI."""
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from PySide6.QtWidgets import QApplication

from polaris.core.extractor import DocumentChunk, chunk_text, extract_text_from_file, chunk_file
from polaris.ai.rag import RagEngine, format_chat_export, STOPWORDS
from polaris.ui.chat_tab import ChatTab, SourceViewerDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["--platform", "offscreen"])
    return app


# ---------------------------------------------------------------------------
# 1. Extractor Tests
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


def test_chunk_file_and_properties(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("# Overview\n\nPolaris is a local assistant.", encoding="utf-8")

    chunks = chunk_file(p, chunk_chars=200, overlap=50)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.page == 1
    assert chunk.file_name == "notes.md"
    assert chunk.extension == ".md"
    assert "Polaris is a local assistant" in chunk.text


def test_extract_code_file(tmp_path):
    p = tmp_path / "script.py"
    p.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    pages = extract_text_from_file(p)
    assert len(pages) == 1
    assert pages[0][0] == 1
    assert "def hello():" in pages[0][1]


# ---------------------------------------------------------------------------
# 2. RAG Engine, FAISS Indexing & Hybrid Search Tests
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


def test_rag_engine_index_and_search(tmp_path):
    (tmp_path / "doc1.txt").write_text("Financial quarterly earnings report for 2024.", encoding="utf-8")
    (tmp_path / "doc2.py").write_text("def calculate_budget(): return 50000\n", encoding="utf-8")

    engine = RagEngine()

    with patch("polaris.ai.rag.embed", side_effect=_mock_embed):
        count = engine.index_folder(str(tmp_path))
        assert count == 2
        assert len(engine.chunks) == 2
        assert engine.index is not None

        # Search without filter
        results = engine.search("quarterly earnings", top_k=2)
        assert len(results) >= 1
        assert isinstance(results[0][0], DocumentChunk)
        assert isinstance(results[0][1], float)


def test_rag_file_type_filtering(tmp_path):
    (tmp_path / "doc1.txt").write_text("Text content here.", encoding="utf-8")
    (tmp_path / "code.py").write_text("def run(): pass", encoding="utf-8")

    engine = RagEngine()

    with patch("polaris.ai.rag.embed", side_effect=_mock_embed):
        engine.index_folder(str(tmp_path))

        # Search filtered only to Python files
        py_results = engine.search("run", top_k=5, file_types=[".py"])
        assert len(py_results) == 1
        assert py_results[0][0].extension == ".py"

        # Search for non-existent extension returns empty
        pdf_results = engine.search("run", top_k=5, file_types=[".pdf"])
        assert len(pdf_results) == 0


def test_rag_hybrid_search_keyword_boost(tmp_path):
    # One file has the exact unique ID 'INV-998877'
    (tmp_path / "invoice.txt").write_text("Invoice reference ID: INV-998877 from ACME Corp.", encoding="utf-8")
    (tmp_path / "other.txt").write_text("General company memo regarding equipment.", encoding="utf-8")

    engine = RagEngine()

    with patch("polaris.ai.rag.embed", side_effect=_mock_embed):
        engine.index_folder(str(tmp_path))

        # Hybrid search for the exact invoice keyword
        results = engine.search("INV-998877 ACME", top_k=2, hybrid=True)
        assert len(results) >= 1
        assert "INV-998877" in results[0][0].text


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


def test_format_chat_export():
    history = [
        {"role": "user", "content": "What is Polaris?"},
        {
            "role": "assistant",
            "content": "Polaris is a local AI desktop assistant.",
            "cited": [DocumentChunk(doc_path="C:/docs/spec.md", page=1, text="Polaris spec", chunk_index=0)]
        }
    ]
    md = format_chat_export(history, title="Test Transcript")
    assert "# Test Transcript" in md
    assert "### 👤 User" in md
    assert "What is Polaris?" in md
    assert "### 🤖 Polaris Assistant" in md
    assert "#### 📄 Sources & Citations" in md
    assert "spec.md" in md


# ---------------------------------------------------------------------------
# 3. UI Component Tests
# ---------------------------------------------------------------------------

def test_chat_tab_ui_controls(qapp):
    tab = ChatTab()
    assert tab.file_type_combo.count() == 4
    assert tab.inspect_sources_btn.isEnabled() is False

    # Check file type mapping
    tab.file_type_combo.setCurrentIndex(1)  # PDFs
    assert tab._get_selected_file_types() == [".pdf"]

    tab.file_type_combo.setCurrentIndex(0)  # All
    assert tab._get_selected_file_types() is None

    # Clear chat resets state
    tab.last_cited_chunks = [DocumentChunk("test.txt", 1, "sample", 0)]
    tab.inspect_sources_btn.setEnabled(True)
    tab.clear_chat()
    assert len(tab.last_cited_chunks) == 0
    assert tab.inspect_sources_btn.isEnabled() is False


def test_source_viewer_dialog(qapp):
    chunks = [
        DocumentChunk("c:/sample/report.pdf", 1, "Page 1 revenue was high.", 0),
        DocumentChunk("c:/sample/report.pdf", 2, "Page 2 expenses were low.", 1),
    ]
    dialog = SourceViewerDialog(chunks)
    assert dialog.chunk_list.count() == 2

    # Verify first chunk preview
    assert "report.pdf" in dialog.meta_label.text()
    assert "Page 1 revenue was high." in dialog.text_preview.toPlainText()

    # Selecting second chunk updates preview
    dialog.chunk_list.setCurrentRow(1)
    assert "Page 2 expenses were low." in dialog.text_preview.toPlainText()
