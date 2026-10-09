import os
import pytest
from pathlib import Path

# Ensure offscreen Qt platform for testing environments
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from sortpilot.config import Settings
from sortpilot.ui.main_window import MainWindow
from sortpilot.ui.ai_organize_tab import AIOrganizeTab
from sortpilot.ui.chat_tab import ChatTab
from sortpilot.ui.summarize_tab import SummarizeTab
from sortpilot.core.extractor import extract_text_from_file, chunk_file, chunk_text


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_init(qapp, tmp_path):
    settings = Settings(data_dir=tmp_path)
    win = MainWindow()
    assert win.tabs.count() == 3
    assert "AI Organize" in win.tabs.tabText(0)
    assert "Search & Chat" in win.tabs.tabText(1)
    assert "Summarize" in win.tabs.tabText(2)


def test_extractor_and_chunking(tmp_path):
    doc_path = tmp_path / "sample.txt"
    doc_path.write_text("Hello world! " * 200, encoding="utf-8")

    pages = extract_text_from_file(doc_path)
    assert len(pages) == 1
    assert "Hello world!" in pages[0][1]

    chunks = chunk_file(doc_path, chunk_chars=500, overlap=50)
    assert len(chunks) > 1
    assert all(c.doc_path == str(doc_path) for c in chunks)


def test_ai_organize_tab_flow(qapp, tmp_path):
    in_dir = tmp_path / "incoming"
    in_dir.mkdir()
    (in_dir / "invoice_2024.pdf").write_text("pdf dummy content")
    (in_dir / "photo.png").write_text("png dummy content")

    settings = Settings(data_dir=tmp_path / "data")
    tab = AIOrganizeTab(settings)
    tab.src_edit.setText(str(in_dir))

    from sortpilot.core.organizer import plan_by_type
    moves = plan_by_type([str(in_dir / "invoice_2024.pdf"), str(in_dir / "photo.png")], str(in_dir))
    tab._show_preview(moves)

    assert tab.table.rowCount() == 2
    assert tab.apply_btn.isEnabled()


def test_chat_tab_init(qapp, tmp_path):
    settings = Settings(data_dir=tmp_path)
    tab = ChatTab(settings)
    assert tab.folder_edit is not None
    assert tab.chat_browser is not None


def test_summarize_tab_init(qapp, tmp_path):
    settings = Settings(data_dir=tmp_path)
    tab = SummarizeTab(settings)
    assert tab.preset_combo.count() == 3
    assert tab.summary_viewer is not None
