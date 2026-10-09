import os
import pytest
from pathlib import Path

# Ensure offscreen Qt platform for testing environments
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from polaris.config import Settings
from polaris.ui.main_window import MainWindow
from polaris.ui.ai_organize_tab import AIOrganizeTab
from polaris.ui.chat_tab import ChatTab
from polaris.ui.summarize_tab import SummarizeTab
from polaris.core.extractor import extract_text_from_file, chunk_file, chunk_text


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

    from polaris.core.organizer import plan_by_type
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
    assert tab.mode_combo.count() == 3
    assert tab.input_stack.count() == 2


def test_summarize_tab_mode_switching(qapp, tmp_path):
    settings = Settings(data_dir=tmp_path)
    tab = SummarizeTab(settings)

    # Default is summary
    assert tab.mode_combo.currentData() == "summary"
    assert tab.input_stack.currentIndex() == 0
    assert not tab.single_opts_widget.isHidden()
    assert "Summarize" in tab.run_btn.text()

    # Switch to entity mode
    tab.mode_combo.setCurrentIndex(1)
    assert tab.mode_combo.currentData() == "entity"
    assert tab.input_stack.currentIndex() == 0
    assert tab.single_opts_widget.isHidden()
    assert "Extract" in tab.run_btn.text()

    # Switch to multidoc mode
    tab.mode_combo.setCurrentIndex(2)
    assert tab.mode_combo.currentData() == "multidoc"
    assert tab.input_stack.currentIndex() == 1
    assert "Compare" in tab.run_btn.text()


def test_summarize_tab_entity_json_parsing(qapp, tmp_path):
    settings = Settings(data_dir=tmp_path)
    tab = SummarizeTab(settings)

    sample_json = (
        '{"action_items": ["Deploy v2"], '
        '"deadlines": ["Nov 1"], '
        '"financial_figures": ["$10,000"], '
        '"key_entities": ["Polaris"]}'
    )
    tab._on_entity_done(sample_json)

    assert "Deploy v2" in tab.summary_viewer.toHtml()
    assert "Nov 1" in tab.summary_viewer.toHtml()
    assert "$10,000" in tab.summary_viewer.toHtml()
    assert "Polaris" in tab.summary_viewer.toHtml()
    assert tab.save_btn.isEnabled()

