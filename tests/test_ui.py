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


def test_ai_organize_tab_strategy_banner(qapp, tmp_path):
    settings = Settings(data_dir=tmp_path / "data")
    tab = AIOrganizeTab(settings)
    in_dir = tmp_path / "incoming"
    in_dir.mkdir()
    tab.src_edit.setText(str(in_dir))

    from polaris.core.organizer import Move
    moves = [
        Move(str(in_dir / "a.docx"), str(in_dir / "Documents" / "Small (under 1MB)" / "a.docx"), "Small doc"),
        Move(str(in_dir / "b.mp4"), str(in_dir / "Videos" / "Large (over 50MB)" / "b.mp4"), "Large video"),
    ]
    tab._show_preview(
        moves,
        strategy_name="✨ Custom Instructions: \"sort by file type, then size\"",
        explanation="Files were categorized by format then subdivided by size brackets.",
    )

    assert tab.table.rowCount() == 2
    assert "sort by file type, then size" in tab.strategy_title_label.text()
    assert "categorized by format" in tab.strategy_desc_label.text()
    assert "Documents/Small (under 1MB)" in tab.strategy_folders_label.text()
    assert "Videos/Large (over 50MB)" in tab.strategy_folders_label.text()


def test_ai_organize_tab_buttons_and_presets(qapp, tmp_path):
    settings = Settings(data_dir=tmp_path / "data")
    tab = AIOrganizeTab(settings)

    # Check distinct planning buttons exist
    assert tab.ai_custom_btn is not None
    assert tab.ai_smart_btn is not None
    assert tab.fast_plan_btn is not None
    assert tab.ai_plan_btn is tab.ai_custom_btn  # alias works

    # Initial state
    assert tab.instruction_edit.text() == ""
    # Test setting type then size preset
    tab.instruction_edit.setText("Sort by file type, then size (Small under 1MB, Medium 1MB-50MB, Large over 50MB)")
    assert "file type, then size" in tab.instruction_edit.text()


def test_chat_tab_init(qapp, tmp_path):
    settings = Settings(data_dir=tmp_path)
    tab = ChatTab(settings)
    assert tab.folder_edit is not None
    assert tab.chat_browser is not None
    assert tab.send_btn is not None
    assert tab.stop_btn is not None
    assert tab.export_btn is not None
    assert tab.clear_btn is not None
    assert "Polaris Document Chat" in tab.chat_browser.toHtml()
    assert "like Messenger" not in tab.chat_browser.toHtml()

    # When folder is selected, index button glows green while text remains intact
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    tab.folder_edit.setText(str(docs_dir))
    assert "#16a34a" in tab.index_btn.styleSheet()
    assert "#4ade80" in tab.index_btn.styleSheet()
    assert tab.index_btn.text() == "⚡ Index Folder for AI Search"


def test_chat_tab_messenger_multi_turn_flow(qapp, tmp_path):
    settings = Settings(data_dir=tmp_path)
    tab = ChatTab(settings)

    # Turn 1
    tab._append_user_bubble("What are the quarterly findings?", "7:00 PM")
    assert "What are the quarterly findings?" in tab.chat_browser.toPlainText()

    tab._insert_assistant_bubble_placeholder("7:00 PM")
    assert "Thinking..." in tab.chat_browser.toPlainText()

    tab.current_assistant_text = "The company grew by 25%."
    tab.current_question = "What are the quarterly findings?"
    tab._on_stream_done(cited=[])

    assert "The company grew by 25%." in tab.chat_browser.toPlainText()
    assert len(tab.history) == 2
    assert tab.history[0] == {"role": "user", "content": "What are the quarterly findings?"}
    assert tab.history[1] == {"role": "assistant", "content": "The company grew by 25%."}

    # Turn 2 - Added at the bottom, Turn 1 remains preserved above
    tab._append_user_bubble("Can you summarize that in 3 words?", "7:01 PM")
    assert "What are the quarterly findings?" in tab.chat_browser.toPlainText()  # Turn 1 user preserved
    assert "The company grew by 25%." in tab.chat_browser.toPlainText()          # Turn 1 answer preserved
    assert "Can you summarize that in 3 words?" in tab.chat_browser.toPlainText() # Turn 2 user at bottom

    tab._insert_assistant_bubble_placeholder("7:01 PM")
    tab.current_assistant_text = "Strong revenue growth."
    tab.current_question = "Can you summarize that in 3 words?"
    tab._on_stream_done(cited=[])

    # Both turns present in chronological order
    full_text = tab.chat_browser.toPlainText()
    assert full_text.index("quarterly findings") < full_text.index("grew by 25%")
    assert full_text.index("grew by 25%") < full_text.index("summarize that")
    assert full_text.index("summarize that") < full_text.index("Strong revenue growth")
    assert len(tab.history) == 4


def test_chat_bubble_formatting():
    from polaris.ui.chat_tab import _format_bubble_text
    formatted = _format_bubble_text("Key **features**:\n- Fast search\n- Grounded citations")
    assert "<b>features</b>" in formatted
    assert "<li>Fast search</li>" in formatted
    assert "<li>Grounded citations</li>" in formatted



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


def test_summarize_tab_stop_button_and_validation(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **kw: None)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.Yes)

    settings = Settings(data_dir=tmp_path)
    tab = SummarizeTab(settings)

    # Stop button initially hidden
    assert tab.stop_btn.isHidden()

    # Empty file check
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")
    assert tab._validate_file_path(str(empty_file)) is False

    # Valid file check
    valid_file = tmp_path / "valid.txt"
    valid_file.write_text("Valid content", encoding="utf-8")
    assert tab._validate_file_path(str(valid_file)) is True

    # Stop button state toggle
    tab._set_running_state(True)
    assert not tab.stop_btn.isHidden()
    assert tab.run_btn.isEnabled() is False

    tab._stop_current_operation()
    assert tab.stop_btn.isHidden()
    assert tab.run_btn.isEnabled() is True
    assert "cancelled" in tab.status_label.text().lower() or "stopped" in tab.status_label.text().lower()



def test_ai_organizer_folder_sanitization():
    from polaris.ai.ai_organizer import _sanitize_folder_component
    assert _sanitize_folder_component("ValidFolder") == "ValidFolder"
    assert _sanitize_folder_component("../../etc/passwd") == "etc/passwd"
    assert _sanitize_folder_component("Reports:2024*Final?") == "Reports_2024_Final_"
    assert _sanitize_folder_component("") == "Organized_Files"


