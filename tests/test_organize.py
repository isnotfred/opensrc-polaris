"""Dedicated unit and integration tests for Feature 1 (AI Organize)."""
import os
import pytest
from pathlib import Path

# Ensure offscreen Qt platform for testing environments
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from polaris.config import Settings
from polaris.core.organizer import Move, apply_moves, plan_by_type, undo_batch, unique_destination
from polaris.db.database import connect
from polaris.ui.ai_organize_tab import AIOrganizeTab
from polaris.ui.suggestion_toast import SuggestionToast
from polaris.ai.ai_organizer import suggest_single_file_placement


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_unique_destination_increments(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("orig")
    c1 = unique_destination(f)
    assert c1.name == "doc (1).txt"
    c1.write_text("copy1")
    c2 = unique_destination(f)
    assert c2.name == "doc (2).txt"


def test_copy_mode_and_undo(tmp_path):
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()
    dst_dir.mkdir()

    f1 = src_dir / "report.pdf"
    f1.write_text("original pdf")

    conn = connect(tmp_path / "test.db")
    moves = [Move(src=str(f1), dst=str(dst_dir / "report.pdf"), reason="copy test")]

    # Apply in copy mode
    batch_id = apply_moves(conn, moves, mode="copy")
    assert f1.exists()  # Source remains intact
    assert (dst_dir / "report.pdf").exists()  # Copy created

    # Undo copy mode: should delete the copied file and keep original
    restored, failed = undo_batch(conn, batch_id)
    assert restored == 1
    assert failed == 0
    assert not (dst_dir / "report.pdf").exists()
    assert f1.exists()


def test_organize_tab_manual_destination_edit_and_filter(qapp, tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "invoice.pdf").write_text("invoice")
    (in_dir / "photo.png").write_text("photo")

    settings = Settings(data_dir=tmp_path / "data")
    tab = AIOrganizeTab(settings)
    tab.src_edit.setText(str(in_dir))

    moves = plan_by_type([str(in_dir / "invoice.pdf"), str(in_dir / "photo.png")], str(in_dir))
    tab._show_preview(moves)

    assert tab.table.rowCount() == 2

    # Test filtering
    tab._on_filter_changed("invoice")
    assert not tab.table.isRowHidden(0)
    assert tab.table.isRowHidden(1)

    tab._on_filter_changed("")
    assert not tab.table.isRowHidden(0)
    assert not tab.table.isRowHidden(1)

    # Test manual destination editing
    custom_dst = str(in_dir / "CustomFolder" / "custom_invoice.pdf")
    tab.table.item(0, 2).setText(custom_dst)

    # Check that item flag is editable
    flags = tab.table.item(0, 2).flags()
    assert flags & Qt.ItemFlag.ItemIsEditable


def test_suggest_single_file_fallback(tmp_path):
    f = tmp_path / "receipt_uber.txt"
    f.write_text("UBER RECEIPT 2024 Total: $25.50")
    res = suggest_single_file_placement(f)
    assert "suggested_filename" in res
    assert "suggested_folder" in res
    assert res["original_path"] == str(f)


def test_suggestion_toast_apply(qapp, tmp_path):
    f = tmp_path / "test_download.pdf"
    f.write_text("download dummy content")

    settings = Settings(data_dir=tmp_path / "data")
    suggestion = {
        "original_path": str(f),
        "suggested_filename": "Renamed_Doc.pdf",
        "suggested_folder": "Organized_Docs",
        "reason": "Test placement"
    }

    toast = SuggestionToast(suggestion, settings)
    assert toast.name_edit.text() == "Renamed_Doc.pdf"
    target_folder = tmp_path / "Organized_Docs"
    toast.folder_edit.setText(str(target_folder))

    # Test apply
    toast.apply_suggestion()
    assert not f.exists()  # Successfully moved
    assert (target_folder / "Renamed_Doc.pdf").exists()
