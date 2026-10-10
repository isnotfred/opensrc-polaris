"""AI-Powered Document Summarizer, Entity Extractor, and Multi-Document Comparator.

Provides a unified card-based UI with mode switching for:
1. Document Summary (Single file)
2. Entity & Action Items Extraction (Single file)
3. Multi-Document Comparison (2–5 files)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings
from ..ai.summarizer import (
    LENGTH_LABELS,
    comparative_summary_stream,
    extract_entities_stream,
    summarize_document_stream,
)
from .common.card import CardFrame
from .common.components import SegmentedControl
from .common.styles import (
    ACCENT_PRIMARY,
    ACCENT_SUCCESS,
    BG_CARD,
    BG_INNER,
    BORDER_SUBTLE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    danger_button_style,
    ghost_button_style,
    primary_button_style,
)


class StreamSummarizeWorker(QThread):
    token = Signal(str)
    done = Signal()
    failed = Signal(str)

    def __init__(
        self,
        file_path: str,
        preset: str,
        settings: Settings,
        length_level: int = 1,
    ):
        super().__init__()
        self.file_path = file_path
        self.preset = preset
        self.settings = settings
        self.length_level = length_level
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True

    def run(self):
        try:
            for t in summarize_document_stream(
                self.file_path,
                self.preset,
                self.settings,
                self.length_level,
            ):
                if self._is_stopped:
                    return
                self.token.emit(t)
            if not self._is_stopped:
                self.done.emit()
        except Exception as e:
            if not self._is_stopped:
                self.failed.emit(str(e))


class ExtractEntitiesWorker(QThread):
    """Background worker that runs entity extraction and emits the accumulated JSON string."""

    done = Signal(str)
    failed = Signal(str)

    def __init__(self, file_path: str, settings: Settings):
        super().__init__()
        self.file_path = file_path
        self.settings = settings
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True

    def run(self):
        try:
            tokens: list[str] = []
            for t in extract_entities_stream(self.file_path, self.settings):
                if self._is_stopped:
                    return
                tokens.append(t)
            if not self._is_stopped:
                self.done.emit("".join(tokens))
        except Exception as e:
            if not self._is_stopped:
                self.failed.emit(str(e))


class ComparativeSummaryWorker(QThread):
    """Background worker for multi-document comparative summary."""

    token = Signal(str)
    done = Signal()
    failed = Signal(str)

    def __init__(
        self,
        file_paths: list[str],
        preset: str,
        settings: Settings,
        length_level: int = 1,
    ):
        super().__init__()
        self.file_paths = file_paths
        self.preset = preset
        self.settings = settings
        self.length_level = length_level
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True

    def run(self):
        try:
            for t in comparative_summary_stream(
                self.file_paths,
                self.preset,
                self.settings,
                self.length_level,
            ):
                if self._is_stopped:
                    return
                self.token.emit(t)
            if not self._is_stopped:
                self.done.emit()
        except Exception as e:
            if not self._is_stopped:
                self.failed.emit(str(e))


class SummarizeTab(QWidget):
    """Clean, unified AI Summarization, Entity Extraction, and Multi-Document tab."""

    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self.settings = settings or Settings()
        self.worker: StreamSummarizeWorker | None = None
        self.entity_worker: ExtractEntitiesWorker | None = None
        self.compare_worker: ComparativeSummaryWorker | None = None

        self.accumulated_text: str = ""
        self._last_json_text: str = ""
        self._last_source_dir: str = ""
        self._current_mode: str = "summary"

        self._init_ui()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(14, 12, 14, 12)
        root_layout.setSpacing(10)

        # ── 1. CONFIGURATION CARD ───────────────────────────────────────────
        self.config_card = CardFrame(
            "Document Configuration",
            subtitle="Select an operation mode and configure document inputs",
        )

        # Mode selector row
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        mode_label = QLabel("Action Mode:")
        mode_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 500;")
        mode_row.addWidget(mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Single Document Summary", "summary")
        self.mode_combo.addItem("Entity & Action Items Extraction", "entity")
        self.mode_combo.addItem("Multi-Document Comparison (2–5 Files)", "multidoc")
        self.mode_combo.setStyleSheet(
            "font-size: 12px; font-weight: 500; min-width: 260px;"
        )
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        self.config_card.content_layout.addLayout(mode_row)

        # ── Dynamic Input Stack (Single vs Multi) ───────────────────────────
        self.input_stack = QStackedWidget()

        # PAGE 0: Single Document Setup
        page_single = QWidget()
        ps_layout = QVBoxLayout(page_single)
        ps_layout.setContentsMargins(0, 0, 0, 0)
        ps_layout.setSpacing(8)

        single_file_row = QHBoxLayout()
        single_file_row.setSpacing(8)
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select a PDF, Word document, Markdown, or text file...")
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_single_file)
        single_file_row.addWidget(self.file_edit, 1)
        single_file_row.addWidget(self.browse_btn)
        ps_layout.addLayout(single_file_row)

        # Single doc options (Style & Depth)
        self.single_opts_widget = QWidget()
        so_layout = QHBoxLayout(self.single_opts_widget)
        so_layout.setContentsMargins(0, 0, 0, 0)
        so_layout.setSpacing(8)

        style_lbl = QLabel("Style:")
        style_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 500;")
        so_layout.addWidget(style_lbl)

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Key Takeaways & Action Items (Bullets)", "key_points")
        self.preset_combo.addItem("Executive Summary (1-2 paragraphs)", "executive")
        self.preset_combo.addItem("Detailed Notes (Structured breakdown)", "detailed")
        so_layout.addWidget(self.preset_combo)

        so_layout.addSpacing(6)
        depth_lbl = QLabel("Depth:")
        depth_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 500;")
        so_layout.addWidget(depth_lbl)

        # Compact Segmented Control
        self.segmented_depth = SegmentedControl(["Brief", "Moderate", "Comprehensive"], default=1)
        self.segmented_depth.selection_changed.connect(self._on_segmented_depth_changed)
        so_layout.addWidget(self.segmented_depth)

        # Hidden slider to preserve complete API compatibility with existing tests
        self.length_slider = QSlider(Qt.Orientation.Horizontal)
        self.length_slider.setRange(0, 2)
        self.length_slider.setValue(1)
        self.length_slider.hide()
        self.length_slider.valueChanged.connect(self._on_single_length_changed)
        so_layout.addWidget(self.length_slider)

        self.length_label = QLabel(LENGTH_LABELS[1])
        self.length_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.length_label.hide()
        so_layout.addWidget(self.length_label)

        so_layout.addStretch()
        ps_layout.addWidget(self.single_opts_widget)

        self.input_stack.addWidget(page_single)

        # PAGE 1: Multi-Document Setup
        page_multi = QWidget()
        pm_layout = QVBoxLayout(page_multi)
        pm_layout.setContentsMargins(0, 0, 0, 0)
        pm_layout.setSpacing(8)

        multi_files_row = QHBoxLayout()
        multi_files_row.setSpacing(8)
        self.compare_file_list = QListWidget()
        self.compare_file_list.setFixedHeight(80)
        self.compare_file_list.setToolTip("Documents to compare (2 to 5 files).")
        multi_files_row.addWidget(self.compare_file_list, 1)

        multi_btn_col = QVBoxLayout()
        multi_btn_col.setSpacing(4)
        self.add_files_btn = QPushButton("Add Files...")
        self.add_files_btn.setStyleSheet(ghost_button_style())
        self.add_files_btn.clicked.connect(self._add_compare_files)
        self.remove_file_btn = QPushButton("Remove Selected")
        self.remove_file_btn.setStyleSheet(ghost_button_style())
        self.remove_file_btn.clicked.connect(self._remove_compare_file)
        self.clear_files_btn = QPushButton("Clear List")
        self.clear_files_btn.setStyleSheet(ghost_button_style())
        self.clear_files_btn.clicked.connect(self._clear_compare_files)
        multi_btn_col.addWidget(self.add_files_btn)
        multi_btn_col.addWidget(self.remove_file_btn)
        multi_btn_col.addWidget(self.clear_files_btn)
        multi_btn_col.addStretch()
        multi_files_row.addLayout(multi_btn_col)
        pm_layout.addLayout(multi_files_row)

        # Multi options (Style & Depth)
        multi_opts_row = QHBoxLayout()
        multi_opts_row.setSpacing(8)
        cmp_style_lbl = QLabel("Comparison Style:")
        cmp_style_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 500;")
        multi_opts_row.addWidget(cmp_style_lbl)

        self.multi_preset_combo = QComboBox()
        self.multi_preset_combo.addItem("Key Takeaways & Comparison (Bullets)", "key_points")
        self.multi_preset_combo.addItem("Executive Comparative Summary", "executive")
        self.multi_preset_combo.addItem("Detailed Cross-Document Breakdown", "detailed")
        multi_opts_row.addWidget(self.multi_preset_combo)

        multi_opts_row.addSpacing(6)
        cmp_depth_lbl = QLabel("Depth:")
        cmp_depth_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 500;")
        multi_opts_row.addWidget(cmp_depth_lbl)

        self.multi_segmented_depth = SegmentedControl(["Brief", "Moderate", "Comprehensive"], default=1)
        self.multi_segmented_depth.selection_changed.connect(self._on_multi_segmented_depth_changed)
        multi_opts_row.addWidget(self.multi_segmented_depth)

        self.multi_length_slider = QSlider(Qt.Orientation.Horizontal)
        self.multi_length_slider.setRange(0, 2)
        self.multi_length_slider.setValue(1)
        self.multi_length_slider.hide()
        self.multi_length_slider.valueChanged.connect(self._on_multi_length_changed)
        multi_opts_row.addWidget(self.multi_length_slider)

        self.multi_length_label = QLabel(LENGTH_LABELS[1])
        self.multi_length_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.multi_length_label.hide()
        multi_opts_row.addWidget(self.multi_length_label)

        multi_opts_row.addStretch()
        pm_layout.addLayout(multi_opts_row)

        self.input_stack.addWidget(page_multi)
        self.config_card.content_layout.addWidget(self.input_stack)

        # ── Primary Action Button & Status ──────────────────────────────────
        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.run_btn = QPushButton("Summarize with Local AI")
        self.run_btn.setStyleSheet(primary_button_style())
        self.run_btn.clicked.connect(self._on_run_clicked)
        action_row.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(danger_button_style())
        self.stop_btn.hide()
        self.stop_btn.clicked.connect(self._stop_current_operation)
        action_row.addWidget(self.stop_btn)

        self.status_label = QLabel("Ready. Select a document and click 'Summarize with Local AI'.")
        self.status_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        action_row.addWidget(self.status_label, 1)
        self.config_card.content_layout.addLayout(action_row)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.hide()
        self.config_card.content_layout.addWidget(self.progress_bar)

        root_layout.addWidget(self.config_card)

        # ── 2. UNIFIED RESULTS & INSIGHTS CARD ──────────────────────────────
        results_header_actions = QWidget()
        rha_layout = QHBoxLayout(results_header_actions)
        rha_layout.setContentsMargins(0, 0, 0, 0)
        rha_layout.setSpacing(6)

        self.copy_btn = QPushButton("Copy Output")
        self.copy_btn.setStyleSheet(ghost_button_style())
        self.copy_btn.clicked.connect(self.copy_summary)
        rha_layout.addWidget(self.copy_btn)

        self.save_btn = QPushButton("Save As...")
        self.save_btn.setStyleSheet(ghost_button_style())
        self.save_btn.setToolTip("Save output as a Markdown (.md) or Text (.txt) file")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_to_file)
        rha_layout.addWidget(self.save_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet(ghost_button_style())
        self.clear_btn.clicked.connect(self.clear_summary)
        rha_layout.addWidget(self.clear_btn)

        self.results_card = CardFrame(
            "Summary Output & Insights",
            subtitle="Synthesized takeaways streamed directly from local Ollama",
            header_action=results_header_actions,
        )

        # Unified viewer
        self.summary_viewer = QTextBrowser()
        self.summary_viewer.setStyleSheet(
            f"background-color: {BG_INNER}; color: {TEXT_PRIMARY}; font-family: 'Segoe UI', system-ui, sans-serif; "
            f"font-size: 13px; padding: 14px; line-height: 1.5; border: 1px solid {BORDER_SUBTLE}; border-radius: 6px;"
        )
        # Aliases for backward compatibility
        self.result_viewer = self.summary_viewer
        self.entity_viewer = self.summary_viewer
        self.results_card.content_layout.addWidget(self.summary_viewer, 1)

        # Bottom stats row
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        bottom_row.addWidget(self.stats_label)
        bottom_row.addStretch()
        self.results_card.content_layout.addLayout(bottom_row)

        root_layout.addWidget(self.results_card, 1)

        # Compatibility aliases
        self.summarize_btn = self.run_btn
        self.compare_btn = self.run_btn
        self.extract_btn = self.run_btn

        self._show_empty_state()

    def _show_empty_state(self):
        html = f"""
        <div align="center" style="margin: 36px 0;">
            <table cellpadding="18" cellspacing="0" style="background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; max-width: 580px;">
                <tr>
                    <td align="center">
                        <div style="font-size: 14px; font-weight: 600; color: #58a6ff; margin-bottom: 6px;">Document Summary & Analysis</div>
                        <div style="font-size: 12px; color: #8b949e; line-height: 1.5;">
                            Select a document above and click <b>Summarize with Local AI</b> to generate executive briefs, bulleted takeaways, or structured action items.
                        </div>
                    </td>
                </tr>
            </table>
        </div>
        """
        self.summary_viewer.setHtml(html)

    # ── Mode Switching ──────────────────────────────────────────────────────

    def _on_mode_changed(self, index: int):
        mode = self.mode_combo.currentData()
        self._current_mode = mode

        if mode == "summary":
            self.input_stack.setCurrentIndex(0)
            self.single_opts_widget.show()
            self.run_btn.setText("Summarize with Local AI")
            self.run_btn.setStyleSheet(primary_button_style())
            self.status_label.setText("Select a document and click 'Summarize with Local AI'.")

        elif mode == "entity":
            self.input_stack.setCurrentIndex(0)
            self.single_opts_widget.hide()
            self.run_btn.setText("Extract Entities & Figures")
            self.run_btn.setStyleSheet(
                "font-weight: 600; font-size: 12px; background-color: #6366f1; color: white; "
                "padding: 7px 16px; border-radius: 6px; border: 1px solid #4f46e5;"
            )
            self.status_label.setText("Select a document and click 'Extract Entities & Figures'.")

        elif mode == "multidoc":
            self.input_stack.setCurrentIndex(1)
            self.run_btn.setText("Compare Documents with Local AI")
            self.run_btn.setStyleSheet(
                "font-weight: 600; font-size: 12px; background-color: #0f766e; color: white; "
                "padding: 7px 16px; border-radius: 6px; border: 1px solid #115e59;"
            )
            n = self.compare_file_list.count()
            self.status_label.setText(
                f"{n} document(s) loaded. Add 2–5 documents and click 'Compare Documents'."
            )

    def _on_segmented_depth_changed(self, idx: int):
        self.length_slider.setValue(idx)
        self.length_label.setText(LENGTH_LABELS.get(idx, "Moderate"))

    def _on_multi_segmented_depth_changed(self, idx: int):
        self.multi_length_slider.setValue(idx)
        self.multi_length_label.setText(LENGTH_LABELS.get(idx, "Moderate"))

    def _on_single_length_changed(self, value: int):
        self.segmented_depth.set_selected(value)
        self.length_label.setText(LENGTH_LABELS.get(value, "Moderate"))

    def _on_multi_length_changed(self, value: int):
        self.multi_segmented_depth.set_selected(value)
        self.multi_length_label.setText(LENGTH_LABELS.get(value, "Moderate"))

    # ── File Selection Handlers ─────────────────────────────────────────────

    def _browse_single_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Select document",
            self._last_source_dir,
            "Documents (*.pdf *.docx *.txt *.md *.py *.json *.csv *.html *.js *.ts);;All Files (*.*)",
        )
        if f:
            self.file_edit.setText(f)
            self._last_source_dir = str(Path(f).parent)

    def _add_compare_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Documents to Compare",
            self._last_source_dir,
            "Documents (*.pdf *.docx *.txt *.md *.py *.json *.csv *.html *.js *.ts);;All Files (*.*)",
        )
        if not files:
            return

        existing: set[str] = set()
        for i in range(self.compare_file_list.count()):
            existing.add(self.compare_file_list.item(i).text())

        added = 0
        for f in files:
            if self.compare_file_list.count() >= 5:
                QMessageBox.information(
                    self, "Limit Reached", "Maximum 5 documents allowed for comparison."
                )
                break
            if f not in existing:
                self.compare_file_list.addItem(f)
                existing.add(f)
                added += 1

        if added:
            self._last_source_dir = str(Path(files[0]).parent)
            n = self.compare_file_list.count()
            self.status_label.setText(
                f"{n} document(s) loaded. "
                + ("Ready — click Compare Documents." if n >= 2 else "Add at least 2 documents.")
            )

    def _remove_compare_file(self):
        selected = self.compare_file_list.selectedItems()
        if not selected:
            return
        for item in selected:
            self.compare_file_list.takeItem(self.compare_file_list.row(item))
        n = self.compare_file_list.count()
        self.status_label.setText(
            f"{n} document(s) loaded." if n else "Add 2–5 documents, then click Compare Documents."
        )

    def _clear_compare_files(self):
        self.compare_file_list.clear()
        self.status_label.setText("Comparison document list cleared.")

    # ── Dispatch Execution ──────────────────────────────────────────────────

    def _on_run_clicked(self):
        if self._current_mode == "summary":
            self.start_summary()
        elif self._current_mode == "entity":
            self._start_extraction()
        elif self._current_mode == "multidoc":
            self._start_comparison()

    # ── Single Document Summarization ───────────────────────────────────────

    def start_summary(self):
        file_path = self.file_edit.text().strip()
        if not self._validate_file_path(file_path):
            return

        length_level = self.length_slider.value()
        preset = self.preset_combo.currentData()
        level_name = LENGTH_LABELS.get(length_level, "Moderate")

        self._set_running_state(True)
        self.status_label.setText(
            f"Analyzing {Path(file_path).name} — {level_name} {preset.replace('_', ' ')} — streaming from Ollama..."
        )

        self._last_source_dir = str(Path(file_path).parent)
        self.accumulated_text = ""
        self._last_json_text = ""
        self.summary_viewer.clear()
        self.stats_label.setText("")

        self.worker = StreamSummarizeWorker(file_path, preset, self.settings, length_level)
        self.worker.token.connect(self._on_token)
        self.worker.done.connect(self._on_summary_done)
        self.worker.failed.connect(self._on_op_failed)
        self.worker.start()

    def _on_token(self, token: str):
        self.accumulated_text += token
        self.summary_viewer.insertPlainText(token)
        sb = self.summary_viewer.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    def _on_summary_done(self):
        self._set_running_state(False)
        self.status_label.setText("Summary complete!")
        if self.accumulated_text:
            self.summary_viewer.setMarkdown(self.accumulated_text)
            self.save_btn.setEnabled(True)
            words = len(self.accumulated_text.split())
            self.stats_label.setText(f"Length: {words} words ({len(self.accumulated_text)} characters)")

    # ── Entity & Action Items Extraction ────────────────────────────────────

    def _start_extraction(self):
        file_path = self.file_edit.text().strip()
        if not self._validate_file_path(file_path):
            return

        self._set_running_state(True)
        self.summary_viewer.clear()
        self.stats_label.setText("")
        self.accumulated_text = ""
        self._last_json_text = ""
        self.status_label.setText(
            f"Analyzing {Path(file_path).name} — extracting entities, figures & tasks..."
        )

        self._last_source_dir = str(Path(file_path).parent)
        self.entity_worker = ExtractEntitiesWorker(file_path, self.settings)
        self.entity_worker.done.connect(self._on_entity_done)
        self.entity_worker.failed.connect(self._on_op_failed)
        self.entity_worker.start()

    def _on_entity_done(self, json_text: str):
        self._set_running_state(False)
        self._last_json_text = json_text

        # Robust multi-tier JSON parsing
        data: dict | None = None
        cleaned = json_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        if data is None:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    repaired = re.sub(r",\s*([\]}])", r"\1", match.group())
                    try:
                        data = json.loads(repaired)
                    except json.JSONDecodeError:
                        pass

        if data is None and "{" in cleaned:
            repaired = cleaned[cleaned.find("{") :]
            repaired = re.sub(r",\s*$", "", repaired.strip())
            if repaired.count('"') % 2 != 0:
                repaired += '"'
            open_brackets = repaired.count("[") - repaired.count("]")
            if open_brackets > 0:
                repaired += "]" * open_brackets
            open_braces = repaired.count("{") - repaired.count("}")
            if open_braces > 0:
                repaired += "}" * open_braces
            try:
                data = json.loads(repaired)
            except Exception:
                pass

        if data is None or not isinstance(data, dict):
            self.status_label.setText("Extraction finished (Raw output displayed below).")
            self.summary_viewer.setPlainText(json_text)
            self.accumulated_text = json_text
            self.save_btn.setEnabled(True)
            return

        # Render Structured Cards HTML
        sections = [
            ("Action Items & To-Dos",   "action_items",      "#3fb950", "#143a22"),
            ("Deadlines & Schedules",    "deadlines",         "#d29922", "#3b2a0c"),
            ("Financial Figures",       "financial_figures", "#58a6ff", "#132b4a"),
            ("Key Entities & Names",     "key_entities",      "#bc8cff", "#2b1b47"),
        ]

        html_parts: list[str] = [
            '<div style="font-family: \'Segoe UI\', sans-serif; padding: 4px;">'
        ]
        total_items = 0
        md_export_lines: list[str] = [f"# Extracted Entities & Action Items\n"]

        for title, key, accent_color, bg_tint in sections:
            items = data.get(key, [])
            if not isinstance(items, list):
                items = []

            item_count = len(items)
            total_items += item_count

            badge = f'<span style="background-color: {bg_tint}; color: {accent_color}; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; margin-left: 8px;">{item_count}</span>'
            html_parts.append(
                f'<div style="margin-bottom: 14px; border-left: 3px solid {accent_color}; padding-left: 10px;">'
                f'<h3 style="color: {accent_color}; margin: 0 0 6px 0; font-size: 13px; font-weight: 600;">{title} {badge}</h3>'
            )

            md_export_lines.append(f"## {title} ({item_count})\n")

            if items:
                items_html = "".join(
                    f'<li style="margin-bottom: 4px; color: #f0f6fc;">{str(item)}</li>'
                    for item in items
                )
                html_parts.append(
                    f'<ul style="margin: 0; padding-left: 20px; line-height: 1.4;">{items_html}</ul>'
                )
                for item in items:
                    md_export_lines.append(f"- {item}")
            else:
                html_parts.append(
                    '<p style="margin: 0; color: #8b949e; font-style: italic; font-size: 12px;">No items detected in this category.</p>'
                )
                md_export_lines.append("_None detected._")

            html_parts.append("</div>")
            md_export_lines.append("")

        html_parts.append("</div>")

        self.summary_viewer.setHtml("".join(html_parts))
        self.accumulated_text = "\n".join(md_export_lines)
        self.save_btn.setEnabled(True)
        self.status_label.setText(f"Extraction complete — {total_items} item(s) found.")
        self.stats_label.setText(f"Found {total_items} structured items across 4 categories")

    # ── Multi-Document Comparative Summary ─────────────────────────────────

    def _start_comparison(self):
        paths: list[str] = [
            self.compare_file_list.item(i).text()
            for i in range(self.compare_file_list.count())
        ]

        if len(paths) < 2:
            QMessageBox.warning(
                self,
                "Not Enough Files",
                "Please add at least 2 documents to compare (up to 5).",
            )
            return

        missing = [p for p in paths if not Path(p).is_file()]
        if missing:
            names = "\n".join(Path(m).name for m in missing)
            QMessageBox.warning(
                self,
                "File(s) Not Found",
                f"The following file(s) could not be found:\n{names}\nPlease re-add them.",
            )
            return

        length_level = self.multi_length_slider.value()
        preset = self.multi_preset_combo.currentData()
        level_name = LENGTH_LABELS.get(length_level, "Moderate")
        n = len(paths)

        self._set_running_state(True)
        self.status_label.setText(
            f"Comparing {n} documents — {level_name} {preset.replace('_', ' ')} — streaming from Ollama..."
        )

        self._last_source_dir = str(Path(paths[0]).parent)
        self.accumulated_text = ""
        self._last_json_text = ""
        self.summary_viewer.clear()
        self.stats_label.setText("")

        self.compare_worker = ComparativeSummaryWorker(
            paths, preset, self.settings, length_level
        )
        self.compare_worker.token.connect(self._on_token)
        self.compare_worker.done.connect(self._on_comparison_done)
        self.compare_worker.failed.connect(self._on_op_failed)
        self.compare_worker.start()

    def _on_comparison_done(self):
        self._set_running_state(False)
        self.status_label.setText("Multi-document comparative analysis complete!")
        if self.accumulated_text:
            self.summary_viewer.setMarkdown(self.accumulated_text)
            self.save_btn.setEnabled(True)
            words = len(self.accumulated_text.split())
            self.stats_label.setText(f"Comparison: {words} words ({len(self.accumulated_text)} characters)")

    def _validate_file_path(self, path_str: str) -> bool:
        if not path_str or not Path(path_str).is_file():
            QMessageBox.warning(self, "Invalid File", "Please select a valid document file.")
            return False
        p = Path(path_str)
        try:
            sz = p.stat().st_size
            if sz == 0:
                QMessageBox.warning(self, "Empty File", f"'{p.name}' is completely empty (0 bytes).")
                return False
            if sz > 35 * 1024 * 1024:
                resp = QMessageBox.question(
                    self,
                    "Large File Warning",
                    f"'{p.name}' is {sz / (1024*1024):.1f} MB. Processing this file may take longer on CPU. Do you want to continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if resp != QMessageBox.StandardButton.Yes:
                    return False
        except OSError:
            pass
        return True

    def _set_running_state(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.mode_combo.setEnabled(not running)
        self.save_btn.setEnabled(False if running else bool(self.accumulated_text))
        if running:
            self.stop_btn.show()
            self.progress_bar.show()
        else:
            self.stop_btn.hide()
            self.progress_bar.hide()

    def _stop_current_operation(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        if self.entity_worker and self.entity_worker.isRunning():
            self.entity_worker.stop()
        if self.compare_worker and self.compare_worker.isRunning():
            self.compare_worker.stop()
        self._set_running_state(False)
        self.status_label.setText("Operation cancelled by user.")

    def _on_op_failed(self, err: str):
        self._set_running_state(False)
        self.status_label.setText(f"Operation failed: {err}")
        QMessageBox.critical(self, "Error", f"Failed to complete operation:\n{err}")

    def copy_summary(self):
        text = self.summary_viewer.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.status_label.setText("Output copied to clipboard!")

    def save_to_file(self):
        if not self.accumulated_text:
            QMessageBox.information(self, "Nothing to Save", "Generate results first.")
            return

        if self._current_mode == "summary":
            src = self.file_edit.text().strip()
            stem = Path(src).stem if src and Path(src).is_file() else "document"
            suggested = f"{stem}_summary.md"
            file_filter = "Markdown (*.md);;Plain Text (*.txt);;All Files (*.*)"
        elif self._current_mode == "entity":
            src = self.file_edit.text().strip()
            stem = Path(src).stem if src and Path(src).is_file() else "document"
            suggested = f"{stem}_entities.md"
            file_filter = "Markdown (*.md);;JSON Data (*.json);;Plain Text (*.txt);;All Files (*.*)"
        else:
            suggested = "comparative_summary.md"
            file_filter = "Markdown (*.md);;Plain Text (*.txt);;All Files (*.*)"

        save_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Output As",
            str(Path(self._last_source_dir) / suggested) if self._last_source_dir else suggested,
            file_filter,
        )
        if not save_path:
            return

        try:
            out_p = Path(save_path)
            if out_p.suffix.lower() == ".json" and self._last_json_text:
                out_p.write_text(self._last_json_text, encoding="utf-8")
            else:
                out_p.write_text(self.accumulated_text, encoding="utf-8")
            self.status_label.setText(f"Saved to {out_p.name}")
        except OSError as exc:
            QMessageBox.critical(self, "Save Failed", f"Could not write file:\n{exc}")

    def clear_summary(self):
        self.summary_viewer.clear()
        self.accumulated_text = ""
        self._last_json_text = ""
        self.save_btn.setEnabled(False)
        self.stats_label.setText("")
        self._show_empty_state()
        self.status_label.setText("Output cleared.")
