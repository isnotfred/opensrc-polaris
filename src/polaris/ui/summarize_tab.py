"""AI-Powered Document Summarizer, Entity Extractor, and Multi-Document Comparator.

Provides an organized, uncluttered UI with a mode switcher for:
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
    QGroupBox,
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

    def run(self):
        try:
            for t in summarize_document_stream(
                self.file_path,
                self.preset,
                self.settings,
                self.length_level,
            ):
                self.token.emit(t)
            self.done.emit()
        except Exception as e:
            self.failed.emit(str(e))


class ExtractEntitiesWorker(QThread):
    """Background worker that runs entity extraction and emits the accumulated JSON string."""

    done = Signal(str)
    failed = Signal(str)

    def __init__(self, file_path: str, settings: Settings):
        super().__init__()
        self.file_path = file_path
        self.settings = settings

    def run(self):
        try:
            tokens: list[str] = []
            for t in extract_entities_stream(self.file_path, self.settings):
                tokens.append(t)
            self.done.emit("".join(tokens))
        except Exception as e:
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

    def run(self):
        try:
            for t in comparative_summary_stream(
                self.file_paths,
                self.preset,
                self.settings,
                self.length_level,
            ):
                self.token.emit(t)
            self.done.emit()
        except Exception as e:
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
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        # ── 1. ACTION & CONFIGURATION CARD ──────────────────────────────────
        config_group = QGroupBox("Action & Target Documents")
        cg_layout = QVBoxLayout(config_group)
        cg_layout.setContentsMargins(12, 10, 12, 10)
        cg_layout.setSpacing(8)

        # Mode selector row
        mode_row = QHBoxLayout()
        mode_label = QLabel("Select Action:")
        mode_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        mode_row.addWidget(mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("📝 Single Document Summary", "summary")
        self.mode_combo.addItem("🔍 Entity & Action Items Extraction", "entity")
        self.mode_combo.addItem("🔀 Multi-Document Comparison (2–5 Files)", "multidoc")
        self.mode_combo.setStyleSheet(
            "font-size: 13px; font-weight: 600; padding: 4px 8px; min-width: 280px;"
        )
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        cg_layout.addLayout(mode_row)

        # ── Dynamic Input Stack (Single vs Multi) ───────────────────────────
        self.input_stack = QStackedWidget()

        # PAGE 0: Single Document Setup (for Summary and Entity Extraction)
        page_single = QWidget()
        ps_layout = QVBoxLayout(page_single)
        ps_layout.setContentsMargins(0, 0, 0, 0)
        ps_layout.setSpacing(8)

        single_file_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select a PDF, Word document, Markdown, or text file...")
        self.browse_btn = QPushButton("Browse File...")
        self.browse_btn.clicked.connect(self._browse_single_file)
        single_file_row.addWidget(self.file_edit, 1)
        single_file_row.addWidget(self.browse_btn)
        ps_layout.addLayout(single_file_row)

        # Single doc options (Style & Depth - visible in Summary mode, hidden in Entity mode)
        self.single_opts_widget = QWidget()
        so_layout = QHBoxLayout(self.single_opts_widget)
        so_layout.setContentsMargins(0, 0, 0, 0)
        so_layout.setSpacing(8)

        so_layout.addWidget(QLabel("Summary Style:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Key Takeaways & Action Items (Bullets)", "key_points")
        self.preset_combo.addItem("Executive Summary (1-2 paragraphs)", "executive")
        self.preset_combo.addItem("Detailed Notes (Structured breakdown)", "detailed")
        so_layout.addWidget(self.preset_combo)

        so_layout.addSpacing(12)
        so_layout.addWidget(QLabel("Depth:"))
        so_layout.addWidget(QLabel("Brief"))

        self.length_slider = QSlider(Qt.Orientation.Horizontal)
        self.length_slider.setRange(0, 2)
        self.length_slider.setValue(1)  # Moderate default
        self.length_slider.setFixedWidth(100)
        self.length_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.length_slider.setTickInterval(1)
        self.length_slider.valueChanged.connect(self._on_single_length_changed)
        so_layout.addWidget(self.length_slider)

        so_layout.addWidget(QLabel("Comprehensive"))
        self.length_label = QLabel(LENGTH_LABELS[1])
        self.length_label.setStyleSheet("color: #2563eb; font-weight: bold; min-width: 90px;")
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
        self.compare_file_list = QListWidget()
        self.compare_file_list.setFixedHeight(82)
        self.compare_file_list.setToolTip("Documents to compare (2 to 5 files).")
        multi_files_row.addWidget(self.compare_file_list, 1)

        multi_btn_col = QVBoxLayout()
        self.add_files_btn = QPushButton("➕ Add Files...")
        self.add_files_btn.clicked.connect(self._add_compare_files)
        self.remove_file_btn = QPushButton("➖ Remove Selected")
        self.remove_file_btn.clicked.connect(self._remove_compare_file)
        self.clear_files_btn = QPushButton("🗑️ Clear List")
        self.clear_files_btn.clicked.connect(self._clear_compare_files)
        multi_btn_col.addWidget(self.add_files_btn)
        multi_btn_col.addWidget(self.remove_file_btn)
        multi_btn_col.addWidget(self.clear_files_btn)
        multi_btn_col.addStretch()
        multi_files_row.addLayout(multi_btn_col)
        pm_layout.addLayout(multi_files_row)

        # Multi options (Style & Depth)
        multi_opts_row = QHBoxLayout()
        multi_opts_row.addWidget(QLabel("Comparison Style:"))
        self.multi_preset_combo = QComboBox()
        self.multi_preset_combo.addItem("Key Takeaways & Comparison (Bullets)", "key_points")
        self.multi_preset_combo.addItem("Executive Comparative Summary", "executive")
        self.multi_preset_combo.addItem("Detailed Cross-Document Breakdown", "detailed")
        multi_opts_row.addWidget(self.multi_preset_combo)

        multi_opts_row.addSpacing(12)
        multi_opts_row.addWidget(QLabel("Depth:"))
        multi_opts_row.addWidget(QLabel("Brief"))

        self.multi_length_slider = QSlider(Qt.Orientation.Horizontal)
        self.multi_length_slider.setRange(0, 2)
        self.multi_length_slider.setValue(1)
        self.multi_length_slider.setFixedWidth(100)
        self.multi_length_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.multi_length_slider.setTickInterval(1)
        self.multi_length_slider.valueChanged.connect(self._on_multi_length_changed)
        multi_opts_row.addWidget(self.multi_length_slider)

        multi_opts_row.addWidget(QLabel("Comprehensive"))
        self.multi_length_label = QLabel(LENGTH_LABELS[1])
        self.multi_length_label.setStyleSheet("color: #0f766e; font-weight: bold; min-width: 90px;")
        multi_opts_row.addWidget(self.multi_length_label)
        multi_opts_row.addStretch()
        pm_layout.addLayout(multi_opts_row)

        self.input_stack.addWidget(page_multi)
        cg_layout.addWidget(self.input_stack)

        # ── Primary Action Button & Status ──────────────────────────────────
        action_row = QHBoxLayout()
        self.run_btn = QPushButton("✨ Summarize with Local AI")
        self.run_btn.setStyleSheet(
            "font-weight: bold; font-size: 13px; background-color: #2563eb; color: white; padding: 7px 18px; border-radius: 4px;"
        )
        self.run_btn.clicked.connect(self._on_run_clicked)
        action_row.addWidget(self.run_btn)

        self.status_label = QLabel("Ready. Select a document and click 'Summarize with Local AI'.")
        self.status_label.setStyleSheet("color: #64748b; font-size: 12px; margin-left: 6px;")
        action_row.addWidget(self.status_label, 1)
        cg_layout.addLayout(action_row)

        # Thin progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(5)
        self.progress_bar.hide()
        cg_layout.addWidget(self.progress_bar)

        root_layout.addWidget(config_group)

        # ── 2. UNIFIED RESULTS & INSIGHTS CARD ──────────────────────────────
        results_group = QGroupBox("AI Results & Insights")
        rg_layout = QVBoxLayout(results_group)
        rg_layout.setContentsMargins(12, 10, 12, 10)
        rg_layout.setSpacing(8)

        # Unified viewer (supports both Markdown text and styled HTML cards)
        self.summary_viewer = QTextBrowser()
        self.summary_viewer.setStyleSheet(
            "background-color: #0f172a; color: #f8fafc; font-family: Segoe UI, sans-serif; "
            "font-size: 13px; padding: 12px; line-height: 1.5; border-radius: 4px;"
        )
        # Aliases for backward compatibility
        self.result_viewer = self.summary_viewer
        self.entity_viewer = self.summary_viewer
        rg_layout.addWidget(self.summary_viewer, 1)

        # Bottom toolbar
        bottom_row = QHBoxLayout()
        self.copy_btn = QPushButton("📋 Copy to Clipboard")
        self.copy_btn.clicked.connect(self.copy_summary)

        self.save_btn = QPushButton("💾 Save to File")
        self.save_btn.setToolTip("Save output as a Markdown (.md) or Text (.txt) file")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_to_file)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_summary)

        bottom_row.addWidget(self.copy_btn)
        bottom_row.addWidget(self.save_btn)
        bottom_row.addWidget(self.clear_btn)

        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #64748b; font-size: 12px;")
        bottom_row.addSpacing(16)
        bottom_row.addWidget(self.stats_label)
        bottom_row.addStretch()

        rg_layout.addLayout(bottom_row)
        root_layout.addWidget(results_group, 1)

        # Compatibility aliases for existing test suite or helper references
        self.summarize_btn = self.run_btn
        self.compare_btn = self.run_btn
        self.extract_btn = self.run_btn

    # ── Mode Switching ──────────────────────────────────────────────────────

    def _on_mode_changed(self, index: int):
        mode = self.mode_combo.currentData()
        self._current_mode = mode

        if mode == "summary":
            self.input_stack.setCurrentIndex(0)
            self.single_opts_widget.show()
            self.run_btn.setText("✨ Summarize with Local AI")
            self.run_btn.setStyleSheet(
                "font-weight: bold; font-size: 13px; background-color: #2563eb; color: white; padding: 7px 18px; border-radius: 4px;"
            )
            self.status_label.setText("Select a document and click 'Summarize with Local AI'.")

        elif mode == "entity":
            self.input_stack.setCurrentIndex(0)
            self.single_opts_widget.hide()  # entity mode doesn't need summary style/length
            self.run_btn.setText("🔍 Extract Entities & Figures")
            self.run_btn.setStyleSheet(
                "font-weight: bold; font-size: 13px; background-color: #7c3aed; color: white; padding: 7px 18px; border-radius: 4px;"
            )
            self.status_label.setText("Select a document and click 'Extract Entities & Figures'.")

        elif mode == "multidoc":
            self.input_stack.setCurrentIndex(1)
            self.run_btn.setText("🔀 Compare Documents with Local AI")
            self.run_btn.setStyleSheet(
                "font-weight: bold; font-size: 13px; background-color: #0f766e; color: white; padding: 7px 18px; border-radius: 4px;"
            )
            n = self.compare_file_list.count()
            self.status_label.setText(
                f"{n} document(s) loaded. Add 2–5 documents and click 'Compare Documents'."
            )

    def _on_single_length_changed(self, value: int):
        self.length_label.setText(LENGTH_LABELS.get(value, "Moderate"))

    def _on_multi_length_changed(self, value: int):
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
        if not file_path or not Path(file_path).is_file():
            QMessageBox.warning(self, "Invalid File", "Please select a valid document file.")
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
        self.status_label.setText("✓ Summary complete!")
        if self.accumulated_text:
            self.summary_viewer.setMarkdown(self.accumulated_text)
            self.save_btn.setEnabled(True)
            words = len(self.accumulated_text.split())
            self.stats_label.setText(f"Length: {words} words ({len(self.accumulated_text)} characters)")

    # ── Entity & Action Items Extraction ────────────────────────────────────

    def _start_extraction(self):
        file_path = self.file_edit.text().strip()
        if not file_path or not Path(file_path).is_file():
            QMessageBox.warning(
                self,
                "No Document Selected",
                "Please select a valid document file first.",
            )
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

        # ── Robust Multi-Tier JSON Parsing ──────────────────────────────────
        data: dict | None = None
        cleaned = json_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        # 1. Direct parse
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 2. Substring object extraction
        if data is None:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    # Clean trailing commas
                    repaired = re.sub(r",\s*([\]}])", r"\1", match.group())
                    try:
                        data = json.loads(repaired)
                    except json.JSONDecodeError:
                        pass

        # 3. Handle unclosed JSON string / bracket cutoff recovery
        if data is None and "{" in cleaned:
            # Auto-close brackets if model was truncated
            repaired = cleaned[cleaned.find("{") :]
            repaired = re.sub(r",\s*$", "", repaired.strip())
            # Close open quotes and arrays
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

        # ── Render Structured Cards HTML ────────────────────────────────────
        sections = [
            ("✅ Action Items & To-Dos",   "action_items",      "#4ade80", "#14532d"),
            ("📅 Deadlines & Schedules",    "deadlines",         "#fbbf24", "#78350f"),
            ("💰 Financial Figures",       "financial_figures", "#60a5fa", "#1e3a8a"),
            ("🏷️ Key Entities & Names",     "key_entities",      "#c084fc", "#581c87"),
        ]

        html_parts: list[str] = [
            '<div style="font-family: Segoe UI, sans-serif; padding: 4px;">'
        ]
        total_items = 0
        md_export_lines: list[str] = [f"# Extracted Entities & Action Items\n"]

        for title, key, accent_color, bg_tint in sections:
            items = data.get(key, [])
            if not isinstance(items, list):
                items = []

            item_count = len(items)
            total_items += item_count

            badge = f'<span style="background-color: {bg_tint}; color: {accent_color}; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; margin-left: 8px;">{item_count}</span>'
            html_parts.append(
                f'<div style="margin-bottom: 14px; border-left: 3px solid {accent_color}; padding-left: 10px;">'
                f'<h3 style="color: {accent_color}; margin: 0 0 6px 0; font-size: 14px;">{title} {badge}</h3>'
            )

            md_export_lines.append(f"## {title} ({item_count})\n")

            if items:
                items_html = "".join(
                    f'<li style="margin-bottom: 4px; color: #f1f5f9;">{str(item)}</li>'
                    for item in items
                )
                html_parts.append(
                    f'<ul style="margin: 0; padding-left: 20px; line-height: 1.4;">{items_html}</ul>'
                )
                for item in items:
                    md_export_lines.append(f"- {item}")
            else:
                html_parts.append(
                    '<p style="margin: 0; color: #94a3b8; font-style: italic; font-size: 12px;">No items detected in this category.</p>'
                )
                md_export_lines.append("_None detected._")

            html_parts.append("</div>")
            md_export_lines.append("")

        html_parts.append("</div>")

        self.summary_viewer.setHtml("".join(html_parts))
        self.accumulated_text = "\n".join(md_export_lines)
        self.save_btn.setEnabled(True)
        self.status_label.setText(f"✓ Extraction complete — {total_items} item(s) found.")
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
        self.status_label.setText("✓ Multi-document comparative analysis complete!")
        if self.accumulated_text:
            self.summary_viewer.setMarkdown(self.accumulated_text)
            self.save_btn.setEnabled(True)
            words = len(self.accumulated_text.split())
            self.stats_label.setText(f"Comparison: {words} words ({len(self.accumulated_text)} characters)")

    # ── General Slots & Helpers ─────────────────────────────────────────────

    def _set_running_state(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.mode_combo.setEnabled(not running)
        self.save_btn.setEnabled(False if running else bool(self.accumulated_text))
        if running:
            self.progress_bar.show()
        else:
            self.progress_bar.hide()

    def _on_op_failed(self, err: str):
        self._set_running_state(False)
        self.status_label.setText(f"Operation failed: {err}")
        QMessageBox.critical(self, "Error", f"Failed to complete operation:\n{err}")

    def copy_summary(self):
        text = self.summary_viewer.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.status_label.setText("✓ Output copied to clipboard!")

    def save_to_file(self):
        if not self.accumulated_text:
            QMessageBox.information(self, "Nothing to Save", "Generate results first.")
            return

        if self._current_mode == "summary":
            src = self.file_edit.text().strip()
            stem = Path(src).stem if src and Path(src).is_file() else "document"
            suggested = f"{stem}_summary.md"
        elif self._current_mode == "entity":
            src = self.file_edit.text().strip()
            stem = Path(src).stem if src and Path(src).is_file() else "document"
            suggested = f"{stem}_entities.md"
        else:
            suggested = "comparative_summary.md"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Output As",
            str(Path(self._last_source_dir) / suggested) if self._last_source_dir else suggested,
            "Markdown (*.md);;Plain Text (*.txt);;All Files (*.*)",
        )
        if not save_path:
            return

        try:
            Path(save_path).write_text(self.accumulated_text, encoding="utf-8")
            self.status_label.setText(f"✓ Saved to {Path(save_path).name}")
        except OSError as exc:
            QMessageBox.critical(self, "Save Failed", f"Could not write file:\n{exc}")

    def clear_summary(self):
        self.summary_viewer.clear()
        self.accumulated_text = ""
        self._last_json_text = ""
        self.save_btn.setEnabled(False)
        self.stats_label.setText("")
        self.status_label.setText("Output cleared.")
