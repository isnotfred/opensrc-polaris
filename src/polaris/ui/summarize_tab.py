"""AI-Powered Document Summarizer tab with real-time streaming."""
from __future__ import annotations

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
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings
from ..ai.summarizer import summarize_document_stream


class StreamSummarizeWorker(QThread):
    token = Signal(str)
    done = Signal()
    failed = Signal(str)

    def __init__(self, file_path: str, preset: str, settings: Settings):
        super().__init__()
        self.file_path = file_path
        self.preset = preset
        self.settings = settings

    def run(self):
        try:
            for t in summarize_document_stream(self.file_path, self.preset, self.settings):
                self.token.emit(t)
            self.done.emit()
        except Exception as e:
            self.failed.emit(str(e))


class SummarizeTab(QWidget):
    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self.settings = settings or Settings()
        self.worker: StreamSummarizeWorker | None = None
        self.accumulated_text: str = ""

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Target file and format selection
        top_group = QGroupBox("1. Select Document to Summarize")
        tg_layout = QVBoxLayout(top_group)

        file_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select a PDF, Word document, Markdown, or text file...")
        browse_btn = QPushButton("Browse File...")
        browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(self.file_edit, 1)
        file_row.addWidget(browse_btn)
        tg_layout.addLayout(file_row)

        opts_row = QHBoxLayout()
        opts_row.addWidget(QLabel("Summary Style:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Key Takeaways & Action Items (Bullets)", "key_points")
        self.preset_combo.addItem("Executive Summary (1-2 paragraphs)", "executive")
        self.preset_combo.addItem("Detailed Notes (Structured breakdown)", "detailed")
        opts_row.addWidget(self.preset_combo)

        opts_row.addStretch()

        self.summarize_btn = QPushButton("✨ Summarize with Local AI")
        self.summarize_btn.setStyleSheet("font-weight: bold; background-color: #2563eb; color: white; padding: 6px 14px;")
        self.summarize_btn.clicked.connect(self.start_summary)
        opts_row.addWidget(self.summarize_btn)

        tg_layout.addLayout(opts_row)
        layout.addWidget(top_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Select a document above and click 'Summarize with Local AI'.")
        self.status_label.setStyleSheet("color: #64748b; font-size: 12px;")
        layout.addWidget(self.status_label)

        # 2. Summary output viewer
        output_group = QGroupBox("2. Generated AI Summary (Streaming)")
        og_layout = QVBoxLayout(output_group)

        self.summary_viewer = QTextBrowser()
        self.summary_viewer.setStyleSheet(
            "background-color: #0f172a; color: #f8fafc; font-family: Segoe UI, sans-serif; font-size: 14px; padding: 12px; line-height: 1.5;"
        )
        og_layout.addWidget(self.summary_viewer, 1)

        bottom_row = QHBoxLayout()
        copy_btn = QPushButton("📋 Copy Summary to Clipboard")
        copy_btn.clicked.connect(self.copy_summary)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_summary)

        bottom_row.addWidget(copy_btn)
        bottom_row.addWidget(clear_btn)
        bottom_row.addStretch()
        og_layout.addLayout(bottom_row)

        layout.addWidget(output_group, 1)

    def _browse_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Select file to summarize",
            "",
            "Documents (*.pdf *.docx *.txt *.md *.py *.json *.csv *.html *.js *.ts);;All Files (*.*)"
        )
        if f:
            self.file_edit.setText(f)

    def start_summary(self):
        file_path = self.file_edit.text().strip()
        if not file_path or not Path(file_path).is_file():
            QMessageBox.warning(self, "Invalid File", "Please select a valid document file.")
            return

        self.summarize_btn.setEnabled(False)
        self.progress_bar.show()
        preset = self.preset_combo.currentData()
        self.status_label.setText(f"Reading {Path(file_path).name} and streaming summary from Ollama...")

        self.accumulated_text = ""
        self.summary_viewer.clear()

        self.worker = StreamSummarizeWorker(file_path, preset, self.settings)
        self.worker.token.connect(self._on_token)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_token(self, token: str):
        self.accumulated_text += token
        self.summary_viewer.insertPlainText(token)
        # Auto-scroll to bottom
        sb = self.summary_viewer.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    def _on_done(self):
        self.progress_bar.hide()
        self.summarize_btn.setEnabled(True)
        self.status_label.setText("✓ Summary complete!")
        # Render clean Markdown formatted output
        if self.accumulated_text:
            self.summary_viewer.setMarkdown(self.accumulated_text)

    def _on_failed(self, err: str):
        self.progress_bar.hide()
        self.summarize_btn.setEnabled(True)
        self.status_label.setText(f"Failed to generate summary: {err}")
        QMessageBox.critical(self, "Error", f"Failed to generate summary: {err}")

    def copy_summary(self):
        text = self.summary_viewer.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.status_label.setText("✓ Summary copied to clipboard!")

    def clear_summary(self):
        self.summary_viewer.clear()
        self.accumulated_text = ""
        self.status_label.setText("Summary cleared.")
