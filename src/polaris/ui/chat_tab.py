"""AI-Powered Search & Document Chatbot tab with real-time token streaming, citation badges, hybrid retrieval, HyDE, and stop generation."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings
from ..core.extractor import DocumentChunk
from ..ai.rag import RagEngine, format_chat_export


class SourceViewerDialog(QDialog):
    """Interactive modal dialog displaying the exact retrieved snippets grounding the answer."""
    def __init__(
        self,
        chunks: list[DocumentChunk],
        initial_index: int = 0,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.chunks = chunks
        self.initial_index = initial_index
        self.setWindowTitle("Polaris - Source Inspection & Citations")
        self.resize(780, 520)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel(f"<b>Retrieved Context Sources ({len(self.chunks)} chunk(s))</b>")
        header.setStyleSheet("font-size: 13px; color: #1e293b; margin-bottom: 4px;")
        layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left list of chunks
        self.chunk_list = QListWidget()
        self.chunk_list.setStyleSheet(
            "QListWidget { border: 1px solid #cbd5e1; border-radius: 6px; padding: 4px; }"
            "QListWidget::item { padding: 6px; border-bottom: 1px solid #f1f5f9; }"
            "QListWidget::item:selected { background-color: #2563eb; color: white; border-radius: 4px; }"
        )
        for i, c in enumerate(self.chunks):
            sec_tag = f" § {c.section}" if getattr(c, "section", "") else ""
            item = QListWidgetItem(f"📄 [{i + 1}] {c.file_name}{sec_tag} (Page {c.page})")
            self.chunk_list.addItem(item)
        self.chunk_list.currentRowChanged.connect(self._on_chunk_selected)
        splitter.addWidget(self.chunk_list)

        # Right pane: details
        right_panel = QWidget()
        rp_layout = QVBoxLayout(right_panel)
        rp_layout.setContentsMargins(6, 0, 0, 0)

        self.meta_label = QLabel()
        self.meta_label.setStyleSheet("color: #475569; font-size: 12px; margin-bottom: 6px;")
        self.meta_label.setWordWrap(True)
        rp_layout.addWidget(self.meta_label)

        # Filter snippet box
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Filter text within snippet...")
        self.search_edit.textChanged.connect(self._on_search_text_changed)
        rp_layout.addWidget(self.search_edit)

        self.text_preview = QTextBrowser()
        self.text_preview.setStyleSheet(
            "background-color: #0f172a; color: #f8fafc; font-family: Segoe UI, sans-serif; "
            "font-size: 12px; border: 1px solid #334155; border-radius: 6px; padding: 8px;"
        )
        rp_layout.addWidget(self.text_preview, 1)

        btn_row = QHBoxLayout()
        copy_btn = QPushButton("📋 Copy Snippet")
        copy_btn.clicked.connect(self._copy_snippet)

        open_file_btn = QPushButton("📄 Open File")
        open_file_btn.clicked.connect(self._open_file)

        open_folder_btn = QPushButton("📁 Open Folder")
        open_folder_btn.clicked.connect(self._open_folder)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        btn_row.addWidget(copy_btn)
        btn_row.addWidget(open_file_btn)
        btn_row.addWidget(open_folder_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        rp_layout.addLayout(btn_row)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        if self.chunks:
            row_to_select = self.initial_index if 0 <= self.initial_index < len(self.chunks) else 0
            self.chunk_list.setCurrentRow(row_to_select)

    def _on_chunk_selected(self, row: int):
        if 0 <= row < len(self.chunks):
            c = self.chunks[row]
            sec_info = f" &nbsp;|&nbsp; <b>Section:</b> <code>{c.section}</code>" if getattr(c, "section", "") else ""
            self.meta_label.setText(
                f"<b>File:</b> {c.file_name} &nbsp;|&nbsp; <b>Page:</b> {c.page}{sec_info} &nbsp;|&nbsp; "
                f"<b>Length:</b> {c.char_count} chars ({c.word_count} words)<br>"
                f"<b>Path:</b> <code>{c.doc_path}</code>"
            )
            self._render_snippet_text(c.text)

    def _render_snippet_text(self, text: str):
        search_query = self.search_edit.text().strip().lower()
        if not search_query:
            self.text_preview.setPlainText(text)
            return

        escaped_lines = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        pattern = re.compile(re.escape(search_query), re.IGNORECASE)
        highlighted = pattern.sub(
            lambda m: f'<span style="background-color: #eab308; color: #000000; font-weight: bold;">{m.group(0)}</span>',
            escaped_lines
        )
        self.text_preview.setHtml(f"<pre style='font-family: inherit; font-size: inherit;'>{highlighted}</pre>")

    def _on_search_text_changed(self):
        row = self.chunk_list.currentRow()
        if 0 <= row < len(self.chunks):
            self._render_snippet_text(self.chunks[row].text)

    def _copy_snippet(self):
        row = self.chunk_list.currentRow()
        if 0 <= row < len(self.chunks):
            QGuiApplication.clipboard().setText(self.chunks[row].text)
            QMessageBox.information(self, "Copied", "Source snippet copied to clipboard!")

    def _open_file(self):
        row = self.chunk_list.currentRow()
        if 0 <= row < len(self.chunks):
            p = Path(self.chunks[row].doc_path)
            if p.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
            else:
                QMessageBox.warning(self, "File Not Found", f"Cannot open missing file:\n{p}")

    def _open_folder(self):
        row = self.chunk_list.currentRow()
        if 0 <= row < len(self.chunks):
            folder = Path(self.chunks[row].doc_path).parent
            if folder.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
            else:
                QMessageBox.warning(self, "Folder Not Found", f"Folder does not exist:\n{folder}")


class IndexWorker(QThread):
    progress = Signal(int, int, str)
    done = Signal(int)
    failed = Signal(str)

    def __init__(self, engine: RagEngine, folder_path: str, force_reindex: bool = False):
        super().__init__()
        self.engine = engine
        self.folder_path = folder_path
        self.force_reindex = force_reindex

    def run(self):
        try:
            count = self.engine.index_folder(
                self.folder_path,
                progress_cb=lambda curr, total, msg: self.progress.emit(curr, total, msg),
                force_reindex=self.force_reindex,
            )
            self.done.emit(count)
        except Exception as e:
            self.failed.emit(str(e))


class StreamQueryWorker(QThread):
    token = Signal(str)
    done = Signal(list)  # cited_chunks
    failed = Signal(str)

    def __init__(
        self,
        engine: RagEngine,
        question: str,
        history: list[dict],
        file_types: list[str] | None = None,
        top_k: int = 3,
        score_threshold: float = 0.0,
        use_hyde: bool = False,
    ):
        super().__init__()
        self.engine = engine
        self.question = question
        self.history = history
        self.file_types = file_types
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.use_hyde = use_hyde
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True

    def run(self):
        try:
            gen = self.engine.chat_with_docs_stream(
                self.question,
                self.history,
                top_k=self.top_k,
                file_types=self.file_types,
                hybrid=True,
                score_threshold=self.score_threshold,
                compress_history_enabled=True,
                use_hyde=self.use_hyde,
            )
            cited = []
            try:
                while not self.is_stopped:
                    t = next(gen)
                    if self.is_stopped:
                        break
                    self.token.emit(t)
            except StopIteration as e:
                cited = e.value or []
            self.done.emit(cited)
        except Exception as e:
            self.failed.emit(str(e))


class ChatTab(QWidget):
    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self.settings = settings or Settings()
        self.engine = RagEngine(self.settings)
        self.history: list[dict] = []
        self.last_cited_chunks: list[DocumentChunk] = []
        self.index_worker: IndexWorker | None = None
        self.query_worker: StreamQueryWorker | None = None
        self.current_assistant_text: str = ""
        self.is_streaming_active: bool = False

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Folder indexing
        index_group = QGroupBox("1. Document Source Folder (Incremental Cache & Instant Reload)")
        ig_layout = QVBoxLayout(index_group)

        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Select folder containing documents you want to search and chat with...")
        self.folder_edit.textChanged.connect(self._on_folder_text_changed)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_folder)

        self.index_btn = QPushButton("⚡ Index Folder for AI Search")
        self.index_btn.setStyleSheet("font-weight: bold; padding: 5px 12px;")
        self.index_btn.clicked.connect(lambda: self.start_indexing(force_reindex=False))

        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(browse_btn)
        folder_row.addWidget(self.index_btn)
        ig_layout.addLayout(folder_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        ig_layout.addWidget(self.progress_bar)

        self.index_status = QLabel("No folder indexed yet. Index a folder to enable semantic search & grounded Q&A.")
        self.index_status.setStyleSheet("color: #64748b; font-size: 12px;")
        ig_layout.addWidget(self.index_status)

        layout.addWidget(index_group)

        # 2. Chat history
        chat_group = QGroupBox("2. Chat with Your Documents (Local Llama 3.2 - Real-Time Streaming)")
        cg_layout = QVBoxLayout(chat_group)

        self.chat_browser = QTextBrowser()
        self.chat_browser.setOpenExternalLinks(False)
        self.chat_browser.anchorClicked.connect(self._on_anchor_clicked)
        self.chat_browser.setStyleSheet(
            "background-color: #0f172a; color: #f8fafc; font-family: Segoe UI, sans-serif; font-size: 13px; padding: 10px;"
        )
        self._append_system_msg(
            "👋 Welcome to <b>Polaris Document Chat</b>!<br>"
            "Index a folder above, then ask any question about your local files.<br>"
            "Responses stream in real-time with verified citation badges you can click to inspect."
        )
        cg_layout.addWidget(self.chat_browser, 1)

        # 3. Filter, Top-K, Strictness & HyDE bar
        filter_row = QHBoxLayout()
        filter_label = QLabel("Scope:")
        filter_label.setStyleSheet("font-size: 12px; color: #64748b; font-weight: bold;")
        self.file_type_combo = QComboBox()
        self.file_type_combo.addItems([
            "All Supported Files",
            "PDFs Only (*.pdf)",
            "Documents & Notes (*.docx, *.txt, *.md, *.rst)",
            "Source Code & Data (*.py, *.js, *.ts, *.json, *.csv, *.sql)",
        ])
        filter_row.addWidget(filter_label)
        filter_row.addWidget(self.file_type_combo)

        top_k_label = QLabel("Top-K:")
        top_k_label.setStyleSheet("font-size: 12px; color: #64748b; font-weight: bold; margin-left: 8px;")
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 10)
        self.top_k_spin.setValue(3)
        self.top_k_spin.setToolTip("Number of context chunks retrieved for answer grounding.")
        filter_row.addWidget(top_k_label)
        filter_row.addWidget(self.top_k_spin)

        thresh_label = QLabel("Strictness:")
        thresh_label.setStyleSheet("font-size: 12px; color: #64748b; font-weight: bold; margin-left: 8px;")
        self.threshold_combo = QComboBox()
        self.threshold_combo.addItems([
            "Normal (0.0)",
            "Balanced (0.012)",
            "Strict (0.022)",
        ])
        self.threshold_combo.setToolTip("Minimum relevance threshold required to include a document.")
        filter_row.addWidget(thresh_label)
        filter_row.addWidget(self.threshold_combo)

        self.hyde_check = QCheckBox("HyDE")
        self.hyde_check.setChecked(False)
        self.hyde_check.setStyleSheet("font-size: 12px; color: #334155; font-weight: bold; margin-left: 8px;")
        self.hyde_check.setToolTip("Hypothetical Document Embeddings: Expands semantic queries for higher factual recall.")
        filter_row.addWidget(self.hyde_check)

        filter_row.addStretch()

        self.inspect_sources_btn = QPushButton("🔎 Inspect Source Citations")
        self.inspect_sources_btn.setEnabled(False)
        self.inspect_sources_btn.setStyleSheet("font-size: 12px; padding: 3px 10px;")
        self.inspect_sources_btn.clicked.connect(lambda: self.show_source_viewer(0))
        filter_row.addWidget(self.inspect_sources_btn)

        cg_layout.addLayout(filter_row)

        # 4. Input row
        input_row = QHBoxLayout()
        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText("Ask a question about your files... (e.g. 'What are the main findings in the report?')")
        self.query_edit.returnPressed.connect(self._on_query_submit)

        self.send_btn = QPushButton("Ask AI")
        self.send_btn.setStyleSheet("font-weight: bold; background-color: #2563eb; color: white; padding: 6px 16px;")
        self.send_btn.clicked.connect(self._on_query_submit)

        export_btn = QPushButton("📥 Export Chat")
        export_btn.clicked.connect(self.export_chat)

        clear_btn = QPushButton("Clear Chat")
        clear_btn.clicked.connect(self.clear_chat)

        input_row.addWidget(self.query_edit, 1)
        input_row.addWidget(self.send_btn)
        input_row.addWidget(export_btn)
        input_row.addWidget(clear_btn)
        cg_layout.addLayout(input_row)

        layout.addWidget(chat_group, 1)

    def _get_selected_threshold(self) -> float:
        idx = self.threshold_combo.currentIndex()
        if idx == 1:
            return 0.012
        elif idx == 2:
            return 0.022
        return 0.0

    def _get_selected_file_types(self) -> list[str] | None:
        idx = self.file_type_combo.currentIndex()
        if idx == 1:
            return [".pdf"]
        elif idx == 2:
            return [".docx", ".doc", ".txt", ".md", ".markdown", ".rst", ".rtf", ".log", ".tex"]
        elif idx == 3:
            return [
                ".py", ".pyw", ".js", ".jsx", ".ts", ".tsx", ".c", ".cpp", ".cc", ".cxx",
                ".h", ".hpp", ".cs", ".java", ".go", ".rs", ".rb", ".php", ".swift",
                ".kt", ".kts", ".dart", ".lua", ".r", ".scala", ".pl", ".pm", ".sh",
                ".bash", ".zsh", ".bat", ".cmd", ".ps1", ".html", ".htm", ".css", ".scss",
                ".json", ".jsonl", ".csv", ".tsv", ".xml", ".yaml", ".yml", ".toml",
                ".ini", ".cfg", ".conf", ".sql", ".env"
            ]
        return None

    def _on_folder_text_changed(self, text: str):
        folder = text.strip()
        if folder and Path(folder).is_dir() and self.engine.has_cache(folder):
            if self.engine.load_cache(folder):
                self.index_status.setText(
                    f"⚡ Cached index loaded instantly: {len(self.engine.chunks)} chunk(s) ready! Click Re-index to scan for edits."
                )
                self.index_status.setStyleSheet("color: #16a34a; font-size: 12px; font-weight: bold;")

    def _on_anchor_clicked(self, url: QUrl):
        scheme = url.scheme()
        if scheme == "citation":
            try:
                chunk_idx = int(url.path())
                self.show_source_viewer(initial_index=chunk_idx)
            except Exception:
                self.show_source_viewer(initial_index=0)
        elif scheme == "file":
            QDesktopServices.openUrl(url)

    def show_source_viewer(self, initial_index: int = 0):
        if not self.last_cited_chunks:
            QMessageBox.information(self, "No Citations", "No citations available to inspect.")
            return
        dialog = SourceViewerDialog(self.last_cited_chunks, initial_index=initial_index, parent=self)
        dialog.exec()

    def export_chat(self):
        if not self.history:
            QMessageBox.information(self, "Export Chat", "No conversation history to export yet.")
            return

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"polaris_chat_export_{timestamp_str}.md"

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Chat Session",
            default_filename,
            "Markdown Files (*.md);;Text Files (*.txt);;All Files (*.*)",
        )
        if not out_path:
            return

        try:
            content = format_chat_export(self.history, include_snippets=True)
            Path(out_path).write_text(content, encoding="utf-8")
            self._append_system_msg(f"💾 Chat exported successfully to <code>{Path(out_path).name}</code>")
            QMessageBox.information(self, "Export Successful", f"Chat saved to:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to save chat export: {e}")

    def _browse_folder(self):
        f = QFileDialog.getExistingDirectory(self, "Select folder to index")
        if f:
            self.folder_edit.setText(f)
            self._on_folder_text_changed(f)

    def start_indexing(self, force_reindex: bool = False):
        folder = self.folder_edit.text().strip()
        if not folder or not Path(folder).is_dir():
            QMessageBox.warning(self, "Invalid Folder", "Please select a valid folder to index.")
            return

        self.index_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)
        self.index_status.setText("Scanning documents and verifying cache...")
        self.index_status.setStyleSheet("color: #64748b; font-size: 12px;")

        self.index_worker = IndexWorker(self.engine, folder, force_reindex=force_reindex)
        self.index_worker.progress.connect(self._on_index_progress)
        self.index_worker.done.connect(self._on_index_done)
        self.index_worker.failed.connect(self._on_index_failed)
        self.index_worker.start()

    def _on_index_progress(self, curr: int, total: int, msg: str):
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(curr)
        self.index_status.setText(msg)

    def _on_index_done(self, count: int):
        self.progress_bar.hide()
        self.index_btn.setEnabled(True)
        self.index_status.setText(f"✓ Ready: {count} searchable chunk(s) indexed & cached. Ask any question below!")
        self.index_status.setStyleSheet("color: #16a34a; font-size: 12px; font-weight: bold;")
        self._append_system_msg(f"✅ Indexed {count} document chunks from <i>{self.engine.indexed_folder}</i>.")

    def _on_index_failed(self, err: str):
        self.progress_bar.hide()
        self.index_btn.setEnabled(True)
        self.index_status.setText(f"Indexing failed: {err}")
        self.index_status.setStyleSheet("color: #ef4444; font-size: 12px;")
        QMessageBox.critical(self, "Indexing Error", f"Failed to index documents: {err}")

    def _on_query_submit(self):
        if self.is_streaming_active:
            self.stop_generation()
        else:
            self.send_question()

    def stop_generation(self):
        """Aborts active token streaming immediately."""
        if self.query_worker and self.query_worker.isRunning():
            self.query_worker.stop()
            self._append_system_msg("⏹ Generation stopped by user.")
        self._reset_input_ui()

    def _reset_input_ui(self):
        self.is_streaming_active = False
        self.send_btn.setText("Ask AI")
        self.send_btn.setStyleSheet("font-weight: bold; background-color: #2563eb; color: white; padding: 6px 16px;")
        self.send_btn.setEnabled(True)
        self.query_edit.setEnabled(True)
        self.query_edit.setFocus()

    def send_question(self):
        query = self.query_edit.text().strip()
        if not query:
            return

        self.query_edit.clear()
        self._append_user_msg(query)
        self.history.append({"role": "user", "content": query})

        self.is_streaming_active = True
        self.send_btn.setText("⏹ Stop")
        self.send_btn.setStyleSheet("font-weight: bold; background-color: #dc2626; color: white; padding: 6px 16px;")
        self.send_btn.setEnabled(True)
        self.query_edit.setEnabled(False)
        self.current_assistant_text = ""

        # Prepare assistant bubble
        self.chat_browser.append(
            '<div style="margin: 8px 0; text-align: left;">'
            '<div id="active_msg" style="background-color: #1e293b; color: #f8fafc; padding: 10px 14px; border-radius: 12px; display: inline-block; max-width: 90%; border: 1px solid #334155;">'
            '<b style="color: #60a5fa;">Polaris:</b><br><span id="content">thinking...</span></div></div>'
        )

        file_types = self._get_selected_file_types()
        top_k = self.top_k_spin.value()
        score_threshold = self._get_selected_threshold()
        use_hyde = self.hyde_check.isChecked()

        self.query_worker = StreamQueryWorker(
            self.engine,
            query,
            self.history,
            file_types=file_types,
            top_k=top_k,
            score_threshold=score_threshold,
            use_hyde=use_hyde,
        )
        self.query_worker.token.connect(self._on_token)
        self.query_worker.done.connect(self._on_stream_done)
        self.query_worker.failed.connect(self._on_stream_failed)
        self.query_worker.start()

    def _on_token(self, token: str):
        if not self.current_assistant_text:
            self.current_assistant_text = token
        else:
            self.current_assistant_text += token

        cursor = self.chat_browser.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.chat_browser.setTextCursor(cursor)
        self._refresh_latest_assistant_bubble(self.current_assistant_text, cited=[])

    def _refresh_latest_assistant_bubble(self, text: str, cited: list[DocumentChunk]):
        formatted_text = text.replace("\n", "<br>")
        citations_html = ""
        if cited:
            cite_badges = []
            for i, c in enumerate(cited):
                sec_tag = f" § {c.section}" if getattr(c, "section", "") else ""
                badge = (
                    f'<a href="citation:{i}" style="text-decoration:none; display:inline-block; '
                    f'background-color:#1e3a8a; color:#93c5fd; padding:3px 8px; border-radius:6px; '
                    f'font-size:11px; margin:2px 4px 2px 0; border:1px solid #3b82f6;">'
                    f'📄 [{i + 1}] {c.file_name}{sec_tag} (Page {c.page})</a>'
                )
                cite_badges.append(badge)

            citations_html = (
                f'<div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #334155;">'
                f'<div style="font-size: 11px; color: #94a3b8; font-weight: bold; margin-bottom: 4px;">'
                f'🔍 Grounded Citations (click to inspect source snippet):</div>'
                f'{" ".join(cite_badges)}'
                f'</div>'
            )

        html = f"""
        <div style="margin: 8px 0; text-align: left;">
            <div style="background-color: #1e293b; color: #f8fafc; padding: 10px 14px; border-radius: 12px; display: inline-block; max-width: 90%; border: 1px solid #334155; line-height: 1.4;">
                <b style="color: #60a5fa;">Polaris:</b><br>{formatted_text}
                {citations_html}
            </div>
        </div>
        """
        doc = self.chat_browser.document()
        cursor = self.chat_browser.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.select(cursor.SelectionType.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.insertHtml(html)
        self.chat_browser.verticalScrollBar().setValue(
            self.chat_browser.verticalScrollBar().maximum()
        )

    def _on_stream_done(self, cited: list):
        self._reset_input_ui()

        self.last_cited_chunks = cited
        if cited:
            self.inspect_sources_btn.setEnabled(True)
            self.inspect_sources_btn.setText(f"🔎 Inspect Citations ({len(cited)})")
        else:
            self.inspect_sources_btn.setEnabled(False)
            self.inspect_sources_btn.setText("🔎 Inspect Source Citations")

        self._refresh_latest_assistant_bubble(self.current_assistant_text, cited=cited)
        self.history.append({"role": "assistant", "content": self.current_assistant_text, "cited": cited})

    def _on_stream_failed(self, err: str):
        self._reset_input_ui()
        self._append_system_msg(f"⚠️ Error: {err}")

    def _append_user_msg(self, text: str):
        html = f"""
        <div style="margin: 8px 0; text-align: right;">
            <span style="background-color: #2563eb; color: white; padding: 6px 12px; border-radius: 12px; display: inline-block;">
                <b>You:</b> {text}
            </span>
        </div>
        """
        self.chat_browser.append(html)

    def _append_system_msg(self, text: str):
        html = f"""
        <div style="margin: 6px 0; text-align: center; color: #94a3b8; font-size: 12px;">
            {text}
        </div>
        """
        self.chat_browser.append(html)

    def clear_chat(self):
        self.history.clear()
        self.last_cited_chunks = []
        self.inspect_sources_btn.setEnabled(False)
        self.inspect_sources_btn.setText("🔎 Inspect Source Citations")
        self.chat_browser.clear()
        self._append_system_msg("Chat history cleared.")
