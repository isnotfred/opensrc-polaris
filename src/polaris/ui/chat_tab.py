"""AI-Powered Search & Document Chatbot tab with Messenger-style conversation stream, real-time token streaming, citation badges, hybrid retrieval, HyDE, and stop generation."""
from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication, QTextCursor
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


def open_file_at_location(file_path: str | Path, line: int = 1, page: int = 1) -> bool:
    """
    Opens a file jumping directly to the specified line number in the user's editor
    (e.g., VS Code 'code --goto <path>:<line>', Cursor, Sublime),
    or falls back to system default opener (QDesktopServices).
    """
    p = Path(file_path).resolve()
    if not p.exists():
        return False

    ext = p.suffix.lower()

    if ext != ".pdf":
        code_bin = shutil.which("code") or shutil.which("cursor")
        if code_bin:
            try:
                subprocess.Popen([code_bin, "--goto", f"{str(p)}:{line}"], shell=sys.platform == "win32")
                return True
            except Exception:
                pass

        subl_bin = shutil.which("subl")
        if subl_bin:
            try:
                subprocess.Popen([subl_bin, f"{str(p)}:{line}"], shell=sys.platform == "win32")
                return True
            except Exception:
                pass

    try:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
        return True
    except Exception:
        return False


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
            line_tag = f" [L{c.start_line}]" if getattr(c, "start_line", 1) > 1 else ""
            item = QListWidgetItem(f"📄 [{i + 1}] {c.file_name}{sec_tag}{line_tag} (Page {c.page})")
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

        self.open_file_btn = QPushButton("📄 Open File")
        self.open_file_btn.clicked.connect(self._open_file)

        open_folder_btn = QPushButton("📁 Open Folder")
        open_folder_btn.clicked.connect(self._open_folder)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        btn_row.addWidget(copy_btn)
        btn_row.addWidget(self.open_file_btn)
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
            line_info = f" &nbsp;|&nbsp; <b>Line:</b> {c.start_line}–{c.end_line}" if getattr(c, "start_line", 1) > 1 else ""
            self.meta_label.setText(
                f"<b>File:</b> {c.file_name} &nbsp;|&nbsp; <b>Page:</b> {c.page}{sec_info}{line_info} &nbsp;|&nbsp; "
                f"<b>Length:</b> {c.char_count} chars ({c.word_count} words)<br>"
                f"<b>Path:</b> <code>{c.doc_path}</code>"
            )
            if getattr(c, "start_line", 1) > 1:
                self.open_file_btn.setText(f"📄 Open at Line {c.start_line}")
                self.open_file_btn.setToolTip(f"Open directly in code editor at line {c.start_line}")
            else:
                self.open_file_btn.setText("📄 Open File")
                self.open_file_btn.setToolTip("Open file in external viewer")
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
            c = self.chunks[row]
            p = Path(c.doc_path)
            if p.exists():
                open_file_at_location(p, line=getattr(c, "start_line", 1), page=getattr(c, "page", 1))
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


def _format_bubble_text(text: str) -> str:
    """Format markdown-style text into clean HTML for chat bubbles."""
    if not text:
        return ""

    escaped = html.escape(text)

    # Multi-line code blocks
    def _code_block_sub(match):
        code = match.group(1).strip()
        return (
            f'<div style="background-color: #0b1120; border: 1px solid #334155; padding: 8px 12px; '
            f'border-radius: 6px; color: #e2e8f0; font-family: Consolas, monospace; font-size: 12px; '
            f'margin: 6px 0; white-space: pre-wrap;">{code}</div>'
        )

    escaped = re.sub(r'```(?:[a-zA-Z0-9_-]*\\n)?([\s\S]*?)```', _code_block_sub, escaped)

    # Inline code
    escaped = re.sub(
        r'`([^`]+)`',
        r'<code style="background-color: #0b1120; padding: 2px 5px; border-radius: 4px; color: #38bdf8; font-family: Consolas, monospace; font-size: 12px;">\1</code>',
        escaped,
    )

    # Bold: **text**
    escaped = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', escaped)

    # Italic: *text*
    escaped = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', escaped)

    # Format bullet lists, numbered lists, and paragraphs
    lines = escaped.split("\n")
    chunks = []
    current_list_type = None  # None, "ul", "ol"
    current_list_items = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            if current_list_type != "ul":
                if current_list_type == "ol":
                    chunks.append("<ol style='margin: 4px 0; padding-left: 18px;'>" + "".join(current_list_items) + "</ol>")
                    current_list_items = []
                current_list_type = "ul"
            current_list_items.append(f"<li>{stripped[2:]}</li>")
        elif re.match(r'^\d+\.\s', stripped):
            if current_list_type != "ol":
                if current_list_type == "ul":
                    chunks.append("<ul style='margin: 4px 0; padding-left: 18px;'>" + "".join(current_list_items) + "</ul>")
                    current_list_items = []
                current_list_type = "ol"
            m = re.match(r'^\d+\.\s+(.*)', stripped)
            current_list_items.append(f"<li>{m.group(1) if m else stripped}</li>")
        else:
            if current_list_type:
                tag = current_list_type
                chunks.append(f"<{tag} style='margin: 4px 0; padding-left: 18px;'>" + "".join(current_list_items) + f"</{tag}>")
                current_list_type = None
                current_list_items = []
            if line:
                chunks.append(line)

    if current_list_type:
        tag = current_list_type
        chunks.append(f"<{tag} style='margin: 4px 0; padding-left: 18px;'>" + "".join(current_list_items) + f"</{tag}>")

    return "<br>".join(chunks)


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
        use_reranker: bool = False,
    ):
        super().__init__()
        self.engine = engine
        self.question = question
        self.history = history
        self.file_types = file_types
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.use_hyde = use_hyde
        self.use_reranker = use_reranker
        self.is_stopped = False
        self._stopped = False

    def stop(self):
        self.is_stopped = True
        self._stopped = True

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
                use_reranker=self.use_reranker,
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
            if not self.is_stopped:
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
        self.current_question: str = ""
        self.active_bubble_start_pos: int = 0
        self.message_count: int = 0
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
        self.folder_edit.textChanged.connect(self._on_folder_changed)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_folder)

        self.index_btn = QPushButton("⚡ Index Folder for AI Search")
        self.index_btn.setStyleSheet(
            "font-weight: bold; padding: 6px 14px; background-color: #1e293b; color: #94a3b8; "
            "border: 1px solid #334155; border-radius: 6px;"
        )
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

        # 2. Messenger-Style Chat Stream
        chat_group = QGroupBox("2. Conversation Stream (Local Ollama Llama 3.2)")
        cg_layout = QVBoxLayout(chat_group)

        self.chat_browser = QTextBrowser()
        self.chat_browser.setOpenExternalLinks(False)
        self.chat_browser.anchorClicked.connect(self._on_anchor_clicked)
        self.chat_browser.setStyleSheet(
            "background-color: #0b1120; color: #f8fafc; font-family: Segoe UI, system-ui, sans-serif; "
            "font-size: 13px; padding: 12px; border: 1px solid #1e293b; border-radius: 8px;"
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

        self.rerank_check = QCheckBox("Re-rank")
        self.rerank_check.setChecked(False)
        self.rerank_check.setStyleSheet("font-size: 12px; color: #334155; font-weight: bold; margin-left: 8px;")
        self.rerank_check.setToolTip("Cross-Encoder Re-ranker: Re-ranks top candidates using cross-scoring for higher precision.")
        filter_row.addWidget(self.rerank_check)

        filter_row.addStretch()

        self.inspect_sources_btn = QPushButton("🔎 Inspect Source Citations")
        self.inspect_sources_btn.setEnabled(False)
        self.inspect_sources_btn.setStyleSheet("font-size: 12px; padding: 3px 10px;")
        self.inspect_sources_btn.clicked.connect(lambda: self.show_source_viewer(0))
        filter_row.addWidget(self.inspect_sources_btn)

        cg_layout.addLayout(filter_row)

        # 4. Messenger-Style Input Bar
        input_container = QWidget()
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(0, 4, 0, 0)

        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText("Type a message... (Press Enter to send)")
        self.query_edit.setStyleSheet(
            "background-color: #1e293b; color: white; border: 1px solid #334155; "
            "border-radius: 8px; padding: 8px 12px; font-size: 13px;"
        )
        self.query_edit.returnPressed.connect(self.send_question)
        input_layout.addWidget(self.query_edit, 1)

        self.send_btn = QPushButton("Ask AI")
        self.send_btn.setStyleSheet(
            "font-weight: bold; background-color: #2563eb; color: white; padding: 8px 18px; "
            "border-radius: 8px; min-width: 80px;"
        )
        self.send_btn.clicked.connect(self.send_question)
        input_layout.addWidget(self.send_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setStyleSheet(
            "font-weight: bold; background-color: #dc2626; color: white; padding: 8px 14px; "
            "border-radius: 8px;"
        )
        self.stop_btn.clicked.connect(self.stop_streaming)
        self.stop_btn.hide()
        input_layout.addWidget(self.stop_btn)

        self.export_btn = QPushButton("💾 Export")
        self.export_btn.setToolTip("Export this chat transcript to a Markdown/Text file")
        self.export_btn.clicked.connect(self.export_chat)
        input_layout.addWidget(self.export_btn)

        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.setToolTip("Clear session history and start fresh")
        self.clear_btn.clicked.connect(self.clear_chat)
        input_layout.addWidget(self.clear_btn)

        cg_layout.addWidget(input_container)
        layout.addWidget(chat_group, 1)

        self._show_welcome_banner()

    def _show_welcome_banner(self):
        self.chat_browser.clear()
        time_str = datetime.now().strftime("%I:%M %p")
        banner_html = f"""
        <div align="center" style="margin: 14px 0 20px 0;">
            <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 14px 20px; display: inline-block; max-width: 85%; text-align: center;">
                <div style="font-size: 15px; font-weight: bold; color: #38bdf8; margin-bottom: 4px;">💬 Polaris Document Chat</div>
                <div style="font-size: 12px; color: #cbd5e1; line-height: 1.4;">
                    Index a folder above, then ask any question about your documents.<br>
                    Responses stream in real-time with verified citation badges you can click to inspect.
                </div>
                <div style="margin-top: 6px; font-size: 10px; color: #64748b;">Session started at {time_str}</div>
            </div>
        </div>
        """
        self.chat_browser.append(banner_html)
        self._scroll_to_bottom()

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
        """Auto-load cached index when a previously-indexed folder is entered."""
        folder = text.strip()
        if folder and Path(folder).is_dir() and self.engine.has_cache(folder):
            if self.engine.load_cache(folder):
                self.index_status.setText(
                    f"⚡ Cached index loaded instantly: {len(self.engine.chunks)} chunk(s) ready! Click Re-index to scan for edits."
                )
                self.index_status.setStyleSheet("color: #16a34a; font-size: 12px; font-weight: bold;")

    def _on_folder_changed(self, text: str):
        """Style the index button based on folder validity."""
        folder = text.strip()
        if folder and Path(folder).is_dir():
            self.index_btn.setStyleSheet(
                "font-weight: bold; font-size: 13px; color: #ffffff; background-color: #16a34a; "
                "border: 2px solid #4ade80; border-radius: 6px; padding: 6px 16px;"
            )
        else:
            self.index_btn.setStyleSheet(
                "font-weight: bold; padding: 6px 14px; background-color: #1e293b; color: #94a3b8; "
                "border: 1px solid #334155; border-radius: 6px;"
            )

    def _scroll_to_bottom(self):
        sb = self.chat_browser.verticalScrollBar()
        sb.setValue(sb.maximum())

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

    def _browse_folder(self):
        f = QFileDialog.getExistingDirectory(self, "Select folder to index")
        if f:
            self.folder_edit.setText(f)

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
        self.index_btn.setStyleSheet(
            "font-weight: bold; color: #a7f3d0; background-color: #064e3b; "
            "border: 1px solid #059669; border-radius: 6px; padding: 6px 14px;"
        )
        self.index_status.setText(f"✓ Ready: {count} searchable chunk(s) indexed & cached. Ask any question below!")
        self.index_status.setStyleSheet("color: #16a34a; font-size: 12px; font-weight: bold;")
        self._append_system_msg(f"✅ Indexed {count} document chunks from <i>{self.engine.indexed_folder}</i>.")

    def _on_index_failed(self, err: str):
        self.progress_bar.hide()
        self.index_btn.setEnabled(True)
        self._on_folder_changed(self.folder_edit.text())
        self.index_status.setText(f"Indexing failed: {err}")
        self.index_status.setStyleSheet("color: #ef4444; font-size: 12px;")
        QMessageBox.critical(self, "Indexing Error", f"Failed to index documents: {err}")

    def stop_generation(self):
        """Aborts active token streaming immediately."""
        if self.query_worker and self.query_worker.isRunning():
            self.query_worker.stop()
            self._append_system_msg("⏹ Generation stopped by user.")
        self.is_streaming_active = False
        self.stop_btn.hide()
        self.send_btn.setText("Ask AI")
        self.send_btn.show()
        self.query_edit.setEnabled(True)
        self.query_edit.setFocus()

    def stop_streaming(self):
        self.stop_generation()

    def send_question(self):
        query = self.query_edit.text().strip()
        if not query:
            return

        self.query_edit.clear()
        self.current_question = query
        self.current_assistant_text = ""
        now_str = datetime.now().strftime("%I:%M %p")

        # 1. Append User Bubble (right-aligned, Messenger style)
        self._append_user_bubble(query, now_str)

        # 2. Append Assistant Bubble Placeholder (left-aligned, Messenger style)
        self._insert_assistant_bubble_placeholder(now_str)

        # 3. Update UI states
        self.is_streaming_active = True
        self.send_btn.setText("⏹ Stop")
        self.send_btn.hide()
        self.stop_btn.show()
        self.query_edit.setEnabled(False)
        self._scroll_to_bottom()

        # 5. Launch streaming query with full retrieval settings
        file_types = self._get_selected_file_types()
        top_k = self.top_k_spin.value()
        score_threshold = self._get_selected_threshold()
        use_hyde = self.hyde_check.isChecked()
        use_reranker = self.rerank_check.isChecked()

        self.query_worker = StreamQueryWorker(
            self.engine,
            query,
            self.history,
            file_types=file_types,
            top_k=top_k,
            score_threshold=score_threshold,
            use_hyde=use_hyde,
            use_reranker=use_reranker,
        )
        self.query_worker.token.connect(self._on_token)
        self.query_worker.done.connect(self._on_stream_done)
        self.query_worker.failed.connect(self._on_stream_failed)
        self.query_worker.start()

    def _append_user_bubble(self, text: str, time_str: str):
        formatted = _format_bubble_text(text)
        bubble_html = f"""
        <table width="100%" border="0" cellpadding="0" cellspacing="0" style="margin: 8px 0;">
            <tr>
                <td align="right">
                    <div style="background-color: #2563eb; color: #ffffff; padding: 10px 16px; border-radius: 16px 16px 4px 16px; display: inline-block; max-width: 80%; font-size: 13px; line-height: 1.4;">
                        {formatted}
                    </div>
                    <div style="color: #64748b; font-size: 10px; margin-top: 3px; margin-right: 4px;">{time_str}</div>
                </td>
            </tr>
        </table>
        """
        self.chat_browser.append(bubble_html)
        self.message_count += 1
        self._scroll_to_bottom()

    def _insert_assistant_bubble_placeholder(self, time_str: str):
        cursor = self.chat_browser.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertBlock()
        self.active_bubble_start_pos = cursor.position()

        placeholder_html = f"""
        <table width="100%" border="0" cellpadding="0" cellspacing="0" style="margin: 8px 0;">
            <tr>
                <td align="left">
                    <div style="color: #38bdf8; font-size: 11px; font-weight: bold; margin-bottom: 3px; margin-left: 4px;">🤖 Polaris AI</div>
                    <div style="background-color: #1e293b; color: #94a3b8; padding: 10px 16px; border-radius: 16px 16px 16px 4px; display: inline-block; max-width: 85%; font-size: 13px; line-height: 1.5; border: 1px solid #334155;">
                        <i>Thinking...</i>
                    </div>
                    <div style="color: #64748b; font-size: 10px; margin-top: 3px; margin-left: 4px;">{time_str}</div>
                </td>
            </tr>
        </table>
        """
        cursor.insertHtml(placeholder_html)
        self.chat_browser.setTextCursor(cursor)
        self.message_count += 1
        self._scroll_to_bottom()

    def _on_token(self, token: str):
        self.current_assistant_text += token
        self._update_active_assistant_bubble(self.current_assistant_text, cited=[], is_final=False)

    def _format_markdown_simple(self, text: str, cited: list[DocumentChunk] | None = None) -> str:
        """
        Renders markdown formatting with interactive inline citation badges.
        Applied on top of _format_bubble_text output for final responses.
        """
        formatted = _format_bubble_text(text)

        # Inline citations: [Source 1], [1], [Source 2] -> interactive links
        if cited:
            num_cited = len(cited)
            def replace_citation(m: re.Match) -> str:
                n = int(m.group(1))
                if 1 <= n <= num_cited:
                    idx = n - 1
                    return (
                        f'<a href="citation:{idx}" style="text-decoration:none; font-weight:bold; '
                        f'background-color:#1e3a8a; color:#93c5fd; padding:1px 5px; border-radius:4px; '
                        f'font-size:11px; border:1px solid #3b82f6;">[{n}]</a>'
                    )
                return m.group(0)

            formatted = re.sub(r"\[(?:Source\s*)?(\d+)\]", replace_citation, formatted)

        return formatted

    def _format_markdown_with_citations(self, text: str, cited: list[DocumentChunk] | None = None) -> str:
        return self._format_markdown_simple(text, cited=cited)

    def _update_active_assistant_bubble(self, text: str, cited: list, is_final: bool):
        now_str = datetime.now().strftime("%I:%M %p")

        if is_final and cited:
            formatted = self._format_markdown_with_citations(text, cited=cited)
        else:
            formatted = _format_bubble_text(text) if text else "<i>Thinking...</i>"

        citations_html = ""
        if is_final and cited:
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

        bubble_html = f"""
        <table width="100%" border="0" cellpadding="0" cellspacing="0" style="margin: 8px 0;">
            <tr>
                <td align="left">
                    <div style="color: #38bdf8; font-size: 11px; font-weight: bold; margin-bottom: 3px; margin-left: 4px;">🤖 Polaris AI</div>
                    <div style="background-color: #1e293b; color: #f8fafc; padding: 12px 16px; border-radius: 16px 16px 16px 4px; display: inline-block; max-width: 85%; font-size: 13px; line-height: 1.5; border: 1px solid #334155;">
                        {formatted}
                        {citations_html}
                    </div>
                    <div style="color: #64748b; font-size: 10px; margin-top: 3px; margin-left: 4px;">{now_str}</div>
                </td>
            </tr>
        </table>
        """

        # Update in-place without touching any previous history messages
        cursor = self.chat_browser.textCursor()
        cursor.setPosition(self.active_bubble_start_pos)
        cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertHtml(bubble_html)
        self.chat_browser.setTextCursor(cursor)
        self._scroll_to_bottom()

    def _on_stream_done(self, cited: list):
        self.is_streaming_active = False
        self.stop_btn.hide()
        self.send_btn.setText("Ask AI")
        self.send_btn.show()
        self.query_edit.setEnabled(True)
        self.query_edit.setFocus()

        # Filter and prioritize actively cited chunks if the assistant specifically cited them
        active_indices: list[int] = []
        if cited:
            for m in re.finditer(r"\[(?:Source\s*)?(\d+)\]", self.current_assistant_text):
                n = int(m.group(1)) - 1
                if 0 <= n < len(cited) and n not in active_indices:
                    active_indices.append(n)
            for i, c in enumerate(cited):
                if c.file_name.lower() in self.current_assistant_text.lower() and i not in active_indices:
                    active_indices.append(i)

        final_cited = [cited[i] for i in active_indices] if active_indices else cited

        self.last_cited_chunks = final_cited
        if final_cited:
            self.inspect_sources_btn.setEnabled(True)
            self.inspect_sources_btn.setText(f"🔎 Inspect Citations ({len(final_cited)})")
        else:
            self.inspect_sources_btn.setEnabled(False)
            self.inspect_sources_btn.setText("🔎 Inspect Source Citations")

        # Finalize the assistant bubble with complete citations
        self._update_active_assistant_bubble(self.current_assistant_text, cited=final_cited, is_final=True)

        # Record turn in multi-turn conversation history
        if not self.history or self.history[-1].get("role") != "user":
            if self.current_question:
                self.history.append({"role": "user", "content": self.current_question})
        asst_entry: dict = {"role": "assistant", "content": self.current_assistant_text}
        if final_cited:
            asst_entry["cited"] = final_cited
        self.history.append(asst_entry)
        self._scroll_to_bottom()

    def _on_stream_failed(self, err: str):
        self.is_streaming_active = False
        self.stop_btn.hide()
        self.send_btn.setText("Ask AI")
        self.send_btn.show()
        self.query_edit.setEnabled(True)
        if self.history and self.history[-1].get("role") == "user":
            self.history.pop()
        self._append_system_msg(f"⚠️ Error: {err}")
        self._scroll_to_bottom()

    def _append_system_msg(self, text: str):
        sys_html = f"""
        <div align="center" style="margin: 8px 0;">
            <span style="background-color: #1e293b; color: #94a3b8; font-size: 11px; padding: 4px 12px; border-radius: 12px; border: 1px solid #334155;">
                {text}
            </span>
        </div>
        """
        self.chat_browser.append(sys_html)
        self._scroll_to_bottom()

    def _append_user_msg(self, text: str):
        """Helper for test compatibility."""
        self._append_user_bubble(text, datetime.now().strftime("%I:%M %p"))

    def clear_chat(self):
        if self.history:
            confirm = QMessageBox.question(
                self,
                "Clear Conversation",
                "Clear the entire chat session and start a new conversation?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return

        self.history.clear()
        self.last_cited_chunks = []
        self.active_bubble_start_pos = 0
        self.message_count = 0
        self.inspect_sources_btn.setEnabled(False)
        self.inspect_sources_btn.setText("🔎 Inspect Source Citations")
        self._show_welcome_banner()

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
