"""AI-Powered Search & Document Chatbot tab with real-time token streaming and hybrid search."""
from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
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
    def __init__(self, chunks: list[DocumentChunk], parent: QWidget | None = None):
        super().__init__(parent)
        self.chunks = chunks
        self.setWindowTitle("Polaris - Source Inspection & Citations")
        self.resize(750, 480)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel(f"<b>Retrieved Context Sources ({len(self.chunks)} chunk(s))</b>")
        header.setStyleSheet("font-size: 13px; color: #1e293b; margin-bottom: 4px;")
        layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left list of chunks
        self.chunk_list = QListWidget()
        for i, c in enumerate(self.chunks):
            item = QListWidgetItem(f"📄 [{i + 1}] {c.file_name} (Page {c.page})")
            self.chunk_list.addItem(item)
        self.chunk_list.currentRowChanged.connect(self._on_chunk_selected)
        splitter.addWidget(self.chunk_list)

        # Right pane: details
        right_panel = QWidget()
        rp_layout = QVBoxLayout(right_panel)
        rp_layout.setContentsMargins(4, 0, 0, 0)

        self.meta_label = QLabel()
        self.meta_label.setStyleSheet("color: #475569; font-size: 12px; margin-bottom: 6px;")
        rp_layout.addWidget(self.meta_label)

        self.text_preview = QTextBrowser()
        self.text_preview.setStyleSheet(
            "background-color: #0f172a; color: #f8fafc; font-family: Segoe UI, sans-serif; "
            "font-size: 12px; border: 1px solid #334155; border-radius: 6px; padding: 8px;"
        )
        rp_layout.addWidget(self.text_preview, 1)

        btn_row = QHBoxLayout()
        copy_btn = QPushButton("📋 Copy Snippet")
        copy_btn.clicked.connect(self._copy_snippet)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(copy_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        rp_layout.addLayout(btn_row)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        if self.chunks:
            self.chunk_list.setCurrentRow(0)

    def _on_chunk_selected(self, row: int):
        if 0 <= row < len(self.chunks):
            c = self.chunks[row]
            self.meta_label.setText(
                f"<b>File:</b> {c.file_name} &nbsp;|&nbsp; <b>Page:</b> {c.page} &nbsp;|&nbsp; <b>Path:</b> <code>{c.doc_path}</code>"
            )
            self.text_preview.setPlainText(c.text)

    def _copy_snippet(self):
        row = self.chunk_list.currentRow()
        if 0 <= row < len(self.chunks):
            QGuiApplication.clipboard().setText(self.chunks[row].text)
            QMessageBox.information(self, "Copied", "Source snippet copied to clipboard!")


class IndexWorker(QThread):
    progress = Signal(int, int, str)
    done = Signal(int)
    failed = Signal(str)

    def __init__(self, engine: RagEngine, folder_path: str):
        super().__init__()
        self.engine = engine
        self.folder_path = folder_path

    def run(self):
        try:
            count = self.engine.index_folder(
                self.folder_path,
                progress_cb=lambda curr, total, msg: self.progress.emit(curr, total, msg),
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
    ):
        super().__init__()
        self.engine = engine
        self.question = question
        self.history = history
        self.file_types = file_types

    def run(self):
        try:
            gen = self.engine.chat_with_docs_stream(
                self.question,
                self.history,
                file_types=self.file_types,
                hybrid=True,
            )
            cited = []
            try:
                while True:
                    t = next(gen)
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

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Folder indexing
        index_group = QGroupBox("1. Document Source Folder (PDF, DOCX, TXT, MD, Code)")
        ig_layout = QVBoxLayout(index_group)

        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Select folder containing documents you want to search and chat with...")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_folder)

        self.index_btn = QPushButton("⚡ Index Folder for AI Search")
        self.index_btn.setStyleSheet("font-weight: bold; padding: 5px 12px;")
        self.index_btn.clicked.connect(self.start_indexing)

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
        self.chat_browser.setStyleSheet(
            "background-color: #0f172a; color: #f8fafc; font-family: Segoe UI, sans-serif; font-size: 13px; padding: 10px;"
        )
        self._append_system_msg(
            "👋 Welcome to <b>Polaris Document Chat</b>!<br>"
            "Index a folder above, then ask any question about your local files.<br>"
            "Responses stream in real-time with verified citations."
        )
        cg_layout.addWidget(self.chat_browser, 1)

        # 3. Filter and source inspection bar
        filter_row = QHBoxLayout()
        filter_label = QLabel("Search Scope:")
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
        filter_row.addStretch()

        self.inspect_sources_btn = QPushButton("🔎 Inspect Source Citations")
        self.inspect_sources_btn.setEnabled(False)
        self.inspect_sources_btn.setStyleSheet("font-size: 12px; padding: 3px 10px;")
        self.inspect_sources_btn.clicked.connect(self.show_source_viewer)
        filter_row.addWidget(self.inspect_sources_btn)

        cg_layout.addLayout(filter_row)

        # 4. Input row
        input_row = QHBoxLayout()
        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText("Ask a question about your files... (e.g. 'What are the main findings in the report?')")
        self.query_edit.returnPressed.connect(self.send_question)

        self.send_btn = QPushButton("Ask AI")
        self.send_btn.setStyleSheet("font-weight: bold; background-color: #2563eb; color: white; padding: 6px 16px;")
        self.send_btn.clicked.connect(self.send_question)

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

    def _get_selected_file_types(self) -> list[str] | None:
        idx = self.file_type_combo.currentIndex()
        if idx == 1:
            return [".pdf"]
        elif idx == 2:
            return [".docx", ".doc", ".txt", ".md", ".rst"]
        elif idx == 3:
            return [
                ".py", ".js", ".ts", ".html", ".css", ".json", ".csv",
                ".xml", ".yaml", ".yml", ".toml", ".ini", ".sql"
            ]
        return None

    def show_source_viewer(self):
        if not self.last_cited_chunks:
            QMessageBox.information(self, "No Citations", "No citations available to inspect.")
            return
        dialog = SourceViewerDialog(self.last_cited_chunks, self)
        dialog.exec()

    def export_chat(self):
        if not self.history:
            QMessageBox.information(self, "Export Chat", "No conversation history to export yet.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Chat Session",
            "polaris_chat_export.md",
            "Markdown Files (*.md);;Text Files (*.txt);;All Files (*.*)",
        )
        if not out_path:
            return

        try:
            content = format_chat_export(self.history)
            Path(out_path).write_text(content, encoding="utf-8")
            self._append_system_msg(f"💾 Chat exported successfully to <code>{Path(out_path).name}</code>")
            QMessageBox.information(self, "Export Successful", f"Chat saved to:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to save chat export: {e}")

    def _browse_folder(self):
        f = QFileDialog.getExistingDirectory(self, "Select folder to index")
        if f:
            self.folder_edit.setText(f)

    def start_indexing(self):
        folder = self.folder_edit.text().strip()
        if not folder or not Path(folder).is_dir():
            QMessageBox.warning(self, "Invalid Folder", "Please select a valid folder to index.")
            return

        self.index_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)
        self.index_status.setText("Scanning documents and preparing embeddings with Ollama...")

        self.index_worker = IndexWorker(self.engine, folder)
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
        self.index_status.setText(f"✓ Ready: Indexed {count} searchable chunk(s). Ask any question below!")
        self._append_system_msg(f"✅ Indexed {count} document chunks from <i>{self.engine.indexed_folder}</i>.")

    def _on_index_failed(self, err: str):
        self.progress_bar.hide()
        self.index_btn.setEnabled(True)
        self.index_status.setText(f"Indexing failed: {err}")
        QMessageBox.critical(self, "Indexing Error", f"Failed to index documents: {err}")

    def send_question(self):
        query = self.query_edit.text().strip()
        if not query:
            return

        self.query_edit.clear()
        self._append_user_msg(query)
        self.history.append({"role": "user", "content": query})

        self.send_btn.setEnabled(False)
        self.query_edit.setEnabled(False)
        self.current_assistant_text = ""

        # Prepare assistant bubble
        self.chat_browser.append(
            '<div style="margin: 8px 0; text-align: left;">'
            '<div id="active_msg" style="background-color: #1e293b; color: #f8fafc; padding: 8px 14px; border-radius: 12px; display: inline-block; max-width: 90%; border: 1px solid #334155;">'
            '<b>Polaris:</b><br><span id="content">thinking...</span></div></div>'
        )

        file_types = self._get_selected_file_types()
        self.query_worker = StreamQueryWorker(self.engine, query, self.history, file_types=file_types)
        self.query_worker.token.connect(self._on_token)
        self.query_worker.done.connect(self._on_stream_done)
        self.query_worker.failed.connect(self._on_stream_failed)
        self.query_worker.start()

    def _on_token(self, token: str):
        if not self.current_assistant_text:
            # First token arrives! Clear "thinking..."
            self.current_assistant_text = token
        else:
            self.current_assistant_text += token

        # Re-render active message cleanly
        cursor = self.chat_browser.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.chat_browser.setTextCursor(cursor)
        self._refresh_latest_assistant_bubble(self.current_assistant_text, cited=[])

    def _refresh_latest_assistant_bubble(self, text: str, cited: list):
        formatted_text = text.replace("\n", "<br>")
        citations_html = ""
        if cited:
            unique_sources = {}
            for c in cited:
                src_name = c.file_name
                pages = unique_sources.setdefault(src_name, set())
                pages.add(c.page)

            cite_items = [
                f"📄 <b>{name}</b> (Page {', '.join(str(p) for p in sorted(pages))})"
                for name, pages in unique_sources.items()
            ]
            citations_html = (
                f'<div style="margin-top: 8px; font-size: 11px; color: #94a3b8; border-top: 1px solid #334155; padding-top: 4px;">'
                f'<b>Sources:</b><br>{"<br>".join(cite_items)}</div>'
            )

        html = f"""
        <div style="margin: 8px 0; text-align: left;">
            <div style="background-color: #1e293b; color: #f8fafc; padding: 8px 14px; border-radius: 12px; display: inline-block; max-width: 90%; border: 1px solid #334155;">
                <b>Polaris:</b><br>{formatted_text}
                {citations_html}
            </div>
        </div>
        """
        # Remove the previous placeholder bubble and insert formatted HTML
        doc = self.chat_browser.document()
        block = doc.lastBlock()
        cursor = self.chat_browser.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.select(cursor.SelectionType.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.insertHtml(html)

    def _on_stream_done(self, cited: list):
        self.send_btn.setEnabled(True)
        self.query_edit.setEnabled(True)
        self.query_edit.setFocus()

        self.last_cited_chunks = cited
        if cited:
            self.inspect_sources_btn.setEnabled(True)

        self._refresh_latest_assistant_bubble(self.current_assistant_text, cited=cited)

        # Update chat history
        self.history.append({"role": "assistant", "content": self.current_assistant_text, "cited": cited})

    def _on_stream_failed(self, err: str):
        self.send_btn.setEnabled(True)
        self.query_edit.setEnabled(True)
        self._append_system_msg(f"⚠️ Error: {err}")

    def _append_user_msg(self, text: str):
        html = f"""
        <div style="margin: 8px 0; text-align: right;">
            <span style="background-color: #3b82f6; color: white; padding: 6px 12px; border-radius: 12px; display: inline-block;">
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
        self.chat_browser.clear()
        self._append_system_msg("Chat history cleared.")
