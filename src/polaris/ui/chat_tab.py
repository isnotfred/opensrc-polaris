"""AI-Powered Search & Document Chatbot tab with real-time token streaming."""
from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
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
from ..ai.rag import RagEngine


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

    def __init__(self, engine: RagEngine, question: str, history: list[dict]):
        super().__init__()
        self.engine = engine
        self.question = question
        self.history = history

    def run(self):
        try:
            gen = self.engine.chat_with_docs_stream(self.question, self.history)
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

        # 3. Input
        input_row = QHBoxLayout()
        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText("Ask a question about your files... (e.g. 'What are the main findings in the report?')")
        self.query_edit.returnPressed.connect(self.send_question)

        self.send_btn = QPushButton("Ask AI")
        self.send_btn.setStyleSheet("font-weight: bold; background-color: #2563eb; color: white; padding: 6px 16px;")
        self.send_btn.clicked.connect(self.send_question)

        clear_btn = QPushButton("Clear Chat")
        clear_btn.clicked.connect(self.clear_chat)

        input_row.addWidget(self.query_edit, 1)
        input_row.addWidget(self.send_btn)
        input_row.addWidget(clear_btn)
        cg_layout.addLayout(input_row)

        layout.addWidget(chat_group, 1)

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

        self.send_btn.setEnabled(False)
        self.query_edit.setEnabled(False)
        self.current_assistant_text = ""

        # Prepare assistant bubble
        self.chat_browser.append(
            '<div style="margin: 8px 0; text-align: left;">'
            '<div id="active_msg" style="background-color: #1e293b; color: #f8fafc; padding: 8px 14px; border-radius: 12px; display: inline-block; max-width: 90%; border: 1px solid #334155;">'
            '<b>Polaris:</b><br><span id="content">thinking...</span></div></div>'
        )

        self.query_worker = StreamQueryWorker(self.engine, query, self.history)
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
                src_name = Path(c.doc_path).name
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

        self._refresh_latest_assistant_bubble(self.current_assistant_text, cited=cited)

        # Update chat history
        self.history.append({"role": "user", "content": self.history[-1]["content"] if self.history else ""})
        self.history.append({"role": "assistant", "content": self.current_assistant_text})

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
        self.chat_browser.clear()
        self._append_system_msg("Chat history cleared.")
