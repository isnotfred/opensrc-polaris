"""AI-Powered Search & Document Chatbot tab with grounded citations and real-time streaming."""
from __future__ import annotations

import html
import re
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QFileDialog,
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
from .common.card import CardFrame
from .common.components import ActionChip
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


def _format_bubble_text(text: str) -> str:
    """Format markdown-style text into clean HTML for chat bubbles."""
    if not text:
        return ""

    escaped = html.escape(text)

    # Multi-line code blocks
    def _code_block_sub(match):
        code = match.group(1).strip()
        return (
            f'<div style="background-color: #0d1117; border: 1px solid #30363d; padding: 8px 12px; '
            f'border-radius: 6px; color: #e2e8f0; font-family: Consolas, monospace; font-size: 12px; '
            f'margin: 6px 0; white-space: pre-wrap;">{code}</div>'
        )

    escaped = re.sub(r'```(?:[a-zA-Z0-9_-]*\n)?([\s\S]*?)```', _code_block_sub, escaped)

    # Inline code
    escaped = re.sub(
        r'`([^`]+)`',
        r'<code style="background-color: #0d1117; padding: 2px 5px; border-radius: 4px; color: #58a6ff; font-family: Consolas, monospace; font-size: 12px;">\1</code>',
        escaped,
    )

    # Bold: **text**
    escaped = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', escaped)

    # Italic: *text*
    escaped = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', escaped)

    # Format lists and paragraphs
    lines = escaped.split("\n")
    chunks = []
    current_list_type = None
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
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            gen = self.engine.chat_with_docs_stream(self.question, self.history)
            cited = []
            try:
                while not self._stopped:
                    t = next(gen)
                    self.token.emit(t)
            except StopIteration as e:
                cited = e.value or []
            self.done.emit(cited)
        except Exception as e:
            if not self._stopped:
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
        self.current_question: str = ""
        self.active_bubble_start_pos: int = 0
        self.message_count: int = 0

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        # ── Card 1: Document Source & Status ──────────────────────────────────
        self.source_card = CardFrame(
            "Document Source",
            subtitle="Select directory containing PDFs, Word documents, text, or source code",
        )

        folder_row = QHBoxLayout()
        folder_row.setSpacing(8)
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Select directory to index...")
        self.folder_edit.textChanged.connect(self._on_folder_changed)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_folder)

        # Button preserves exact label and style tokens required by tests
        self.index_btn = QPushButton("⚡ Index Folder for AI Search")
        self.index_btn.setStyleSheet(
            "font-weight: 600; padding: 6px 14px; background-color: #21262d; color: #8b949e; "
            "border: 1px solid #30363d; border-radius: 6px;"
        )
        self.index_btn.clicked.connect(self.start_indexing)

        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(browse_btn)
        folder_row.addWidget(self.index_btn)
        self.source_card.content_layout.addLayout(folder_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        self.source_card.content_layout.addWidget(self.progress_bar)

        self.index_status = QLabel("No folder indexed yet. Index a folder to enable semantic search & grounded Q&A.")
        self.index_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.source_card.content_layout.addWidget(self.index_status)

        layout.addWidget(self.source_card)

        # ── Card 2: Document Chat (Conversation Stream) ───────────────────────
        header_actions = QWidget()
        ha_layout = QHBoxLayout(header_actions)
        ha_layout.setContentsMargins(0, 0, 0, 0)
        ha_layout.setSpacing(6)

        self.export_btn = QPushButton("Export")
        self.export_btn.setStyleSheet(ghost_button_style())
        self.export_btn.setToolTip("Export this chat transcript to a Markdown/Text file")
        self.export_btn.clicked.connect(self.export_chat)
        ha_layout.addWidget(self.export_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet(ghost_button_style())
        self.clear_btn.setToolTip("Clear session history and start fresh")
        self.clear_btn.clicked.connect(self.clear_chat)
        ha_layout.addWidget(self.clear_btn)

        self.chat_card = CardFrame(
            "Polaris Document Chat",
            subtitle="Answers are grounded in local files with verifiable page citations",
            header_action=header_actions,
        )

        self.chat_browser = QTextBrowser()
        self.chat_browser.setOpenExternalLinks(False)
        self.chat_browser.setStyleSheet(
            f"background-color: {BG_INNER}; color: {TEXT_PRIMARY}; "
            f"font-family: 'Segoe UI', system-ui, sans-serif; "
            f"font-size: 13px; padding: 12px; border: 1px solid {BORDER_SUBTLE}; border-radius: 6px;"
        )
        self.chat_card.content_layout.addWidget(self.chat_browser, 1)

        # Suggested Prompt Starter Chips
        self.starters_container = QWidget()
        sc_layout = QHBoxLayout(self.starters_container)
        sc_layout.setContentsMargins(0, 0, 0, 0)
        sc_layout.setSpacing(6)

        starters_lbl = QLabel("Suggested:")
        starters_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 500;")
        sc_layout.addWidget(starters_lbl)

        s1 = ActionChip("What was the total revenue in the report?")
        s1.clicked.connect(lambda: self._set_starter_prompt("What was the total revenue in the report?"))
        sc_layout.addWidget(s1)

        s2 = ActionChip("Summarize deliverables and deadlines")
        s2.clicked.connect(lambda: self._set_starter_prompt("Summarize the key deliverables and deadlines"))
        sc_layout.addWidget(s2)

        s3 = ActionChip("List mentioned invoices & figures")
        s3.clicked.connect(lambda: self._set_starter_prompt("List all mentioned invoices, amounts, and figures"))
        sc_layout.addWidget(s3)

        sc_layout.addStretch()
        self.chat_card.content_layout.addWidget(self.starters_container)

        # Input Row
        input_container = QWidget()
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(0, 2, 0, 0)
        input_layout.setSpacing(8)

        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText("Ask a question about your documents... (Enter to send)")
        self.query_edit.returnPressed.connect(self.send_question)
        input_layout.addWidget(self.query_edit, 1)

        self.send_btn = QPushButton("Send")
        self.send_btn.setStyleSheet(primary_button_style())
        self.send_btn.clicked.connect(self.send_question)
        input_layout.addWidget(self.send_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(danger_button_style())
        self.stop_btn.clicked.connect(self.stop_streaming)
        self.stop_btn.hide()
        input_layout.addWidget(self.stop_btn)

        self.chat_card.content_layout.addWidget(input_container)
        layout.addWidget(self.chat_card, 1)

        self._show_welcome_banner()

    def _set_starter_prompt(self, text: str):
        self.query_edit.setText(text)
        self.query_edit.setFocus()

    def _show_welcome_banner(self):
        self.chat_browser.clear()
        time_str = datetime.now().strftime("%I:%M %p")
        html = f"""
        <div align="center" style="margin: 16px 0 20px 0;">
            <table cellpadding="14" cellspacing="0" style="background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; max-width: 600px;">
                <tr>
                    <td align="center">
                        <div style="font-size: 14px; font-weight: 600; color: #58a6ff; margin-bottom: 4px;">Polaris Document Chat</div>
                        <div style="font-size: 12px; color: #8b949e; line-height: 1.4;">
                            Select a folder above, click <b>Index Folder</b>, and ask questions grounded directly in your files.
                        </div>
                        <div style="margin-top: 8px; font-size: 10px; color: #6e7681;">
                            Session initialized at {time_str} · FAISS Local Vector Store
                        </div>
                    </td>
                </tr>
            </table>
        </div>
        """
        self.chat_browser.append(html)
        self._scroll_to_bottom()

    def _on_folder_changed(self, text: str):
        folder = text.strip()
        if folder and Path(folder).is_dir():
            # Glowing green state required by tests
            self.index_btn.setStyleSheet(
                "font-weight: bold; font-size: 13px; color: #ffffff; background-color: #16a34a; "
                "border: 2px solid #4ade80; border-radius: 6px; padding: 6px 16px;"
            )
        else:
            self.index_btn.setStyleSheet(
                "font-weight: 600; padding: 6px 14px; background-color: #21262d; color: #8b949e; "
                "border: 1px solid #30363d; border-radius: 6px;"
            )

    def _scroll_to_bottom(self):
        sb = self.chat_browser.verticalScrollBar()
        sb.setValue(sb.maximum())

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
        self.index_btn.setStyleSheet(
            "font-weight: 600; color: #ffffff; background-color: #238636; "
            "border: 1px solid #2ea043; border-radius: 6px; padding: 6px 14px;"
        )
        self.index_status.setText(f"●  Ready: Indexed {count} searchable chunk(s). Ask any question below!")
        self._append_system_msg(f"Indexed {count} document chunks from <i>{self.engine.indexed_folder}</i>.")

    def _on_index_failed(self, err: str):
        self.progress_bar.hide()
        self.index_btn.setEnabled(True)
        self._on_folder_changed(self.folder_edit.text())
        self.index_status.setText(f"Indexing failed: {err}")
        QMessageBox.critical(self, "Indexing Error", f"Failed to index documents: {err}")

    def send_question(self):
        query = self.query_edit.text().strip()
        if not query:
            return

        self.query_edit.clear()
        self.current_question = query
        self.current_assistant_text = ""
        now_str = datetime.now().strftime("%I:%M %p")

        # 1. Append User Bubble (right-aligned, compact width)
        self._append_user_bubble(query, now_str)

        # 2. Append Assistant Bubble Placeholder (left-aligned)
        self._insert_assistant_bubble_placeholder(now_str)

        # 3. Update UI states
        self.send_btn.hide()
        self.stop_btn.show()
        self.query_edit.setEnabled(False)
        self._scroll_to_bottom()

        # 4. Launch streaming query with multi-turn history context
        self.query_worker = StreamQueryWorker(self.engine, query, self.history)
        self.query_worker.token.connect(self._on_token)
        self.query_worker.done.connect(self._on_stream_done)
        self.query_worker.failed.connect(self._on_stream_failed)
        self.query_worker.start()

    def stop_streaming(self):
        if self.query_worker and self.query_worker.isRunning():
            self.query_worker.stop()

    def _append_user_bubble(self, text: str, time_str: str):
        formatted = _format_bubble_text(text)
        # Using inner table align="right" ensures Qt sizes the bubble tightly to the text instead of 100% width!
        html = f"""
        <table width="100%" border="0" cellpadding="0" cellspacing="0" style="margin: 8px 0;">
            <tr>
                <td align="right">
                    <table border="0" cellpadding="8" cellspacing="0" style="background-color: #2563eb; border-radius: 12px; margin-left: 80px;">
                        <tr>
                            <td style="color: #ffffff; font-size: 13px; font-family: 'Segoe UI', sans-serif; line-height: 1.4;">
                                {formatted}
                            </td>
                        </tr>
                    </table>
                    <div style="color: #6e7681; font-size: 10px; margin-top: 2px; margin-right: 4px;">{time_str}</div>
                </td>
            </tr>
        </table>
        """
        self.chat_browser.append(html)
        self.message_count += 1
        self._scroll_to_bottom()

    def _insert_assistant_bubble_placeholder(self, time_str: str):
        cursor = self.chat_browser.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertBlock()
        self.active_bubble_start_pos = cursor.position()

        html = f"""
        <table width="100%" border="0" cellpadding="0" cellspacing="0" style="margin: 8px 0;">
            <tr>
                <td align="left">
                    <div style="color: #58a6ff; font-size: 11px; font-weight: 600; margin-bottom: 3px; margin-left: 2px;">Polaris AI</div>
                    <table border="0" cellpadding="10" cellspacing="0" style="background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; margin-right: 80px;">
                        <tr>
                            <td style="color: #8b949e; font-size: 13px; font-family: 'Segoe UI', sans-serif;">
                                <i>Thinking...</i>
                            </td>
                        </tr>
                    </table>
                    <div style="color: #6e7681; font-size: 10px; margin-top: 2px; margin-left: 2px;">{time_str}</div>
                </td>
            </tr>
        </table>
        """
        cursor.insertHtml(html)
        self.chat_browser.setTextCursor(cursor)
        self.message_count += 1
        self._scroll_to_bottom()

    def _on_token(self, token: str):
        self.current_assistant_text += token
        self._update_active_assistant_bubble(self.current_assistant_text, cited=[], is_final=False)

    def _update_active_assistant_bubble(self, text: str, cited: list, is_final: bool):
        now_str = datetime.now().strftime("%I:%M %p")
        formatted = _format_bubble_text(text) if text else "<i>Thinking...</i>"

        citations_html = ""
        if is_final and cited:
            unique_sources = {}
            for c in cited:
                src_name = Path(c.doc_path).name
                pages = unique_sources.setdefault(src_name, set())
                pages.add(c.page)

            cite_items = [
                f'<span style="background-color: #21262d; border: 1px solid #30363d; border-radius: 8px; padding: 2px 7px; color: #58a6ff; font-size: 11px; margin-right: 6px;">'
                f'{name} · p. {", ".join(str(p) for p in sorted(pages))}</span>'
                for name, pages in unique_sources.items()
            ]
            citations_html = (
                f'<div style="margin-top: 8px; font-size: 11px; color: #8b949e; border-top: 1px solid #30363d; padding-top: 6px;">'
                f'<div style="color: #c9d1d9; font-weight: 600; margin-bottom: 4px;">Sources Cited:</div>'
                f'{" ".join(cite_items)}</div>'
            )

        html = f"""
        <table width="100%" border="0" cellpadding="0" cellspacing="0" style="margin: 8px 0;">
            <tr>
                <td align="left">
                    <div style="color: #58a6ff; font-size: 11px; font-weight: 600; margin-bottom: 3px; margin-left: 2px;">Polaris AI</div>
                    <table border="0" cellpadding="10" cellspacing="0" style="background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; margin-right: 80px;">
                        <tr>
                            <td style="color: #f0f6fc; font-size: 13px; font-family: 'Segoe UI', sans-serif; line-height: 1.45;">
                                {formatted}
                                {citations_html}
                            </td>
                        </tr>
                    </table>
                    <div style="color: #6e7681; font-size: 10px; margin-top: 2px; margin-left: 2px;">{now_str}</div>
                </td>
            </tr>
        </table>
        """

        # Update in-place without touching any previous history messages
        cursor = self.chat_browser.textCursor()
        cursor.setPosition(self.active_bubble_start_pos)
        cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertHtml(html)
        self.chat_browser.setTextCursor(cursor)
        self._scroll_to_bottom()

    def _on_stream_done(self, cited: list):
        self.stop_btn.hide()
        self.send_btn.show()
        self.query_edit.setEnabled(True)
        self.query_edit.setFocus()

        # Finalize the assistant bubble with citations
        self._update_active_assistant_bubble(self.current_assistant_text, cited=cited, is_final=True)

        # Record turn in multi-turn conversation history
        self.history.append({"role": "user", "content": self.current_question})
        self.history.append({"role": "assistant", "content": self.current_assistant_text})
        self._scroll_to_bottom()

    def _on_stream_failed(self, err: str):
        self.stop_btn.hide()
        self.send_btn.show()
        self.query_edit.setEnabled(True)
        self._append_system_msg(f"Error: {err}")
        self._scroll_to_bottom()

    def _append_system_msg(self, text: str):
        html = f"""
        <div align="center" style="margin: 8px 0;">
            <span style="background-color: #161b22; color: #8b949e; font-size: 11px; padding: 4px 12px; border-radius: 12px; border: 1px solid #30363d;">
                {text}
            </span>
        </div>
        """
        self.chat_browser.append(html)
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
        self.message_count = 0
        self._show_welcome_banner()

    def export_chat(self):
        if not self.history:
            QMessageBox.information(self, "No Chat History", "There is no chat history in this session to export.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Chat History",
            f"polaris_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            "Markdown Files (*.md);;Text Files (*.txt)",
        )
        if not save_path:
            return

        try:
            lines = [
                f"# Polaris Document Chat Transcript",
                f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Source Folder:** {self.engine.indexed_folder or 'None'}",
                f"---",
                "",
            ]
            for msg in self.history:
                role = "User" if msg["role"] == "user" else "Polaris AI"
                lines.append(f"### {role}")
                lines.append(msg["content"])
                lines.append("")

            Path(save_path).write_text("\n".join(lines), encoding="utf-8")
            QMessageBox.information(self, "Export Successful", f"Chat conversation exported successfully to:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to save chat export: {e}")
