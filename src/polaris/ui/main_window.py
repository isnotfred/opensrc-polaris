"""Main window: Focused 3-pillar local AI file assistant powered by Ollama."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings
from ..ai.ollama_client import is_available
from .ai_organize_tab import AIOrganizeTab
from .chat_tab import ChatTab
from .summarize_tab import SummarizeTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = Settings()
        self.setWindowTitle("Polaris - Local AI File Assistant")
        self.resize(1120, 720)

        # Central tab container
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.organize_tab = AIOrganizeTab(self.settings)
        self.chat_tab = ChatTab(self.settings)
        self.summarize_tab = SummarizeTab(self.settings)

        self.tabs.addTab(self.organize_tab, "📁 AI Organize")
        self.tabs.addTab(self.chat_tab, "💬 Search & Chat")
        self.tabs.addTab(self.summarize_tab, "📝 Summarize")

        # Top-right corner widget: Model selector
        model_widget = QWidget()
        mw_layout = QHBoxLayout(model_widget)
        mw_layout.setContentsMargins(0, 0, 8, 0)
        mw_layout.addWidget(QLabel("🧠 Model:"))

        self.model_combo = QComboBox()
        self.model_combo.addItem("qwen2.5:1.5b (Fast & Low RAM)", "qwen2.5:1.5b")
        self.model_combo.addItem("llama3.2:3b", "llama3.2:3b")

        # Set default selection based on current settings
        idx = self.model_combo.findData(self.settings.chat_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        else:
            self.model_combo.addItem(self.settings.chat_model, self.settings.chat_model)
            self.model_combo.setCurrentIndex(self.model_combo.count() - 1)

        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        mw_layout.addWidget(self.model_combo)

        self.tabs.setCornerWidget(model_widget, Qt.Corner.TopRightCorner)
        self.setCentralWidget(self.tabs)

        # Status Bar with Ollama connection check
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self._update_status_bar()

    def _on_model_changed(self):
        new_model = self.model_combo.currentData()
        if new_model:
            self.settings.chat_model = new_model
            self._update_status_bar()

    def _update_status_bar(self):
        ollama_ok = is_available(self.settings)
        if ollama_ok:
            status_text = (
                f"🟢 Ollama Local AI Connected  |  Active LLM: {self.settings.chat_model}  |  "
                f"Embeddings: {self.settings.embed_model}  |  100% Private"
            )
        else:
            status_text = (
                f"🔴 Ollama Not Detected at {self.settings.ollama_url}  |  "
                "Please run 'ollama serve' in terminal"
            )

        self.status_bar.showMessage(status_text)
