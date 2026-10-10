"""Main window: Focused 3-pillar local AI file assistant powered by Ollama."""
from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QWidget,
)

from ..config import Settings
from ..ai.ollama_client import is_available
from .ai_organize_tab import AIOrganizeTab
from .chat_tab import ChatTab
from .summarize_tab import SummarizeTab
from .common.styles import (
    ACCENT_DANGER,
    ACCENT_SUCCESS,
    BG_CARD,
    BG_HOVER,
    BORDER_SUBTLE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    get_app_icon,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = Settings()
        self.setWindowTitle("Polaris")
        self.resize(1140, 740)
        self.setWindowIcon(get_app_icon())

        # Central tab container
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.organize_tab = AIOrganizeTab(self.settings)
        self.chat_tab = ChatTab(self.settings)
        self.summarize_tab = SummarizeTab(self.settings)

        self.tabs.addTab(self.organize_tab, "Organize")
        self.tabs.addTab(self.chat_tab, "Search & Chat")
        self.tabs.addTab(self.summarize_tab, "Summarize")

        # Top-right corner widget: Model selector capsule
        model_widget = QWidget()
        mw_layout = QHBoxLayout(model_widget)
        mw_layout.setContentsMargins(0, 4, 12, 4)
        mw_layout.setSpacing(6)

        model_label = QLabel("Model:")
        model_label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 500; "
            f"background: transparent; border: none;"
        )
        mw_layout.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.addItem("qwen2.5:1.5b", "qwen2.5:1.5b")
        self.model_combo.addItem("llama3.2:3b", "llama3.2:3b")
        self.model_combo.setStyleSheet(
            f"font-size: 11px; min-width: 130px; "
            f"background-color: {BG_HOVER}; border: 1px solid {BORDER_SUBTLE}; "
            f"border-radius: 6px; padding: 2px 8px;"
        )

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

        # Status Bar with dedicated permanent badges
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        self._conn_lbl = QLabel()
        self._model_lbl = QLabel()
        self._embed_lbl = QLabel()
        self._privacy_lbl = QLabel("Local & Private")
        self._privacy_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; padding: 0 8px;")

        self.status_bar.addWidget(self._conn_lbl, 1)
        self.status_bar.addPermanentWidget(self._model_lbl)
        self.status_bar.addPermanentWidget(self._embed_lbl)
        self.status_bar.addPermanentWidget(self._privacy_lbl)

        self._update_status_bar()

    def _on_model_changed(self):
        new_model = self.model_combo.currentData()
        if new_model:
            self.settings.chat_model = new_model
            self._update_status_bar()

    def _update_status_bar(self):
        ollama_ok = is_available(self.settings)
        if ollama_ok:
            self._conn_lbl.setText("●  Ollama Connected")
            self._conn_lbl.setStyleSheet(f"color: {ACCENT_SUCCESS}; font-size: 11px; font-weight: 500;")
            self._model_lbl.setText(f"Chat: {self.settings.chat_model}")
            self._model_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; padding: 0 8px;")
            self._embed_lbl.setText(f"Embed: {self.settings.embed_model}")
            self._embed_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; padding: 0 8px;")
        else:
            self._conn_lbl.setText(f"●  Disconnected at {self.settings.ollama_url} (run 'ollama serve')")
            self._conn_lbl.setStyleSheet(f"color: {ACCENT_DANGER}; font-size: 11px; font-weight: 500;")
            self._model_lbl.setText("")
            self._embed_lbl.setText("")
