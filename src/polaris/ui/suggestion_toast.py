"""Floating desktop toast notification for incoming file rename & folder suggestions."""
from __future__ import annotations

import os
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings
from ..core.organizer import Move, apply_moves, unique_destination
from ..db.database import connect


class SuggestionToast(QWidget):
    applied = Signal(str, str)  # src, dst
    dismissed = Signal()

    def __init__(self, suggestion: dict, settings: Settings | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.settings = settings or Settings()
        self.suggestion = suggestion
        self.orig_path = Path(suggestion.get("original_path", ""))

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        self._init_ui()
        self._position_bottom_right()

    def _init_ui(self):
        self.setFixedWidth(420)
        self.setStyleSheet("""
            QWidget {
                background-color: #0f172a;
                color: #f8fafc;
                font-family: Segoe UI, sans-serif;
                border: 2px solid #2563eb;
                border-radius: 10px;
            }
            QLabel {
                border: none;
            }
            QLineEdit {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 5px;
                padding: 5px 8px;
                color: #ffffff;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid #3b82f6;
            }
            QPushButton {
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 12px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        # Header
        head_row = QHBoxLayout()
        icon_label = QLabel("⚡ <b>Polaris Smart Download Watcher</b>")
        icon_label.setStyleSheet("color: #60a5fa; font-size: 13px;")
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(22, 22)
        close_btn.setStyleSheet("""
            QPushButton { background: transparent; border: none; color: #94a3b8; font-size: 12px; }
            QPushButton:hover { color: #f87171; }
        """)
        close_btn.clicked.connect(self.dismiss)

        head_row.addWidget(icon_label, 1)
        head_row.addWidget(close_btn)
        layout.addLayout(head_row)

        # Original file info
        orig_label = QLabel(f"Detected: <b>{self.orig_path.name}</b>")
        orig_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(orig_label)

        # Suggested Name
        layout.addWidget(QLabel("Suggested Filename:"))
        self.name_edit = QLineEdit(self.suggestion.get("suggested_filename", self.orig_path.name))
        layout.addWidget(self.name_edit)

        # Suggested Folder
        layout.addWidget(QLabel("Suggested Destination Folder:"))
        default_folder = self.suggestion.get("suggested_folder", "Organized")
        target_root = self.orig_path.parent
        self.folder_edit = QLineEdit(str(target_root / default_folder))
        layout.addWidget(self.folder_edit)

        # Reason tag
        reason = self.suggestion.get("reason", "Smart placement")
        reason_label = QLabel(f"💡 <i>{reason}</i>")
        reason_label.setStyleSheet("color: #38bdf8; font-size: 11px; margin-top: 2px;")
        layout.addWidget(reason_label)

        # Action Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()

        ignore_btn = QPushButton("Keep In Place")
        ignore_btn.setStyleSheet("background-color: #334155; color: #cbd5e1; border: none;")
        ignore_btn.clicked.connect(self.dismiss)

        apply_btn = QPushButton("Move & Rename")
        apply_btn.setStyleSheet("background-color: #2563eb; color: white; border: none;")
        apply_btn.clicked.connect(self.apply_suggestion)

        btn_row.addWidget(ignore_btn)
        btn_row.addWidget(apply_btn)
        layout.addLayout(btn_row)

    def _position_bottom_right(self):
        screen = QGuiApplication.primaryScreen().availableGeometry()
        x = screen.right() - self.width() - 24
        y = screen.bottom() - 260
        self.move(x, y)

    def apply_suggestion(self):
        new_name = self.name_edit.text().strip()
        folder_str = self.folder_edit.text().strip()

        if not new_name or not folder_str:
            return

        target_dir = Path(folder_str)
        target_path = unique_destination(target_dir / new_name)

        move_op = Move(
            src=str(self.orig_path),
            dst=str(target_path),
            reason=f"Download Watcher: {self.suggestion.get('reason', '')}"
        )

        try:
            conn = connect(self.settings.db_path)
            apply_moves(conn, [move_op], mode="move")
            conn.close()
            self.applied.emit(str(self.orig_path), str(target_path))
        except Exception:
            pass

        self.close()

    def dismiss(self):
        self.dismissed.emit()
        self.close()
