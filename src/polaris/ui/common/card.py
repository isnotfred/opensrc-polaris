"""CardFrame: Modern elevated card container with a crisp header and flexible layout."""
from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .styles import BG_CARD, BORDER_SUBTLE, TEXT_MUTED, TEXT_PRIMARY


class CardFrame(QFrame):
    """Modern elevated card container with a clean title, optional subtitle,
    and optional header action widget."""

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        header_action: QWidget | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("polarisCard")
        self.setStyleSheet(
            f"QFrame#polarisCard {{"
            f"  background-color: {BG_CARD};"
            f"  border: 1px solid {BORDER_SUBTLE};"
            f"  border-radius: 8px;"
            f"}}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        # ── Header Row ──────────────────────────────────────────────
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)

        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title_box.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(
            f"font-weight: 600; font-size: 13px; color: {TEXT_PRIMARY}; "
            f"background: transparent; border: none;"
        )
        title_box.addWidget(self.title_label)

        if subtitle:
            self.subtitle_label = QLabel(subtitle)
            self.subtitle_label.setStyleSheet(
                f"font-size: 11px; color: {TEXT_MUTED}; "
                f"background: transparent; border: none;"
            )
            title_box.addWidget(self.subtitle_label)
        else:
            self.subtitle_label = None

        header_row.addLayout(title_box)
        header_row.addStretch()

        if header_action:
            header_row.addWidget(header_action)

        root.addLayout(header_row)

        # ── Content Area ────────────────────────────────────────────
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)
        root.addLayout(self.content_layout)
