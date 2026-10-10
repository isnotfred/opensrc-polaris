"""Reusable micro-components: StatusBadge, SegmentedControl, ActionChip."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from .styles import (
    ACCENT_PRIMARY,
    ACCENT_SUCCESS,
    BG_ACTIVE,
    BG_HOVER,
    BG_INNER,
    BORDER_MEDIUM,
    BORDER_SUBTLE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class StatusBadge(QLabel):
    """A compact pill-shaped status indicator with a colored dot and text."""

    def __init__(self, text: str = "", color: str = TEXT_MUTED, parent: QWidget | None = None):
        super().__init__(parent)
        self._color = color
        self._text = text
        self._render()

    def update_status(self, text: str, color: str | None = None):
        self._text = text
        if color:
            self._color = color
        self._render()

    def _render(self):
        self.setText(f"●  {self._text}")
        self.setStyleSheet(
            f"color: {self._color}; font-size: 11px; font-weight: 500; "
            f"background: transparent; border: none; padding: 0 4px;"
        )


class SegmentedControl(QFrame):
    """A sleek, compact segmented toggle control with pixel-perfect alignment."""

    selection_changed = Signal(int)

    def __init__(
        self,
        options: list[str],
        default: int = 0,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("segmentedControl")
        self.setFixedHeight(28)
        self.setStyleSheet(
            f"QFrame#segmentedControl {{"
            f"  background-color: {BG_INNER};"
            f"  border: 1px solid {BORDER_SUBTLE};"
            f"  border-radius: 6px;"
            f"}}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self._buttons: list[QPushButton] = []
        self._selected_index = default

        for i, label in enumerate(options):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(22)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, idx=i: self._on_clicked(idx))
            self._buttons.append(btn)
            layout.addWidget(btn)

        self._update_styles()

    def _on_clicked(self, index: int):
        if index == self._selected_index:
            self._buttons[index].setChecked(True)
            return
        self._selected_index = index
        self._update_styles()
        self.selection_changed.emit(index)

    def _update_styles(self):
        for i, btn in enumerate(self._buttons):
            if i == self._selected_index:
                btn.setChecked(True)
                btn.setStyleSheet(
                    f"background-color: {ACCENT_PRIMARY}; color: #ffffff; "
                    f"font-weight: 600; font-size: 11px; "
                    f"border: none; border-radius: 4px; padding: 2px 10px;"
                )
            else:
                btn.setChecked(False)
                btn.setStyleSheet(
                    f"background-color: transparent; color: {TEXT_SECONDARY}; "
                    f"font-weight: 500; font-size: 11px; "
                    f"border: none; border-radius: 4px; padding: 2px 10px;"
                )

    def selected_index(self) -> int:
        return self._selected_index

    def set_selected(self, index: int):
        if 0 <= index < len(self._buttons):
            self._selected_index = index
            self._update_styles()


class ActionChip(QPushButton):
    """A compact, clickable pill-shaped chip for quick presets."""

    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(24)
        self.setStyleSheet(
            f"background-color: {BG_HOVER}; color: {TEXT_SECONDARY}; "
            f"border: 1px solid {BORDER_SUBTLE}; border-radius: 12px; "
            f"padding: 2px 10px; font-size: 11px; font-weight: 500;"
        )

    def enterEvent(self, event):
        self.setStyleSheet(
            f"background-color: {BG_ACTIVE}; color: {TEXT_PRIMARY}; "
            f"border: 1px solid {BORDER_MEDIUM}; border-radius: 12px; "
            f"padding: 2px 10px; font-size: 11px; font-weight: 500;"
        )
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet(
            f"background-color: {BG_HOVER}; color: {TEXT_SECONDARY}; "
            f"border: 1px solid {BORDER_SUBTLE}; border-radius: 12px; "
            f"padding: 2px 10px; font-size: 11px; font-weight: 500;"
        )
        super().leaveEvent(event)
