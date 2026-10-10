"""Polaris Design System: Obsidian dark palette, refined typography, and master QSS stylesheet."""
from __future__ import annotations

# ── Color Palette (Modern Obsidian & Slate) ──────────────────────────────────

# Backgrounds & Surfaces
BG_APP = "#0d1117"        # Obsidian dark canvas
BG_CARD = "#161b22"       # Elevated card surface
BG_INNER = "#0b0f14"      # Inner sunken containers, code blocks, chat viewer
BG_HOVER = "#21262d"      # Subtle interactive hover
BG_ACTIVE = "#2d333b"     # Pressed / active state

# Borders
BORDER_SUBTLE = "#30363d" # Crisp card and separator borders
BORDER_MEDIUM = "#484f58" # Interactive element borders
BORDER_FOCUS = "#388bfd"  # Focus highlight ring

# Text
TEXT_PRIMARY = "#f0f6fc"   # High-contrast headers, input values
TEXT_SECONDARY = "#8b949e" # Labels, subtitles, chip text
TEXT_MUTED = "#6e7681"     # Placeholders, timestamps, hints

# Semantic Accents
ACCENT_PRIMARY = "#3b82f6"       # Electric Blue CTA
ACCENT_PRIMARY_HOVER = "#2563eb"
ACCENT_SUCCESS = "#3fb950"       # Emerald connected / success
ACCENT_WARNING = "#d29922"       # Amber warning
ACCENT_DANGER = "#f85149"        # Rose destructive / stop


# ── Master Application Stylesheet ────────────────────────────────────────────

MASTER_QSS = f"""
/* ── Global Defaults ───────────────────────────────────────────── */
QWidget {{
    background-color: {BG_APP};
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", -apple-system, system-ui, sans-serif;
}}

/* ── Modern Pill Tab Bar ──────────────────────────────────────── */
QTabWidget::pane {{
    border: none;
    background-color: {BG_APP};
}}

QTabBar {{
    background-color: {BG_APP};
    qproperty-drawBase: 0;
}}

QTabBar::tab {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    padding: 7px 18px;
    margin: 4px 6px 4px 0;
    border-radius: 6px;
    border: 1px solid transparent;
    font-weight: 500;
    font-size: 13px;
}}

QTabBar::tab:selected {{
    color: {TEXT_PRIMARY};
    background-color: {BG_HOVER};
    border: 1px solid {BORDER_SUBTLE};
    font-weight: 600;
}}

QTabBar::tab:hover:!selected {{
    color: {TEXT_PRIMARY};
    background-color: rgba(255, 255, 255, 0.04);
}}

/* ── Inputs ───────────────────────────────────────────────────── */
QLineEdit {{
    background-color: {BG_INNER};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    padding: 7px 12px;
    font-size: 12px;
    selection-background-color: {ACCENT_PRIMARY};
}}

QLineEdit:focus {{
    border: 1px solid {BORDER_FOCUS};
    background-color: {BG_APP};
}}

QLineEdit::placeholder {{
    color: {TEXT_MUTED};
}}

/* ── Combo Boxes ──────────────────────────────────────────────── */
QComboBox {{
    background-color: {BG_HOVER};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 12px;
    min-height: 22px;
}}

QComboBox:hover {{
    border: 1px solid {BORDER_MEDIUM};
    background-color: {BG_ACTIVE};
}}

QComboBox:focus {{
    border: 1px solid {BORDER_FOCUS};
}}

QComboBox::drop-down {{
    border: none;
    width: 22px;
}}

QComboBox QAbstractItemView {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    selection-background-color: {BG_HOVER};
    selection-color: {TEXT_PRIMARY};
    outline: none;
    padding: 4px;
}}

/* ── Base Push Buttons ────────────────────────────────────────── */
QPushButton {{
    background-color: {BG_HOVER};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: {BG_ACTIVE};
    border: 1px solid {BORDER_MEDIUM};
}}

QPushButton:pressed {{
    background-color: #21262d;
}}

QPushButton:disabled {{
    background-color: {BG_APP};
    color: {TEXT_MUTED};
    border: 1px solid #21262d;
}}

/* ── Checkboxes ───────────────────────────────────────────────── */
QCheckBox {{
    color: {TEXT_SECONDARY};
    spacing: 7px;
    font-size: 12px;
}}

QCheckBox:hover {{
    color: {TEXT_PRIMARY};
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER_MEDIUM};
    border-radius: 4px;
    background-color: {BG_INNER};
}}

QCheckBox::indicator:hover {{
    border: 1px solid {BORDER_FOCUS};
}}

QCheckBox::indicator:checked {{
    background-color: {ACCENT_PRIMARY};
    border: 1px solid {ACCENT_PRIMARY};
}}

/* ── Tables ───────────────────────────────────────────────────── */
QTableWidget {{
    background-color: {BG_INNER};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    gridline-color: #1e2530;
    font-size: 12px;
}}

QTableWidget::item {{
    padding: 6px 10px;
    border-bottom: 1px solid #1e2530;
}}

QTableWidget::item:selected {{
    background-color: #1f293d;
    color: {TEXT_PRIMARY};
}}

QHeaderView::section {{
    background-color: {BG_CARD};
    color: {TEXT_SECONDARY};
    border: none;
    border-bottom: 1px solid {BORDER_SUBTLE};
    padding: 7px 10px;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* ── Scroll Bars ──────────────────────────────────────────────── */
QScrollBar:vertical {{
    background-color: transparent;
    width: 8px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background-color: {BORDER_SUBTLE};
    min-height: 24px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {BORDER_MEDIUM};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: transparent;
    height: 8px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background-color: {BORDER_SUBTLE};
    min-width: 24px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {BORDER_MEDIUM};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── Progress Bars ────────────────────────────────────────────── */
QProgressBar {{
    background-color: {BG_INNER};
    border: none;
    border-radius: 2px;
    text-align: center;
    color: transparent;
    max-height: 4px;
}}

QProgressBar::chunk {{
    background-color: {ACCENT_PRIMARY};
    border-radius: 2px;
}}

/* ── Text Browser ─────────────────────────────────────────────── */
QTextBrowser {{
    background-color: {BG_INNER};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    padding: 12px;
    selection-background-color: {ACCENT_PRIMARY};
}}

/* ── Status Bar ───────────────────────────────────────────────── */
QStatusBar {{
    background-color: {BG_CARD};
    color: {TEXT_SECONDARY};
    border-top: 1px solid {BORDER_SUBTLE};
    font-size: 11px;
    padding: 3px 12px;
}}

/* ── List Widget ──────────────────────────────────────────────── */
QListWidget {{
    background-color: {BG_INNER};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    font-size: 12px;
    padding: 4px;
}}

QListWidget::item {{
    padding: 5px 8px;
    border-radius: 4px;
}}

QListWidget::item:selected {{
    background-color: #1f293d;
    color: {TEXT_PRIMARY};
}}

/* ── Tool Tips ────────────────────────────────────────────────── */
QToolTip {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_MEDIUM};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11px;
}}

/* ── Message Box ──────────────────────────────────────────────── */
QMessageBox {{
    background-color: {BG_CARD};
}}

QMessageBox QLabel {{
    color: {TEXT_PRIMARY};
    font-size: 12px;
}}
"""


# ── Button Style Helpers ─────────────────────────────────────────────────────

def primary_button_style() -> str:
    """Crisp high-priority primary CTA button."""
    return (
        f"font-weight: 600; font-size: 12px; "
        f"background-color: {ACCENT_PRIMARY}; color: #ffffff; "
        f"padding: 7px 16px; border-radius: 6px; border: 1px solid {BORDER_FOCUS};"
    )


def success_button_style() -> str:
    """Action button for apply / execute operations."""
    return (
        f"font-weight: 600; font-size: 12px; "
        f"background-color: #238636; color: #ffffff; "
        f"padding: 7px 16px; border-radius: 6px; border: 1px solid #2ea043;"
    )


def ghost_button_style() -> str:
    """Minimalist utility button with subtle border."""
    return (
        f"font-weight: 500; font-size: 11px; "
        f"background-color: transparent; color: {TEXT_SECONDARY}; "
        f"padding: 5px 12px; border-radius: 6px; border: 1px solid {BORDER_SUBTLE};"
    )


def danger_button_style() -> str:
    """Destructive / stop streaming button."""
    return (
        f"font-weight: 600; font-size: 11px; "
        f"background-color: #da3633; color: #ffffff; "
        f"padding: 6px 14px; border-radius: 6px; border: 1px solid #f85149;"
    )


def get_app_icon():
    """Returns a crisp multi-resolution QIcon for window & taskbar."""
    from pathlib import Path
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QIcon

    icon = QIcon()
    assets = Path(__file__).resolve().parent.parent.parent / "assets"
    for sz in (16, 32, 64, 512):
        f = assets / (f"icon-{sz}.png" if sz != 512 else "icon.png")
        if f.exists():
            icon.addFile(str(f), QSize(sz, sz))
    if not icon.isNull():
        return icon
    svg = assets / "icon.svg"
    if svg.exists():
        return QIcon(str(svg))
    return icon

