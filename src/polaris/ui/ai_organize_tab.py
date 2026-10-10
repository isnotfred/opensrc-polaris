"""AI-Powered Organize tab: Natural language and intelligent file sorting with Ollama, safe preview, and undo."""
from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings
from ..core.organizer import Move, apply_moves, plan_by_type, undo_batch
from ..ai.ai_organizer import plan_with_ai
from ..db.database import connect
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
    ghost_button_style,
    primary_button_style,
    success_button_style,
)


class AIPlanWorker(QThread):
    done = Signal(list, str, str)  # moves, strategy_name, explanation
    failed = Signal(str)

    def __init__(
        self,
        file_paths: list[str],
        dest_dir: str,
        instruction: str,
        settings: Settings,
        mode: str = "custom",
    ):
        super().__init__()
        self.file_paths = file_paths
        self.dest_dir = dest_dir
        self.instruction = instruction
        self.settings = settings
        self.mode = mode

    def run(self):
        try:
            if self.mode == "rule_based" or self.instruction.strip().lower() == "__rule_based__":
                moves = plan_by_type(self.file_paths, self.dest_dir)
                strategy_name = "Fast Rule-based (by Type)"
                explanation = (
                    "Files were deterministically sorted into standard category folders "
                    "(Documents, Presentations, Spreadsheets, Images, Videos, Audio, Archives, Code) "
                    "based purely on file extensions."
                )
            elif self.mode == "smart_ai" or not self.instruction.strip():
                moves = plan_with_ai(self.file_paths, self.dest_dir, "", self.settings)
                strategy_name = "Smart Auto-Organize (Autonomous AI Logic)"
                explanation = (
                    "Files were analyzed by Ollama and automatically organized into logical subfolders "
                    "based on detected project names, topics, file types, and contents."
                )
            else:
                moves = plan_with_ai(self.file_paths, self.dest_dir, self.instruction, self.settings)
                strategy_name = f'Custom Instructions: "{self.instruction.strip()}"'
                norm = self.instruction.lower().strip()
                if "size" in norm and any(k in norm for k in ["type", "ext", "format", "kind"]):
                    explanation = (
                        f"Files were first categorized by file format/type, then partitioned into size categories "
                        f"(Small under 1MB, Medium 1MB-50MB, Large over 50MB) adhering strictly to your instruction: '{self.instruction.strip()}'."
                    )
                elif "size" in norm:
                    explanation = (
                        f"Files were grouped into size category subfolders "
                        f"(Small under 1MB, Medium 1MB-50MB, Large over 50MB) adhering strictly to your instruction: '{self.instruction.strip()}'."
                    )
                elif "date" in norm or "year" in norm:
                    explanation = (
                        f"Files were grouped chronologically into subfolders by year/date "
                        f"adhering strictly to your instruction: '{self.instruction.strip()}'."
                    )
                elif "client" in norm or "project" in norm:
                    explanation = (
                        f"Files were grouped by detected project or client names "
                        f"adhering strictly to your instruction: '{self.instruction.strip()}'."
                    )
                else:
                    explanation = (
                        f"Files were evaluated with Ollama and organized "
                        f"adhering strictly to your custom instruction: '{self.instruction.strip()}'."
                    )

            self.done.emit(moves, strategy_name, explanation)
        except Exception as e:
            self.failed.emit(str(e))


class ApplyWorker(QThread):
    done = Signal(str, int)  # batch_id, count
    failed = Signal(str)

    def __init__(self, db_path: Path, moves: list[Move]):
        super().__init__()
        self.db_path = db_path
        self.moves = moves

    def run(self):
        try:
            conn = connect(self.db_path)
            batch_id = apply_moves(conn, self.moves)
            conn.close()
            self.done.emit(batch_id, len(self.moves))
        except Exception as e:
            self.failed.emit(str(e))


class UndoWorker(QThread):
    done = Signal(int, int)  # restored, failed
    failed = Signal(str)

    def __init__(self, db_path: Path, batch_id: str):
        super().__init__()
        self.db_path = db_path
        self.batch_id = batch_id

    def run(self):
        try:
            conn = connect(self.db_path)
            restored, failed = undo_batch(conn, self.batch_id)
            conn.close()
            self.done.emit(restored, failed)
        except Exception as e:
            self.failed.emit(str(e))


class AIOrganizeTab(QWidget):
    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self.settings = settings or Settings()
        self.current_moves: list[Move] = []
        self._last_applied_moves: list[Move] = []
        self.last_strategy_name: str = "Ready to plan"
        self.last_explanation: str = "Choose a planning method to organize files."
        self.last_mode: str = "custom"
        self.plan_worker: AIPlanWorker | None = None
        self.apply_worker: ApplyWorker | None = None
        self.undo_worker: UndoWorker | None = None

        self._init_ui()
        self.refresh_history()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        # ── Card 1: Setup & Instructions ──────────────────────────────────────
        self.setup_card = CardFrame(
            "Folder & Organization Rules",
            subtitle="Select the directory and choose how files should be sorted",
        )

        # Source folder row
        src_row = QHBoxLayout()
        src_row.setSpacing(8)
        self.src_edit = QLineEdit()
        self.src_edit.setPlaceholderText("Select directory to organize...")
        src_btn = QPushButton("Browse...")
        src_btn.clicked.connect(self._browse_source)
        self.subfolders_check = QCheckBox("Include subfolders (recursive)")

        src_row.addWidget(self.src_edit, 1)
        src_row.addWidget(src_btn)
        src_row.addWidget(self.subfolders_check)
        self.setup_card.content_layout.addLayout(src_row)

        # Preset Chips Row
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        presets_label = QLabel("Presets:")
        presets_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 500;")
        preset_row.addWidget(presets_label)

        p_type_size = ActionChip("By Type, Then Size")
        p_type_size.clicked.connect(lambda: self.instruction_edit.setText("Sort by file type, then size (Small under 1MB, Medium 1MB-50MB, Large over 50MB)"))
        preset_row.addWidget(p_type_size)

        p_size_only = ActionChip("By Size Brackets")
        p_size_only.clicked.connect(lambda: self.instruction_edit.setText("Group into size categories: Small (under 1MB), Medium (1MB-50MB), Large (over 50MB)"))
        preset_row.addWidget(p_size_only)

        p_smart = ActionChip("Smart Auto-Group")
        p_smart.clicked.connect(lambda: self.instruction_edit.setText("Group into logical folders based on project, topic, and file contents"))
        preset_row.addWidget(p_smart)

        p_date = ActionChip("By Year & Date")
        p_date.clicked.connect(lambda: self.instruction_edit.setText("Group files by their relevant year or date"))
        preset_row.addWidget(p_date)

        p_proj = ActionChip("By Project / Client")
        p_proj.clicked.connect(lambda: self.instruction_edit.setText("Identify project names or clients and group corresponding files"))
        preset_row.addWidget(p_proj)

        preset_row.addStretch()
        self.setup_card.content_layout.addLayout(preset_row)

        # Instructions & Execution Row
        exec_row = QHBoxLayout()
        exec_row.setSpacing(8)

        self.instruction_edit = QLineEdit()
        self.instruction_edit.setPlaceholderText(
            "Enter custom sorting instructions or click a preset above..."
        )
        exec_row.addWidget(self.instruction_edit, 1)

        self.ai_custom_btn = QPushButton("Plan with Custom Instructions")
        self.ai_custom_btn.setStyleSheet(primary_button_style())
        self.ai_custom_btn.setToolTip("Executes custom instructions with Ollama AI.")
        self.ai_custom_btn.clicked.connect(lambda: self.generate_plan(mode="custom"))
        exec_row.addWidget(self.ai_custom_btn)

        self.ai_plan_btn = self.ai_custom_btn  # Compatibility alias

        self.ai_smart_btn = QPushButton("Smart Auto-Organize")
        self.ai_smart_btn.setStyleSheet(ghost_button_style())
        self.ai_smart_btn.setToolTip("Autonomous AI grouping without custom instructions.")
        self.ai_smart_btn.clicked.connect(lambda: self.generate_plan(mode="smart_ai"))
        exec_row.addWidget(self.ai_smart_btn)

        self.fast_plan_btn = QPushButton("Fast Rule-based (by Type)")
        self.fast_plan_btn.setStyleSheet(ghost_button_style())
        self.fast_plan_btn.setToolTip("Deterministic instant sorting by file extension without AI.")
        self.fast_plan_btn.clicked.connect(lambda: self.generate_plan(mode="rule_based"))
        exec_row.addWidget(self.fast_plan_btn)

        self.setup_card.content_layout.addLayout(exec_row)
        layout.addWidget(self.setup_card)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # ── Card 2: Operations & Preview ──────────────────────────────────────
        card_header_action = QWidget()
        cha_layout = QHBoxLayout(card_header_action)
        cha_layout.setContentsMargins(0, 0, 0, 0)
        cha_layout.setSpacing(6)

        quick_undo_btn = QPushButton("Undo Last Batch")
        quick_undo_btn.setStyleSheet(ghost_button_style())
        quick_undo_btn.clicked.connect(self.undo_selected_batch)
        cha_layout.addWidget(quick_undo_btn)

        self.table_card = CardFrame(
            "Proposed Operations",
            subtitle="Review operations before applying changes to disk",
            header_action=card_header_action,
        )

        # Strategy & Rationale strip
        self.strategy_banner = QWidget()
        sb_layout = QVBoxLayout(self.strategy_banner)
        sb_layout.setContentsMargins(0, 0, 0, 0)
        sb_layout.setSpacing(2)

        self.strategy_title_label = QLabel("Strategy: Ready to plan")
        self.strategy_title_label.setStyleSheet("font-weight: 600; color: #58a6ff; font-size: 12px;")
        self.strategy_desc_label = QLabel("Rationale: Choose an organization method above to preview.")
        self.strategy_desc_label.setWordWrap(True)
        self.strategy_desc_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        self.strategy_folders_label = QLabel("")
        self.strategy_folders_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")

        sb_layout.addWidget(self.strategy_title_label)
        sb_layout.addWidget(self.strategy_desc_label)
        sb_layout.addWidget(self.strategy_folders_label)
        self.table_card.content_layout.addWidget(self.strategy_banner)

        # Selection and Count Row
        sel_row = QHBoxLayout()
        sel_row.setSpacing(6)
        sel_all_btn = QPushButton("Select All")
        sel_all_btn.setStyleSheet(ghost_button_style())
        sel_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        sel_none_btn = QPushButton("Select None")
        sel_none_btn.setStyleSheet(ghost_button_style())
        sel_none_btn.clicked.connect(lambda: self._set_all_checked(False))
        self.count_label = QLabel("No preview generated.")
        self.count_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")

        sel_row.addWidget(sel_all_btn)
        sel_row.addWidget(sel_none_btn)
        sel_row.addWidget(self.count_label, 1)
        self.table_card.content_layout.addLayout(sel_row)

        # Moves Table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "Apply", "File", "Proposed Destination", "AI Reason", "Original Path"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table_card.content_layout.addWidget(self.table, 1)

        # Footer Row (Apply action on left, SQLite history & undo on right)
        footer_row = QHBoxLayout()
        footer_row.setSpacing(8)

        self.apply_btn = QPushButton("Apply Selected Moves")
        self.apply_btn.setEnabled(False)
        self.apply_btn.setStyleSheet(success_button_style())
        self.apply_btn.clicked.connect(self.apply_selected)
        footer_row.addWidget(self.apply_btn)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        footer_row.addWidget(self.status_label, 1)

        # Ledger history integration
        hist_label = QLabel("Recent Batches:")
        hist_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 500;")
        footer_row.addWidget(hist_label)

        self.batch_combo = QComboBox()
        self.batch_combo.setMinimumWidth(220)
        footer_row.addWidget(self.batch_combo)

        self.undo_btn = QPushButton("Undo Selected")
        self.undo_btn.setStyleSheet(ghost_button_style())
        self.undo_btn.clicked.connect(self.undo_selected_batch)
        footer_row.addWidget(self.undo_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setStyleSheet(ghost_button_style())
        refresh_btn.clicked.connect(self.refresh_history)
        footer_row.addWidget(refresh_btn)

        self.table_card.content_layout.addLayout(footer_row)
        layout.addWidget(self.table_card, 1)

    def _browse_source(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder to organize")
        if folder:
            self.src_edit.setText(folder)

    def generate_plan(self, mode: str = "custom", is_rule_based: bool | None = None):
        if is_rule_based is True:
            mode = "rule_based"

        folder = self.src_edit.text().strip()
        if not folder or not Path(folder).is_dir():
            QMessageBox.warning(self, "Invalid Folder", "Please choose a valid existing folder.")
            return

        instruction = self.instruction_edit.text().strip()
        if mode == "custom" and not instruction:
            QMessageBox.information(
                self,
                "Custom Instruction Needed",
                "Please enter your custom sort instructions in the text box (e.g. 'sort by file type, then size') "
                "or click one of the Presets.\n\n"
                "To organize automatically without typing instructions, click 'Smart Auto-Organize'.",
            )
            self.instruction_edit.setFocus()
            return

        self.last_mode = mode
        self.ai_custom_btn.setEnabled(False)
        self.ai_smart_btn.setEnabled(False)
        self.fast_plan_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.progress_bar.show()

        if mode == "rule_based":
            method_name = "Fast Rule-based (by Type)"
        elif mode == "smart_ai":
            method_name = "Ollama AI (Autonomous Logic)"
        else:
            method_name = f"Ollama AI (Adhering to: '{instruction}')"

        self.status_label.setText(f"Analyzing files with {method_name}...")

        p = Path(folder)
        if self.subfolders_check.isChecked():
            files = [str(f) for f in p.rglob("*") if f.is_file() and not f.name.startswith(".")]
        else:
            files = [str(f) for f in p.iterdir() if f.is_file() and not f.name.startswith(".")]

        if not files:
            self.progress_bar.hide()
            self.ai_custom_btn.setEnabled(True)
            self.ai_smart_btn.setEnabled(True)
            self.fast_plan_btn.setEnabled(True)
            self.table.setRowCount(0)
            self.count_label.setText("No files found in folder.")
            self.status_label.setText("No files to organize.")
            return

        self.plan_worker = AIPlanWorker(files, folder, instruction, self.settings, mode=mode)
        self.plan_worker.done.connect(self._show_preview)
        self.plan_worker.failed.connect(self._plan_failed)
        self.plan_worker.start()

    def _show_preview(self, moves: list[Move], strategy_name: str = "", explanation: str = ""):
        self.progress_bar.hide()
        self.ai_custom_btn.setEnabled(True)
        self.ai_smart_btn.setEnabled(True)
        self.fast_plan_btn.setEnabled(True)
        self.current_moves = moves

        if not strategy_name:
            strategy_name = "Organized Files Preview"
            explanation = "Preview of proposed file moves based on extension rules."

        self.last_strategy_name = strategy_name
        self.last_explanation = explanation

        self.strategy_title_label.setText(f"Strategy: {strategy_name}")
        self.strategy_desc_label.setText(f"Rationale: {explanation}")

        root_str = self.src_edit.text().strip()
        root = Path(root_str) if root_str else None
        counts: dict[str, int] = {}
        for m in moves:
            dst = Path(m.dst)
            folder_display = "Subfolder"
            if root:
                try:
                    rel = dst.relative_to(root).parent
                    folder_display = str(rel).replace("\\", "/")
                    if folder_display == ".":
                        folder_display = "Root"
                except Exception:
                    folder_display = dst.parent.name
            else:
                folder_display = dst.parent.name
            counts[folder_display] = counts.get(folder_display, 0) + 1

        summary_items = [f"{k} ({v})" for k, v in sorted(counts.items())]
        if summary_items:
            self.strategy_folders_label.setText(
                f"Destination Folders ({len(counts)}): {', '.join(summary_items[:5])}{'...' if len(counts) > 5 else ''}"
            )
        else:
            self.strategy_folders_label.setText("")

        self.table.setRowCount(len(moves))
        for i, m in enumerate(moves):
            src_p = Path(m.src)

            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk_item.setCheckState(Qt.CheckState.Checked)

            file_item = QTableWidgetItem(src_p.name)
            file_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

            dst_item = QTableWidgetItem(m.dst)
            dst_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

            reason_item = QTableWidgetItem(m.reason)
            reason_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

            src_item = QTableWidgetItem(m.src)
            src_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

            self.table.setItem(i, 0, chk_item)
            self.table.setItem(i, 1, file_item)
            self.table.setItem(i, 2, dst_item)
            self.table.setItem(i, 3, reason_item)
            self.table.setItem(i, 4, src_item)

        total = len(moves)
        self.count_label.setText(f"{total} proposed move(s).")
        self.status_label.setText(f"Preview ready: {total} files planned.")
        self.apply_btn.setEnabled(total > 0)

    def _plan_failed(self, err: str):
        self.progress_bar.hide()
        self.ai_custom_btn.setEnabled(True)
        self.ai_smart_btn.setEnabled(True)
        self.fast_plan_btn.setEnabled(True)
        self.status_label.setText(f"Planning failed: {err}")
        QMessageBox.critical(self, "Error", f"Failed to plan moves: {err}")

    def _set_all_checked(self, checked: bool):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                item.setCheckState(state)

    def apply_selected(self):
        selected_moves: list[Move] = []
        for row in range(self.table.rowCount()):
            chk = self.table.item(row, 0)
            if chk and chk.checkState() == Qt.CheckState.Checked:
                if row < len(self.current_moves):
                    selected_moves.append(self.current_moves[row])

        if not selected_moves:
            QMessageBox.information(self, "No Selection", "Please check at least one move to apply.")
            return

        confirm = QMessageBox.question(
            self,
            "Confirm Organization",
            f"Move {len(selected_moves)} file(s)?\n\n"
            f"Strategy: {self.last_strategy_name}\n\n"
            "This operation will organize files safely and can be undone at any time.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self._last_applied_moves = list(selected_moves)
        self.apply_btn.setEnabled(False)
        self.ai_custom_btn.setEnabled(False)
        self.ai_smart_btn.setEnabled(False)
        self.fast_plan_btn.setEnabled(False)
        self.progress_bar.show()
        self.status_label.setText(f"Applying {len(selected_moves)} moves...")

        self.apply_worker = ApplyWorker(self.settings.db_path, selected_moves)
        self.apply_worker.done.connect(self._apply_done)
        self.apply_worker.failed.connect(self._apply_failed)
        self.apply_worker.start()

    def _apply_done(self, batch_id: str, count: int):
        self.progress_bar.hide()
        self.ai_custom_btn.setEnabled(True)
        self.ai_smart_btn.setEnabled(True)
        self.fast_plan_btn.setEnabled(True)

        root_str = self.src_edit.text().strip()
        root = Path(root_str) if root_str else None
        counts: dict[str, int] = {}
        for m in self._last_applied_moves:
            dst = Path(m.dst)
            folder_display = "Subfolder"
            if root:
                try:
                    rel = dst.relative_to(root).parent
                    folder_display = str(rel).replace("\\", "/")
                    if folder_display == ".":
                        folder_display = "Root"
                except Exception:
                    folder_display = dst.parent.name
            else:
                folder_display = dst.parent.name
            counts[folder_display] = counts.get(folder_display, 0) + 1

        folder_lines = "\n".join(f"  • {k}: {v} file(s)" for k, v in sorted(counts.items()))
        if not folder_lines:
            folder_lines = "  • Destination folders updated"

        strategy_name = self.last_strategy_name
        explanation = self.last_explanation

        dialog_msg = (
            f"Successfully organized {count} file(s)!\n"
            f"Batch ID: {batch_id}\n\n"
            f"Strategy Applied:\n{strategy_name}\n\n"
            f"Reason & How It Sorted:\n{explanation}\n\n"
            f"Destination Folders Created:\n{folder_lines}\n\n"
            f"Undo Available:\nYou can undo this batch at any time using the Recent Batches controls."
        )

        self.status_label.setText(f"Done! Moved {count} file(s). Batch ID: {batch_id}")
        QMessageBox.information(self, "Organization Complete", dialog_msg)
        self.refresh_history()
        self.generate_plan(mode=self.last_mode)

    def _apply_failed(self, err: str):
        self.progress_bar.hide()
        self.ai_custom_btn.setEnabled(True)
        self.ai_smart_btn.setEnabled(True)
        self.fast_plan_btn.setEnabled(True)
        self.apply_btn.setEnabled(True)
        self.status_label.setText(f"Apply failed: {err}")
        QMessageBox.critical(self, "Error", f"Failed to apply moves: {err}")

    def refresh_history(self):
        self.batch_combo.clear()
        try:
            conn = connect(self.settings.db_path)
            rows = conn.execute(
                "SELECT batch_id, created_at, COUNT(*) as cnt "
                "FROM operations WHERE status='done' "
                "GROUP BY batch_id ORDER BY id DESC LIMIT 25"
            ).fetchall()
            conn.close()

            if not rows:
                self.batch_combo.addItem("No active batches found", "")
                self.undo_btn.setEnabled(False)
                return

            for r in rows:
                label = f"Batch {r['batch_id']} ({r['cnt']} files) - {r['created_at']}"
                self.batch_combo.addItem(label, r["batch_id"])

            self.undo_btn.setEnabled(True)
        except Exception as e:
            self.batch_combo.addItem(f"Error loading: {e}", "")
            self.undo_btn.setEnabled(False)

    def undo_selected_batch(self):
        batch_id = self.batch_combo.currentData()
        if not batch_id:
            QMessageBox.information(self, "No Batch Selected", "No valid batch selected to undo.")
            return

        confirm = QMessageBox.question(
            self,
            "Confirm Undo",
            f"Undo batch '{batch_id}' and restore files to their previous locations?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self.undo_btn.setEnabled(False)
        self.progress_bar.show()
        self.status_label.setText(f"Undoing batch {batch_id}...")

        self.undo_worker = UndoWorker(self.settings.db_path, batch_id)
        self.undo_worker.done.connect(self._undo_done)
        self.undo_worker.failed.connect(self._undo_failed)
        self.undo_worker.start()

    def _undo_done(self, restored: int, failed: int):
        self.progress_bar.hide()
        self.undo_btn.setEnabled(True)
        msg = f"Undo complete: restored {restored} file(s)."
        if failed > 0:
            msg += f" ({failed} failed)"
        self.status_label.setText(msg)
        QMessageBox.information(self, "Undo Complete", msg)
        self.refresh_history()
        if self.src_edit.text().strip():
            self.generate_plan(mode=self.last_mode)

    def _undo_failed(self, err: str):
        self.progress_bar.hide()
        self.undo_btn.setEnabled(True)
        self.status_label.setText(f"Undo failed: {err}")
        QMessageBox.critical(self, "Error", f"Failed to undo batch: {err}")
