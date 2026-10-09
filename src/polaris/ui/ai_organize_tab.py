"""AI-Powered Organize tab: Natural language and intelligent file sorting with Ollama, safe preview, and undo."""
from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
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
                strategy_name = "⚡ Fast Rule-based (by Type)"
                explanation = (
                    "Files were deterministically sorted into standard category folders "
                    "(Documents, Presentations, Spreadsheets, Images, Videos, Audio, Archives, Code) "
                    "based purely on file extensions."
                )
            elif self.mode == "smart_ai" or not self.instruction.strip():
                moves = plan_with_ai(self.file_paths, self.dest_dir, "", self.settings)
                strategy_name = "🧠 Smart Auto-Organize (Autonomous AI Logic)"
                explanation = (
                    "Files were analyzed by Ollama Llama 3.2 and automatically organized into logical subfolders "
                    "based on detected project names, topics, file types, and contents."
                )
            else:
                moves = plan_with_ai(self.file_paths, self.dest_dir, self.instruction, self.settings)
                strategy_name = f'✨ Custom Instructions: "{self.instruction.strip()}"'
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
                        f"Files were evaluated with Ollama Llama 3.2 and organized "
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

        # 1. Target Folder
        folder_group = QGroupBox("1. Target Folder & Files")
        fg_layout = QVBoxLayout(folder_group)

        src_row = QHBoxLayout()
        src_label = QLabel("Folder:")
        self.src_edit = QLineEdit()
        self.src_edit.setPlaceholderText("Select the folder containing files you want to organize...")
        src_btn = QPushButton("Browse...")
        src_btn.clicked.connect(self._browse_source)
        src_row.addWidget(src_label)
        src_row.addWidget(self.src_edit, 1)
        src_row.addWidget(src_btn)

        opts_row = QHBoxLayout()
        self.subfolders_check = QCheckBox("Include subfolders (recursive)")
        opts_row.addWidget(self.subfolders_check)
        opts_row.addStretch()

        fg_layout.addLayout(src_row)
        fg_layout.addLayout(opts_row)
        layout.addWidget(folder_group)

        # 2. AI Instructions & Strategy Modes
        ai_group = QGroupBox("2. Organization Instructions & Strategy Modes")
        ai_layout = QVBoxLayout(ai_group)

        self.instruction_edit = QLineEdit()
        self.instruction_edit.setPlaceholderText(
            "e.g. 'sort by file type, then size', 'group invoices by client', or click a preset..."
        )
        ai_layout.addWidget(self.instruction_edit)

        # Quick Presets Row
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Quick Presets:"))

        p_type_size = QPushButton("📁 Sort by Type, then Size")
        p_type_size.clicked.connect(lambda: self.instruction_edit.setText("Sort by file type, then size (Small under 1MB, Medium 1MB-50MB, Large over 50MB)"))
        preset_row.addWidget(p_type_size)

        p_size_only = QPushButton("📊 Sort by Size Brackets")
        p_size_only.clicked.connect(lambda: self.instruction_edit.setText("Group into size categories: Small (under 1MB), Medium (1MB-50MB), Large (over 50MB)"))
        preset_row.addWidget(p_size_only)

        p_smart = QPushButton("⚡ Smart Auto-Group")
        p_smart.clicked.connect(lambda: self.instruction_edit.setText("Group into logical folders based on project, topic, and file contents"))
        preset_row.addWidget(p_smart)

        p_date = QPushButton("📅 Sort by Year & Date")
        p_date.clicked.connect(lambda: self.instruction_edit.setText("Group files by their relevant year or date"))
        preset_row.addWidget(p_date)

        p_proj = QPushButton("🏢 Sort by Project / Client")
        p_proj.clicked.connect(lambda: self.instruction_edit.setText("Identify project names or clients and group corresponding files"))
        preset_row.addWidget(p_proj)

        preset_row.addStretch()
        ai_layout.addLayout(preset_row)

        # Action Buttons Row
        action_row = QHBoxLayout()
        action_row.addWidget(QLabel("Plan Method:"))

        # Button 1: Adhere strictly to user's entered instructions
        self.ai_custom_btn = QPushButton("✨ Plan with Custom Instructions")
        self.ai_custom_btn.setStyleSheet("font-weight: bold; background-color: #2563eb; color: white; padding: 6px 14px; border-radius: 4px;")
        self.ai_custom_btn.setToolTip("Strictly executes the instructions typed into the box above using Ollama AI.")
        self.ai_custom_btn.clicked.connect(lambda: self.generate_plan(mode="custom"))
        action_row.addWidget(self.ai_custom_btn)

        # Alias for backward compatibility
        self.ai_plan_btn = self.ai_custom_btn

        # Button 2: Autonomous AI Logic
        self.ai_smart_btn = QPushButton("🧠 Smart Auto-Organize (AI Logic)")
        self.ai_smart_btn.setStyleSheet("font-weight: bold; background-color: #059669; color: white; padding: 6px 14px; border-radius: 4px;")
        self.ai_smart_btn.setToolTip("Uses AI's own autonomous logic to categorize files by topic, project, and type without custom text.")
        self.ai_smart_btn.clicked.connect(lambda: self.generate_plan(mode="smart_ai"))
        action_row.addWidget(self.ai_smart_btn)

        # Button 3: Deterministic Rule-based
        self.fast_plan_btn = QPushButton("⚡ Fast Rule-based (by Type)")
        self.fast_plan_btn.setStyleSheet("font-weight: bold; padding: 6px 12px;")
        self.fast_plan_btn.setToolTip("Instantly organizes files into standard format categories (Documents, Images, Code, etc.) without calling AI.")
        self.fast_plan_btn.clicked.connect(lambda: self.generate_plan(mode="rule_based"))
        action_row.addWidget(self.fast_plan_btn)

        action_row.addStretch()
        ai_layout.addLayout(action_row)

        layout.addWidget(ai_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # 3. Preview Table & Selection
        table_group = QGroupBox("3. Proposed Moves (Preview)")
        tg_layout = QVBoxLayout(table_group)

        # Active Strategy & Reasoning Banner
        self.strategy_banner = QGroupBox("Active Strategy & AI Rationale")
        sb_layout = QVBoxLayout(self.strategy_banner)
        sb_layout.setContentsMargins(10, 8, 10, 8)

        self.strategy_title_label = QLabel("📌 Strategy: Ready to plan")
        self.strategy_title_label.setStyleSheet("font-weight: bold; color: #38bdf8;")
        self.strategy_desc_label = QLabel("💡 How & Why It Sorted: Choose an organization method above to preview.")
        self.strategy_desc_label.setWordWrap(True)
        self.strategy_folders_label = QLabel("")
        self.strategy_folders_label.setStyleSheet("color: #94a3b8; font-size: 11px;")

        sb_layout.addWidget(self.strategy_title_label)
        sb_layout.addWidget(self.strategy_desc_label)
        sb_layout.addWidget(self.strategy_folders_label)
        tg_layout.addWidget(self.strategy_banner)

        sel_row = QHBoxLayout()
        sel_all_btn = QPushButton("Select All")
        sel_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        sel_none_btn = QPushButton("Select None")
        sel_none_btn.clicked.connect(lambda: self._set_all_checked(False))
        self.count_label = QLabel("No preview generated.")

        sel_row.addWidget(sel_all_btn)
        sel_row.addWidget(sel_none_btn)
        sel_row.addWidget(self.count_label, 1)
        tg_layout.addLayout(sel_row)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "Apply", "File", "Proposed Destination", "AI Reason", "Original Path"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        tg_layout.addWidget(self.table)

        apply_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Selected Moves")
        self.apply_btn.setEnabled(False)
        self.apply_btn.setStyleSheet("font-weight: bold; padding: 6px 14px;")
        self.apply_btn.clicked.connect(self.apply_selected)

        self.status_label = QLabel("")
        apply_row.addWidget(self.apply_btn)
        apply_row.addWidget(self.status_label, 1)
        tg_layout.addLayout(apply_row)

        layout.addWidget(table_group, 1)

        # 4. History & Undo
        history_group = QGroupBox("History & Undo")
        hg_layout = QHBoxLayout(history_group)

        hg_layout.addWidget(QLabel("Recent Batches:"))
        self.batch_combo = QComboBox()
        self.batch_combo.setMinimumWidth(260)
        hg_layout.addWidget(self.batch_combo)

        self.undo_btn = QPushButton("Undo Selected Batch")
        self.undo_btn.clicked.connect(self.undo_selected_batch)
        hg_layout.addWidget(self.undo_btn)

        refresh_btn = QPushButton("Refresh History")
        refresh_btn.clicked.connect(self.refresh_history)
        hg_layout.addWidget(refresh_btn)
        hg_layout.addStretch()

        layout.addWidget(history_group)

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
                "or click one of the Quick Presets.\n\n"
                "To organize automatically without typing instructions, click '🧠 Smart Auto-Organize (AI Logic)'.",
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

        self.strategy_title_label.setText(f"📌 Active Strategy: {strategy_name}")
        self.strategy_desc_label.setText(f"💡 Rationale: {explanation}")

        # Compute destination folder breakdown
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
                f"📁 Destination Folders ({len(counts)}): {', '.join(summary_items[:5])}{'...' if len(counts) > 5 else ''}"
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
        self.status_label.setText(f"Preview ready: {total} files planned under {strategy_name}.")
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

        # Build bulleted list of destination folders for selected moves
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
            f"📋 Strategy Applied:\n{strategy_name}\n\n"
            f"💡 Reason & How It Sorted:\n{explanation}\n\n"
            f"📁 Destination Folders Created:\n{folder_lines}\n\n"
            f"⏪ Undo Available:\nYou can undo this batch at any time using the History & Undo section below."
        )

        self.status_label.setText(f"Done! Moved {count} file(s) using {strategy_name}. Batch ID: {batch_id}")
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
