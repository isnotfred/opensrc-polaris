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
    done = Signal(list)
    failed = Signal(str)

    def __init__(self, file_paths: list[str], dest_dir: str, instruction: str, settings: Settings):
        super().__init__()
        self.file_paths = file_paths
        self.dest_dir = dest_dir
        self.instruction = instruction
        self.settings = settings

    def run(self):
        try:
            if self.instruction.strip().lower() == "__rule_based__":
                moves = plan_by_type(self.file_paths, self.dest_dir)
            else:
                moves = plan_with_ai(self.file_paths, self.dest_dir, self.instruction, self.settings)
            self.done.emit(moves)
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

        # 2. AI Instructions
        ai_group = QGroupBox("2. Local AI Instructions (Powered by Ollama Llama 3.2)")
        ai_layout = QVBoxLayout(ai_group)

        self.instruction_edit = QLineEdit()
        self.instruction_edit.setPlaceholderText(
            "e.g. 'Organize research papers by topic', 'Group invoices by client', or leave blank for smart auto-grouping"
        )
        ai_layout.addWidget(self.instruction_edit)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Quick Presets:"))

        p1 = QPushButton("Smart Auto-Group")
        p1.clicked.connect(lambda: self.instruction_edit.setText("Group into logical folders based on project, topic, and file contents"))
        preset_row.addWidget(p1)

        p2 = QPushButton("Sort by Year & Date")
        p2.clicked.connect(lambda: self.instruction_edit.setText("Group files by their relevant year or date"))
        preset_row.addWidget(p2)

        p3 = QPushButton("Sort by Project / Client")
        p3.clicked.connect(lambda: self.instruction_edit.setText("Identify project names or clients and group corresponding files"))
        preset_row.addWidget(p3)

        preset_row.addStretch()

        self.ai_plan_btn = QPushButton("✨ Plan with Local AI")
        self.ai_plan_btn.setStyleSheet("font-weight: bold; background-color: #2563eb; color: white; padding: 6px 12px;")
        self.ai_plan_btn.clicked.connect(lambda: self.generate_plan(is_rule_based=False))
        preset_row.addWidget(self.ai_plan_btn)

        self.fast_plan_btn = QPushButton("Fast Rule-based (by Type)")
        self.fast_plan_btn.clicked.connect(lambda: self.generate_plan(is_rule_based=True))
        preset_row.addWidget(self.fast_plan_btn)

        ai_layout.addLayout(preset_row)
        layout.addWidget(ai_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # 3. Preview Table & Selection
        table_group = QGroupBox("3. Proposed Moves (Preview)")
        tg_layout = QVBoxLayout(table_group)

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

    def generate_plan(self, is_rule_based: bool = False):
        folder = self.src_edit.text().strip()
        if not folder or not Path(folder).is_dir():
            QMessageBox.warning(self, "Invalid Folder", "Please choose a valid existing folder.")
            return

        self.ai_plan_btn.setEnabled(False)
        self.fast_plan_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.progress_bar.show()

        instruction = "__rule_based__" if is_rule_based else self.instruction_edit.text().strip()
        method_name = "Rule-based organizer" if is_rule_based else "Ollama AI"
        self.status_label.setText(f"Analyzing files with {method_name}...")

        p = Path(folder)
        if self.subfolders_check.isChecked():
            files = [str(f) for f in p.rglob("*") if f.is_file() and not f.name.startswith(".")]
        else:
            files = [str(f) for f in p.iterdir() if f.is_file() and not f.name.startswith(".")]

        if not files:
            self.progress_bar.hide()
            self.ai_plan_btn.setEnabled(True)
            self.fast_plan_btn.setEnabled(True)
            self.table.setRowCount(0)
            self.count_label.setText("No files found in folder.")
            self.status_label.setText("No files to organize.")
            return

        self.plan_worker = AIPlanWorker(files, folder, instruction, self.settings)
        self.plan_worker.done.connect(self._show_preview)
        self.plan_worker.failed.connect(self._plan_failed)
        self.plan_worker.start()

    def _show_preview(self, moves: list[Move]):
        self.progress_bar.hide()
        self.ai_plan_btn.setEnabled(True)
        self.fast_plan_btn.setEnabled(True)
        self.current_moves = moves

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
        self.ai_plan_btn.setEnabled(True)
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
            "This operation will organize files safely and can be undone at any time.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self.apply_btn.setEnabled(False)
        self.ai_plan_btn.setEnabled(False)
        self.fast_plan_btn.setEnabled(False)
        self.progress_bar.show()
        self.status_label.setText(f"Applying {len(selected_moves)} moves...")

        self.apply_worker = ApplyWorker(self.settings.db_path, selected_moves)
        self.apply_worker.done.connect(self._apply_done)
        self.apply_worker.failed.connect(self._apply_failed)
        self.apply_worker.start()

    def _apply_done(self, batch_id: str, count: int):
        self.progress_bar.hide()
        self.ai_plan_btn.setEnabled(True)
        self.fast_plan_btn.setEnabled(True)
        self.status_label.setText(f"Done! Moved {count} file(s). Batch ID: {batch_id}")
        QMessageBox.information(
            self,
            "Moves Applied",
            f"Successfully organized {count} file(s)!\nBatch ID: {batch_id}\n\n"
            "You can undo this batch at any time using the History & Undo section.",
        )
        self.refresh_history()
        self.generate_plan(is_rule_based=False)

    def _apply_failed(self, err: str):
        self.progress_bar.hide()
        self.ai_plan_btn.setEnabled(True)
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
            self.generate_plan(is_rule_based=False)

    def _undo_failed(self, err: str):
        self.progress_bar.hide()
        self.undo_btn.setEnabled(True)
        self.status_label.setText(f"Undo failed: {err}")
        QMessageBox.critical(self, "Error", f"Failed to undo batch: {err}")
