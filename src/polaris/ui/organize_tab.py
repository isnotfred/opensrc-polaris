"""Organize tab: preview rule-based file organization, apply moves, and undo batches."""
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
from ..db.database import connect


class PlanWorker(QThread):
    done = Signal(list)
    failed = Signal(str)

    def __init__(self, file_paths: list[str], dest_dir: str):
        super().__init__()
        self.file_paths = file_paths
        self.dest_dir = dest_dir

    def run(self):
        try:
            moves = plan_by_type(self.file_paths, self.dest_dir)
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


class OrganizeTab(QWidget):
    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self.settings = settings or Settings()
        self.current_moves: list[Move] = []
        self.plan_worker: PlanWorker | None = None
        self.apply_worker: ApplyWorker | None = None
        self.undo_worker: UndoWorker | None = None

        self._init_ui()
        self.refresh_history()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Folder selection controls
        folder_group = QGroupBox("Target Folder")
        fg_layout = QVBoxLayout(folder_group)

        src_row = QHBoxLayout()
        src_label = QLabel("Source:")
        self.src_edit = QLineEdit()
        self.src_edit.setPlaceholderText("Select folder containing files to organize...")
        src_btn = QPushButton("Browse...")
        src_btn.clicked.connect(self._browse_source)
        src_row.addWidget(src_label)
        src_row.addWidget(self.src_edit, 1)
        src_row.addWidget(src_btn)

        opts_row = QHBoxLayout()
        self.subfolders_check = QCheckBox("Include subfolders (recursive)")
        self.preview_btn = QPushButton("Preview Organization")
        self.preview_btn.setStyleSheet("font-weight: bold;")
        self.preview_btn.clicked.connect(self.generate_preview)

        opts_row.addWidget(self.subfolders_check)
        opts_row.addStretch()
        opts_row.addWidget(self.preview_btn)

        fg_layout.addLayout(src_row)
        fg_layout.addLayout(opts_row)
        layout.addWidget(folder_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Preview table & controls
        table_group = QGroupBox("Proposed Moves (Preview)")
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
            "Apply", "File", "Category", "Source Path", "Destination Path"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        tg_layout.addWidget(self.table)

        # Apply button row
        apply_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Selected Moves")
        self.apply_btn.setEnabled(False)
        self.apply_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self.apply_btn.clicked.connect(self.apply_selected)

        self.status_label = QLabel("")
        apply_row.addWidget(self.apply_btn)
        apply_row.addWidget(self.status_label, 1)
        tg_layout.addLayout(apply_row)

        layout.addWidget(table_group, 1)

        # History & Undo section
        history_group = QGroupBox("History & Undo")
        hg_layout = QHBoxLayout(history_group)

        hg_layout.addWidget(QLabel("Recent Batches:"))
        self.batch_combo = QComboBox()
        self.batch_combo.setMinimumWidth(260)
        hg_layout.addWidget(self.batch_combo)

        self.undo_btn = QPushButton("Undo Selected Batch")
        self.undo_btn.clicked.connect(self.undo_selected_batch)
        hg_layout.addWidget(self.undo_btn)

        refresh_history_btn = QPushButton("Refresh History")
        refresh_history_btn.clicked.connect(self.refresh_history)
        hg_layout.addWidget(refresh_history_btn)
        hg_layout.addStretch()

        layout.addWidget(history_group)

    def set_source_folder(self, folder: str):
        """Allow other tabs (like Scan) to pre-fill the source folder."""
        self.src_edit.setText(folder)
        self.generate_preview()

    def _browse_source(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder to organize")
        if folder:
            self.src_edit.setText(folder)

    def generate_preview(self):
        folder = self.src_edit.text().strip()
        if not folder or not Path(folder).is_dir():
            QMessageBox.warning(self, "Invalid Folder", "Please choose a valid existing folder.")
            return

        self.preview_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.progress_bar.show()
        self.status_label.setText("Scanning and planning moves...")

        # Discover files
        p = Path(folder)
        if self.subfolders_check.isChecked():
            files = [str(f) for f in p.rglob("*") if f.is_file()]
        else:
            files = [str(f) for f in p.iterdir() if f.is_file()]

        if not files:
            self.progress_bar.hide()
            self.preview_btn.setEnabled(True)
            self.table.setRowCount(0)
            self.count_label.setText("No files found in folder.")
            self.status_label.setText("No files to organize.")
            return

        self.plan_worker = PlanWorker(files, folder)
        self.plan_worker.done.connect(self._show_preview)
        self.plan_worker.failed.connect(self._preview_failed)
        self.plan_worker.start()

    def _show_preview(self, moves: list[Move]):
        self.progress_bar.hide()
        self.preview_btn.setEnabled(True)
        self.current_moves = moves

        self.table.setRowCount(len(moves))
        for i, m in enumerate(moves):
            src_p = Path(m.src)
            dst_p = Path(m.dst)

            # Checkbox item
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk_item.setCheckState(Qt.CheckState.Checked)

            file_item = QTableWidgetItem(src_p.name)
            file_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

            # Category is the folder name inside the destination root
            cat_item = QTableWidgetItem(dst_p.parent.name)
            cat_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

            src_item = QTableWidgetItem(m.src)
            src_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

            dst_item = QTableWidgetItem(m.dst)
            dst_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

            self.table.setItem(i, 0, chk_item)
            self.table.setItem(i, 1, file_item)
            self.table.setItem(i, 2, cat_item)
            self.table.setItem(i, 3, src_item)
            self.table.setItem(i, 4, dst_item)

        total = len(moves)
        self.count_label.setText(f"{total} proposed move(s).")
        self.status_label.setText(f"Preview ready: {total} files can be organized.")
        self.apply_btn.setEnabled(total > 0)

    def _preview_failed(self, err: str):
        self.progress_bar.hide()
        self.preview_btn.setEnabled(True)
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
            f"Are you sure you want to move {len(selected_moves)} file(s)?\n\n"
            "This operation is safe and can be fully undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self.apply_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.progress_bar.show()
        self.status_label.setText(f"Applying {len(selected_moves)} moves...")

        self.apply_worker = ApplyWorker(self.settings.db_path, selected_moves)
        self.apply_worker.done.connect(self._apply_done)
        self.apply_worker.failed.connect(self._apply_failed)
        self.apply_worker.start()

    def _apply_done(self, batch_id: str, count: int):
        self.progress_bar.hide()
        self.preview_btn.setEnabled(True)
        self.status_label.setText(f"Done! Moved {count} file(s). Batch ID: {batch_id}")
        QMessageBox.information(
            self,
            "Moves Applied",
            f"Successfully organized {count} file(s)!\nBatch ID: {batch_id}\n\n"
            "You can undo this batch at any time using the History & Undo section.",
        )
        self.refresh_history()
        # Refresh preview for any remaining unorganized files
        self.generate_preview()

    def _apply_failed(self, err: str):
        self.progress_bar.hide()
        self.preview_btn.setEnabled(True)
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
            f"Undo batch '{batch_id}' and move files back to their original locations?",
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
        # Refresh preview if currently viewing that folder
        if self.src_edit.text().strip():
            self.generate_preview()

    def _undo_failed(self, err: str):
        self.progress_bar.hide()
        self.undo_btn.setEnabled(True)
        self.status_label.setText(f"Undo failed: {err}")
        QMessageBox.critical(self, "Error", f"Failed to undo batch: {err}")
