"""AI-Powered Organize tab: Natural language file sorting with Drag & Drop, editable preview, search filter, and background download watcher."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings
from ..core.organizer import ApplyResult, Move, apply_moves, plan_by_type, undo_batch
from ..core.watcher import DownloadWatcherWorker
from ..ai.ai_organizer import plan_with_ai, suggest_single_file_placement
from ..db.database import connect
from .suggestion_toast import SuggestionToast


def _friendly_timestamp(ts: str) -> str:
    """Convert a SQLite CURRENT_TIMESTAMP string to a human-readable label."""
    import datetime
    try:
        dt = datetime.datetime.fromisoformat(ts)
        now = datetime.datetime.now(datetime.timezone.utc)
        diff = (now.date() - dt.date()).days
        time_str = dt.strftime("%I:%M %p").lstrip("0")
        if diff == 0:
            return f"Today {time_str}"
        elif diff == 1:
            return f"Yesterday {time_str}"
        elif diff < 7:
            return f"{dt.strftime('%A')} {time_str}"
        else:
            return dt.strftime("%b %d, %Y")
    except Exception:
        return ts  # fall back to raw string if parsing fails


class AIPlanWorker(QThread):
    done = Signal(list)
    failed = Signal(str)
    progress = Signal(int, int)  # current_batch, total_batches

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
                moves = plan_with_ai(
                    self.file_paths, self.dest_dir, self.instruction, self.settings,
                    progress_callback=lambda cur, tot: self.progress.emit(cur, tot),
                )
            self.done.emit(moves)
        except Exception as e:
            self.failed.emit(str(e))


class ApplyWorker(QThread):
    done = Signal(str, int, int, str)  # batch_id, succeeded, failed, mode
    failed = Signal(str)

    def __init__(self, db_path: Path, moves: list[Move], mode: str = "move"):
        super().__init__()
        self.db_path = db_path
        self.moves = moves
        self.mode = mode

    def run(self):
        try:
            conn = connect(self.db_path)
            result = apply_moves(conn, self.moves, mode=self.mode)
            conn.close()
            self.done.emit(result.batch_id, result.succeeded, result.failed, self.mode)
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


class AnalyzeIncomingWorker(QThread):
    ready = Signal(dict)

    def __init__(self, file_path: str, settings: Settings):
        super().__init__()
        self.file_path = file_path
        self.settings = settings

    def run(self):
        try:
            res = suggest_single_file_placement(self.file_path, settings=self.settings)
            if res:
                self.ready.emit(res)
        except Exception:
            pass


class AIOrganizeTab(QWidget):
    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self.settings = settings or Settings()
        self.current_moves: list[Move] = []
        self.plan_worker: AIPlanWorker | None = None
        self.apply_worker: ApplyWorker | None = None
        self.undo_worker: UndoWorker | None = None

        # Background Watcher state
        self.watcher: DownloadWatcherWorker | None = None
        self.watch_folder = str(Path.home() / "Downloads")
        self.active_toasts: list[SuggestionToast] = []
        self.analyze_workers: list[AnalyzeIncomingWorker] = []
        self._session_file_count = 0

        self.setAcceptDrops(True)
        self._init_ui()
        self.refresh_history()
        # Defer watcher auto-start until after the main window is fully shown
        # (avoids QMessageBox popping up before the window renders)
        QTimer.singleShot(600, lambda: self._toggle_watcher(silent=True))

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Target Folder & Drag-and-Drop Zone
        folder_group = QGroupBox("1. Target Folder & Files (Drag & Drop Supported)")
        fg_layout = QVBoxLayout(folder_group)

        src_row = QHBoxLayout()
        src_label = QLabel("Folder:")
        self.src_edit = QLineEdit()
        self.src_edit.setPlaceholderText("Drag and drop a folder here, or click Browse...")
        src_btn = QPushButton("Browse...")
        src_btn.clicked.connect(self._browse_source)
        src_row.addWidget(src_label)
        src_row.addWidget(self.src_edit, 1)
        src_row.addWidget(src_btn)

        opts_row = QHBoxLayout()
        self.subfolders_check = QCheckBox("Include subfolders (recursive)")

        # Mode selection: Move vs Copy
        opts_row.addWidget(self.subfolders_check)
        opts_row.addSpacing(20)
        opts_row.addWidget(QLabel("Action Mode:"))
        self.move_radio = QRadioButton("Move Files (Default)")
        self.move_radio.setChecked(True)
        self.copy_radio = QRadioButton("Copy Files (Preserve Originals)")
        opts_row.addWidget(self.move_radio)
        opts_row.addWidget(self.copy_radio)
        opts_row.addStretch()

        fg_layout.addLayout(src_row)
        fg_layout.addLayout(opts_row)
        layout.addWidget(folder_group)

        # 2. AI Instructions & Presets
        ai_group = QGroupBox("2. Local AI Instructions (Powered by Qwen 2.5 / Llama 3.2)")
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
        self.ai_plan_btn.setStyleSheet("font-weight: bold; background-color: #2563eb; color: white; padding: 6px 14px;")
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
        table_group = QGroupBox("3. Proposed Moves (Preview & Customize)")
        tg_layout = QVBoxLayout(table_group)

        top_table_row = QHBoxLayout()
        sel_all_btn = QPushButton("Select All")
        sel_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        sel_none_btn = QPushButton("Select None")
        sel_none_btn.clicked.connect(lambda: self._set_all_checked(False))

        # Real-time search filter in preview table
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("🔍 Filter preview table (e.g. .pdf, invoices, 2024)...")
        self.filter_edit.textChanged.connect(self._on_filter_changed)

        self.count_label = QLabel("No preview generated.")

        top_table_row.addWidget(sel_all_btn)
        top_table_row.addWidget(sel_none_btn)
        top_table_row.addWidget(self.filter_edit, 1)
        top_table_row.addWidget(self.count_label)
        tg_layout.addLayout(top_table_row)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "Apply", "File", "Proposed Destination (Double-click to edit)", "AI Reason", "Original Path"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

        # Enable right-click context menu
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)

        tg_layout.addWidget(self.table)

        apply_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Selected Operations")
        self.apply_btn.setEnabled(False)
        self.apply_btn.setStyleSheet("font-weight: bold; padding: 6px 16px; font-size: 13px;")
        self.apply_btn.clicked.connect(self.apply_selected)

        self.status_label = QLabel("")
        apply_row.addWidget(self.apply_btn)
        apply_row.addWidget(self.status_label, 1)
        tg_layout.addLayout(apply_row)

        layout.addWidget(table_group, 1)

        # 4. Live Downloads Watcher Panel
        watcher_group = QGroupBox("⚡ Background Downloads Monitor (Desktop Toasts)")
        wg_layout = QHBoxLayout(watcher_group)

        self.watcher_status = QLabel(f"Folder: {self.watch_folder} (Inactive)")
        self.watcher_status.setStyleSheet("color: #94a3b8; font-size: 12px;")

        self.watcher_count_label = QLabel("0 files detected")
        self.watcher_count_label.setStyleSheet("color: #64748b; font-size: 11px;")

        change_watch_btn = QPushButton("Change Folder...")
        change_watch_btn.clicked.connect(self._change_watch_folder)

        self.toggle_watcher_btn = QPushButton("▶ Start Live Watcher")
        self.toggle_watcher_btn.setStyleSheet("font-weight: bold; padding: 4px 12px;")
        self.toggle_watcher_btn.clicked.connect(self._toggle_watcher)

        test_trigger_btn = QPushButton("Simulate New File")
        test_trigger_btn.setToolTip("Test the AI rename toast on a sample file")
        test_trigger_btn.clicked.connect(self._test_trigger_incoming)

        wg_layout.addWidget(self.watcher_status, 1)
        wg_layout.addWidget(self.watcher_count_label)
        wg_layout.addWidget(change_watch_btn)
        wg_layout.addWidget(self.toggle_watcher_btn)
        wg_layout.addWidget(test_trigger_btn)
        layout.addWidget(watcher_group)

        # 5. History & Undo
        history_group = QGroupBox("History & Undo")
        hg_layout = QHBoxLayout(history_group)

        hg_layout.addWidget(QLabel("Recent Batches:"))
        self.batch_combo = QComboBox()
        self.batch_combo.setMinimumWidth(280)
        hg_layout.addWidget(self.batch_combo)

        self.undo_btn = QPushButton("Undo Selected Batch")
        self.undo_btn.clicked.connect(self.undo_selected_batch)
        hg_layout.addWidget(self.undo_btn)

        refresh_btn = QPushButton("Refresh History")
        refresh_btn.clicked.connect(self.refresh_history)
        hg_layout.addWidget(refresh_btn)
        hg_layout.addStretch()

        layout.addWidget(history_group)

    # Background Watcher Controls
    def _toggle_watcher(self, silent: bool = False):
        if self.watcher and self.watcher.isRunning():
            self.watcher.stop()
            self.watcher.wait()
            self.watcher = None
            self.analyze_workers = [w for w in self.analyze_workers if w.isRunning()]
            self.toggle_watcher_btn.setText("▶ Start Live Watcher")
            self.toggle_watcher_btn.setStyleSheet("font-weight: bold; padding: 4px 12px;")
            self.watcher_status.setText(f"Folder: {self.watch_folder} (Inactive)")
            self.watcher_status.setStyleSheet("color: #94a3b8; font-size: 12px;")
        else:
            p = Path(self.watch_folder)
            if not p.is_dir():
                if silent:
                    self.watcher_status.setText(f"Folder: {self.watch_folder} (Not found)")
                    self.watcher_status.setStyleSheet("color: #f87171; font-size: 12px;")
                else:
                    QMessageBox.warning(self, "Folder Not Found", f"Cannot watch {self.watch_folder}: folder does not exist.")
                return

            self._session_file_count = 0
            self.watcher_count_label.setText("0 files detected")

            self.watcher = DownloadWatcherWorker(p)
            self.watcher.new_file_ready.connect(self._on_incoming_file_detected)
            self.watcher.session_count.connect(self._on_session_count_updated)
            self.watcher.start()
            self.toggle_watcher_btn.setText("⏹ Stop Live Watcher")
            self.toggle_watcher_btn.setStyleSheet("font-weight: bold; padding: 4px 12px; background-color: #dc2626; color: white;")
            self.watcher_status.setText(f"🟢 Monitoring {self.watch_folder} — toasts will pop up when new files land!")
            self.watcher_status.setStyleSheet("color: #4ade80; font-weight: bold; font-size: 12px;")

    def _change_watch_folder(self):
        f = QFileDialog.getExistingDirectory(self, "Select folder to monitor", self.watch_folder)
        if f:
            was_running = self.watcher and self.watcher.isRunning()
            if was_running:
                self._toggle_watcher()
            self.watch_folder = f
            self.watcher_status.setText(f"Folder: {self.watch_folder} (Inactive)")
            if was_running:
                self._toggle_watcher()

    def _on_session_count_updated(self, count: int):
        self.watcher_count_label.setText(f"{count} file{'s' if count != 1 else ''} detected this session")
        self.watcher_count_label.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: bold;")

    def _on_incoming_file_detected(self, file_path: str):
        # Clean up finished workers before adding a new one
        self.analyze_workers = [w for w in self.analyze_workers if w.isRunning()]

        worker = AnalyzeIncomingWorker(file_path, self.settings)
        worker.ready.connect(self._show_suggestion_toast)
        self.analyze_workers.append(worker)
        worker.start()

    def _show_suggestion_toast(self, suggestion: dict):
        # Clean up closed toasts before computing stack position
        self.active_toasts = [t for t in self.active_toasts if t.isVisible()]
        stack_idx = len(self.active_toasts)
        toast = SuggestionToast(suggestion, self.settings, stack_index=stack_idx)
        toast.applied.connect(lambda _s, _d: self.refresh_history())
        toast.dismissed.connect(self._on_toast_dismissed)
        self.active_toasts.append(toast)
        toast.show()

    def _test_trigger_incoming(self):
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Select a file to test AI auto-placement toast",
            str(Path.cwd() / "sample_files"),
            "All Files (*.*)"
        )
        if f:
            self._on_incoming_file_detected(f)

    def _on_toast_dismissed(self):
        """Prune the active_toasts list whenever any toast is closed."""
        self.active_toasts = [t for t in self.active_toasts if t.isVisible()]

    # Drag & Drop Events
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if Path(path).is_dir():
                self.src_edit.setText(path)
                event.acceptProposedAction()
                self.generate_plan(is_rule_based=False)

    def _browse_source(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder to organize")
        if folder:
            self.src_edit.setText(folder)

    def _on_filter_changed(self, text: str):
        query = text.strip().lower()
        for row in range(self.table.rowCount()):
            if not query:
                self.table.setRowHidden(row, False)
                continue

            file_text = (self.table.item(row, 1).text() if self.table.item(row, 1) else "").lower()
            dst_text = (self.table.item(row, 2).text() if self.table.item(row, 2) else "").lower()
            reason_text = (self.table.item(row, 3).text() if self.table.item(row, 3) else "").lower()

            matched = query in file_text or query in dst_text or query in reason_text
            self.table.setRowHidden(row, not matched)

    def _show_context_menu(self, pos):
        item = self.table.itemAt(pos)
        if not item:
            return

        row = item.row()
        src_path_item = self.table.item(row, 4)
        if not src_path_item:
            return

        src_path = src_path_item.text()
        menu = QMenu(self)

        reveal_action = QAction("📂 Reveal in File Explorer", self)
        reveal_action.triggered.connect(lambda: self._reveal_in_explorer(src_path))
        menu.addAction(reveal_action)

        open_action = QAction("📄 Open File", self)
        open_action.triggered.connect(lambda: self._open_file(src_path))
        menu.addAction(open_action)

        menu.addSeparator()

        chk_item = self.table.item(row, 0)
        if chk_item and chk_item.checkState() == Qt.CheckState.Checked:
            toggle_action = QAction("❌ Exclude from Organization", self)
            toggle_action.triggered.connect(lambda: chk_item.setCheckState(Qt.CheckState.Unchecked))
        else:
            toggle_action = QAction("✅ Include in Organization", self)
            toggle_action.triggered.connect(lambda: chk_item.setCheckState(Qt.CheckState.Checked))
        menu.addAction(toggle_action)

        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _reveal_in_explorer(self, file_path: str):
        p = Path(file_path)
        if p.exists():
            subprocess.Popen(f'explorer /select,"{p.resolve()}"')

    def _open_file(self, file_path: str):
        p = Path(file_path)
        if p.exists():
            os.startfile(str(p.resolve()))

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
        method_name = "Rule-based organizer" if is_rule_based else "Local AI (Qwen / Llama)"
        self.status_label.setText(f"Scanning files for {method_name}...")

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
        self.plan_worker.progress.connect(self._on_plan_progress)
        self.plan_worker.start()

    def _on_plan_progress(self, current: int, total: int):
        self.status_label.setText(f"AI planning batch {current} of {total}…")

    def _show_preview(self, moves: list[Move]):
        self.progress_bar.hide()
        self.ai_plan_btn.setEnabled(True)
        self.fast_plan_btn.setEnabled(True)
        self.current_moves = moves

        self.table.setRowCount(len(moves))
        for i, m in enumerate(moves):
            src_p = Path(m.src)

            # Checkbox
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk_item.setCheckState(Qt.CheckState.Checked)

            # File name
            file_item = QTableWidgetItem(src_p.name)
            file_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

            # Editable Destination
            dst_item = QTableWidgetItem(m.dst)
            dst_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable)
            dst_item.setToolTip("Double-click to manually customize destination path before applying")

            # AI Reason
            reason_item = QTableWidgetItem(m.reason)
            reason_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

            # Source Path
            src_item = QTableWidgetItem(m.src)
            src_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

            self.table.setItem(i, 0, chk_item)
            self.table.setItem(i, 1, file_item)
            self.table.setItem(i, 2, dst_item)
            self.table.setItem(i, 3, reason_item)
            self.table.setItem(i, 4, src_item)

        total = len(moves)
        self.count_label.setText(f"{total} proposed move(s).")
        self.status_label.setText(f"Preview ready: {total} files planned. Double-click any destination cell to customize.")
        self.apply_btn.setEnabled(total > 0)
        self._on_filter_changed(self.filter_edit.text())

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
                src_item = self.table.item(row, 4)
                dst_item = self.table.item(row, 2)
                reason_item = self.table.item(row, 3)

                if src_item and dst_item:
                    selected_moves.append(Move(
                        src=src_item.text().strip(),
                        dst=dst_item.text().strip(),
                        reason=reason_item.text().strip() if reason_item else "",
                    ))

        if not selected_moves:
            QMessageBox.information(self, "No Selection", "Please check at least one move to apply.")
            return

        mode = "copy" if self.copy_radio.isChecked() else "move"
        verb = "Copy" if mode == "copy" else "Move"

        confirm = QMessageBox.question(
            self,
            f"Confirm {verb}",
            f"{verb} {len(selected_moves)} file(s)?\n\n"
            f"Mode: {'Duplicate without deleting originals' if mode == 'copy' else 'Relocate to organized subfolders'}.\n"
            "This operation can be completely undone at any time.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self.apply_btn.setEnabled(False)
        self.ai_plan_btn.setEnabled(False)
        self.fast_plan_btn.setEnabled(False)
        self.progress_bar.show()
        self.status_label.setText(f"Applying {len(selected_moves)} operations ({mode})...")

        self.apply_worker = ApplyWorker(self.settings.db_path, selected_moves, mode=mode)
        self.apply_worker.done.connect(self._apply_done)
        self.apply_worker.failed.connect(self._apply_failed)
        self.apply_worker.start()

    def _apply_done(self, batch_id: str, succeeded: int, failed: int, mode: str):
        self.progress_bar.hide()
        self.ai_plan_btn.setEnabled(True)
        self.fast_plan_btn.setEnabled(True)
        action_name = "copied" if mode == "copy" else "moved"
        status = f"Done! {action_name.capitalize()} {succeeded} file(s). Batch: {batch_id[:8]}"
        if failed:
            status += f"  ({failed} failed)"
        self.status_label.setText(status)
        detail = f"Successfully {action_name} {succeeded} file(s)!"
        if failed:
            detail += f"\n⚠️ {failed} file(s) could not be {action_name} (check permissions)."
        detail += f"\nBatch ID: {batch_id}\n\nYou can undo this batch at any time."
        QMessageBox.information(self, "Operations Applied", detail)
        self.refresh_history()
        # Only re-plan if the user still has a folder selected
        if self.src_edit.text().strip():
            self.generate_plan(is_rule_based=False)

    def _apply_failed(self, err: str):
        self.progress_bar.hide()
        self.ai_plan_btn.setEnabled(True)
        self.fast_plan_btn.setEnabled(True)
        self.apply_btn.setEnabled(True)
        self.status_label.setText(f"Operation failed: {err}")
        QMessageBox.critical(self, "Error", f"Failed to apply moves: {err}")

    def refresh_history(self):
        self.batch_combo.clear()
        try:
            conn = connect(self.settings.db_path)
            rows = conn.execute(
                "SELECT batch_id, action, created_at, COUNT(*) as cnt "
                "FROM operations WHERE status='done' "
                "GROUP BY batch_id ORDER BY id DESC LIMIT 25"
            ).fetchall()
            conn.close()

            if not rows:
                self.batch_combo.addItem("No active batches found", "")
                self.undo_btn.setEnabled(False)
                return

            for r in rows:
                action_tag = r["action"].upper()
                bid_short = r["batch_id"][:8]
                ts = _friendly_timestamp(r["created_at"])
                label = f"[{action_tag}] {bid_short}… · {r['cnt']} file(s) · {ts}"
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
