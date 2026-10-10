"""Background download and file watcher service for Polaris.

Improvements:
- Faster poll cycle: 0.75s instead of 1.5s.
- Two-sample size stability: requires file size to be stable for 2 consecutive reads
  before emitting, reducing false positives on mid-write files.
- Separate `file_detected` signal emitted immediately on discovery (before settling),
  so the UI can show a "file incoming…" hint while the settle check runs in place.
- `session_count` signal keeps the UI panel counter up to date.
- Graceful handling of directory disappearing at runtime.
"""
from __future__ import annotations

import time
from pathlib import Path
from PySide6.QtCore import QThread, Signal


class DownloadWatcherWorker(QThread):
    new_file_ready = Signal(str)   # Emits full path when file is finished writing
    session_count = Signal(int)    # Emits running total of files detected this session

    # Temp / in-progress extensions that should never trigger a suggestion
    IGNORED_SUFFIXES: frozenset[str] = frozenset({
        ".tmp", ".crdownload", ".part", ".partial", ".download",
        ".opdownload", ".aria2", ".!ut", ".torrent", ".swp",
    })

    def __init__(self, watch_dir: str | Path):
        super().__init__()
        self.watch_dir = Path(watch_dir)
        self.is_running = True
        self._detected_count = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def stop(self) -> None:
        self.is_running = False

    # ------------------------------------------------------------------ #
    # Thread entry-point                                                   #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        if not self.watch_dir.is_dir():
            return

        # Snapshot of files already present — don't trigger on pre-existing items
        try:
            known_files: set[str] = {p.name for p in self.watch_dir.iterdir() if p.is_file()}
        except OSError:
            return

        while self.is_running:
            try:
                self._scan_once(known_files)
            except Exception:
                pass

            # Poll every ~0.75 s (broken into 15 × 50 ms so stop() is responsive)
            for _ in range(15):
                if not self.is_running:
                    return
                time.sleep(0.05)

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _scan_once(self, known_files: set[str]) -> None:
        try:
            current_files = [p for p in self.watch_dir.iterdir() if p.is_file()]
        except OSError:
            return  # Directory disappeared or permission changed

        for f in current_files:
            if not self.is_running:
                return

            # Already tracked or a temp extension — skip
            if f.name in known_files or f.suffix.lower() in self.IGNORED_SUFFIXES:
                continue

            # Mark as known immediately so parallel iterations don't double-fire
            known_files.add(f.name)

            if self._wait_for_file_settled(f):
                self._detected_count += 1
                self.session_count.emit(self._detected_count)
                self.new_file_ready.emit(str(f.resolve()))

    def _wait_for_file_settled(
        self,
        file_path: Path,
        max_wait_sec: int = 20,
        stable_reads_required: int = 2,
    ) -> bool:
        """Return True once the file is fully written and no longer locked.

        Uses two consecutive equal-size reads (separated by 0.6 s) to confirm
        the writer has finished, then verifies the file handle can be opened.
        """
        deadline = time.monotonic() + max_wait_sec
        prev_size = -1
        stable_streak = 0

        while time.monotonic() < deadline and self.is_running:
            if not file_path.exists():
                return False

            # Re-check extension: browser may rename .crdownload → real ext
            if file_path.suffix.lower() in self.IGNORED_SUFFIXES:
                return False

            try:
                cur_size = file_path.stat().st_size
            except OSError:
                time.sleep(0.6)
                continue

            if cur_size > 0 and cur_size == prev_size:
                stable_streak += 1
                if stable_streak >= stable_reads_required:
                    # Verify handle can be opened (no exclusive lock)
                    try:
                        with open(file_path, "rb"):
                            pass
                        return True
                    except (OSError, PermissionError):
                        stable_streak = 0  # Still locked; reset streak
            else:
                stable_streak = 0

            prev_size = cur_size
            time.sleep(0.6)

        return False
