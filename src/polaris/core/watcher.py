"""Background download and file watcher service for Polaris."""
from __future__ import annotations

import time
from pathlib import Path
from PySide6.QtCore import QThread, Signal


class DownloadWatcherWorker(QThread):
    new_file_ready = Signal(str)  # Emits full path when file is finished writing

    def __init__(self, watch_dir: str | Path):
        super().__init__()
        self.watch_dir = Path(watch_dir)
        self.is_running = True
        self.ignored_suffixes = {
            ".tmp", ".crdownload", ".part", ".partial", ".download",
            ".opdownload", ".aria2", ".!ut", ".torrent"
        }

    def stop(self):
        self.is_running = False

    def run(self):
        if not self.watch_dir.is_dir():
            return

        # Initialize existing files
        known_files = {p.name for p in self.watch_dir.iterdir() if p.is_file()}

        while self.is_running:
            try:
                current_files = [p for p in self.watch_dir.iterdir() if p.is_file()]
                for f in current_files:
                    if not self.is_running:
                        break

                    # Check if it's a new file and not a temporary download file
                    if f.name not in known_files and f.suffix.lower() not in self.ignored_suffixes:
                        # Wait for download/write completion (size stability check)
                        if self._wait_for_file_settled(f):
                            known_files.add(f.name)
                            self.new_file_ready.emit(str(f.resolve()))
                        else:
                            # Still downloading, will retry next cycle
                            continue
            except Exception:
                pass

            # Sleep briefly before next scan cycle
            for _ in range(15):
                if not self.is_running:
                    break
                time.sleep(0.1)

    def _wait_for_file_settled(self, file_path: Path, max_wait_sec: int = 15) -> bool:
        """Checks if file size is stable and file lock is released."""
        start_time = time.time()
        last_size = -1

        while time.time() - start_time < max_wait_sec and self.is_running:
            if not file_path.exists():
                return False

            if file_path.suffix.lower() in self.ignored_suffixes:
                return False

            try:
                cur_size = file_path.stat().st_size
                if cur_size > 0 and cur_size == last_size:
                    # Test if file can be opened (no exclusive browser lock)
                    with open(file_path, "rb"):
                        pass
                    return True
                last_size = cur_size
            except (OSError, PermissionError):
                pass

            time.sleep(1.0)

        return False
