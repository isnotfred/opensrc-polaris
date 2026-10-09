"""Read-only scanner. Never modifies files. Deterministic: no AI involved."""
from __future__ import annotations
import hashlib
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

LARGE_FILE_BYTES = 500 * 1024 * 1024


@dataclass
class FileRecord:
    path: str
    name: str
    ext: str
    size: int
    mtime: float
    sha256: Optional[str] = None


@dataclass
class DuplicateGroup:
    sha256: str
    size: int
    files: list[FileRecord]

    @property
    def wasted_bytes(self) -> int:
        return self.size * (len(self.files) - 1)


@dataclass
class ScanResult:
    root: str
    files: list[FileRecord] = field(default_factory=list)
    duplicate_groups: list[DuplicateGroup] = field(default_factory=list)  # confirmed (hash)
    large_files: list[FileRecord] = field(default_factory=list)           # informational only
    empty_files: list[FileRecord] = field(default_factory=list)
    broken_links: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)           # (path, reason)
    total_size: int = 0
    by_ext: dict[str, dict] = field(default_factory=dict)                 # ext -> {count, bytes}
    duration_s: float = 0.0


def hash_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk_size):
            h.update(block)
    return h.hexdigest()


def scan(root: str, large_threshold: int = LARGE_FILE_BYTES,
         progress: Optional[Callable[[int], None]] = None) -> ScanResult:
    """Walk `root`, collect metadata, then hash only files that share a size."""
    t0 = time.perf_counter()
    root_p = Path(root).expanduser()
    if not root_p.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    res = ScanResult(root=str(root_p))

    def on_error(e: OSError) -> None:
        res.errors.append((str(e.filename), str(e)))

    for dirpath, _dirs, names in os.walk(root_p, onerror=on_error, followlinks=False):
        for name in names:
            p = os.path.join(dirpath, name)
            try:
                if os.path.islink(p):
                    if not os.path.exists(p):
                        res.broken_links.append(p)
                    continue
                st = os.stat(p)
            except OSError as e:
                res.errors.append((p, str(e)))
                continue
            ext = os.path.splitext(name)[1].lower()
            rec = FileRecord(p, name, ext, st.st_size, st.st_mtime)
            res.files.append(rec)
            res.total_size += rec.size
            bucket = res.by_ext.setdefault(ext or "(none)", {"count": 0, "bytes": 0})
            bucket["count"] += 1
            bucket["bytes"] += rec.size
            if rec.size == 0:
                res.empty_files.append(rec)
            elif rec.size >= large_threshold:
                res.large_files.append(rec)
            if progress and len(res.files) % 200 == 0:
                progress(len(res.files))

    # Exact duplicates: group by size first (cheap), hash only the candidates.
    by_size: dict[int, list[FileRecord]] = defaultdict(list)
    for r in res.files:
        if r.size > 0:
            by_size[r.size].append(r)
    for size, group in by_size.items():
        if len(group) < 2:
            continue
        by_hash: dict[str, list[FileRecord]] = defaultdict(list)
        for r in group:
            try:
                r.sha256 = hash_file(r.path)
                by_hash[r.sha256].append(r)
            except OSError as e:
                res.errors.append((r.path, str(e)))
        for digest, same in by_hash.items():
            if len(same) > 1:
                res.duplicate_groups.append(DuplicateGroup(digest, size, same))

    res.large_files.sort(key=lambda r: r.size, reverse=True)
    res.duplicate_groups.sort(key=lambda g: g.wasted_bytes, reverse=True)
    res.duration_s = time.perf_counter() - t0
    return res
