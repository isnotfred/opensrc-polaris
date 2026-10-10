"""Rule-based organization with preview, conflict-safe apply (move/copy), and undo."""
from __future__ import annotations

import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

CATEGORY_BY_EXT = {
    "Documents": {
        ".pdf", ".docx", ".doc", ".txt", ".md", ".rtf", ".odt",
        ".epub", ".pages", ".tex", ".log",
    },
    "Presentations": {".pptx", ".ppt", ".key"},
    "Spreadsheets": {".xlsx", ".xls", ".csv", ".ods", ".numbers", ".tsv"},
    "Images": {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg", ".ico",
        ".heic", ".heif", ".avif", ".tiff", ".tif", ".psd", ".ai", ".fig", ".raw",
    },
    "Audio & Video": {
        ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
        ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv",
    },
    "Archives": {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"},
    "Source Code": {
        ".py", ".java", ".js", ".ts", ".c", ".cpp", ".cs", ".html", ".css",
        ".sql", ".ipynb", ".go", ".rs", ".rb", ".php", ".sh", ".bat", ".ps1",
        ".toml", ".ini", ".cfg", ".env", ".yaml", ".yml", ".json", ".xml",
    },
    "Creative Projects": {".aep", ".prproj", ".blend", ".unity", ".fig"},
    "Databases": {".db", ".sqlite", ".sqlite3", ".mdb"},
    "Executables": {".exe", ".msi", ".dmg", ".pkg", ".deb", ".appimage"},
}


@dataclass
class Move:
    src: str
    dst: str
    reason: str = ""


@dataclass
class ApplyResult:
    """Returned by apply_moves; carries the batch ID and per-outcome counts."""
    batch_id: str
    succeeded: int
    failed: int

    def __str__(self) -> str:
        return self.batch_id


def category_for(ext: str) -> str:
    for cat, exts in CATEGORY_BY_EXT.items():
        if ext.lower() in exts:
            return cat
    return "Other"


def format_size_human(size_bytes: int) -> str:
    """Format bytes into concise human-readable strings."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def size_category_for(size_bytes: int) -> str:
    """Classify file size into Windows-safe category brackets."""
    if size_bytes < 1024 * 1024:
        return "Small (under 1MB)"
    elif size_bytes < 50 * 1024 * 1024:
        return "Medium (1MB-50MB)"
    else:
        return "Large (over 50MB)"


def unique_destination(dst: Path, reserved: set[str] | None = None) -> Path:
    """Never overwrite: add ' (1)', ' (2)'… until the path is free.

    Capped at 9999 to prevent an infinite loop in pathological cases.
    """
    reserved = reserved or set()
    candidate, n = dst, 1
    while (candidate.exists() or str(candidate) in reserved) and n <= 9999:
        candidate = dst.with_name(f"{dst.stem} ({n}){dst.suffix}")
        n += 1
    return candidate


def plan_by_type(file_paths: list[str], dest_root: str) -> list[Move]:
    """Preview only: returns proposed moves, touches nothing."""
    root, reserved, moves = Path(dest_root), set(), []
    for p in file_paths:
        src = Path(p)
        cat = category_for(src.suffix)
        if src.parent == root / cat:
            continue  # already in place
        dst = unique_destination(root / cat / src.name, reserved)
        reserved.add(str(dst))
        moves.append(Move(str(src), str(dst), f"{src.suffix or 'no extension'} -> {cat}"))
    return moves


def plan_by_type_and_size(file_paths: list[str], dest_root: str) -> list[Move]:
    """Preview only: categorizes files by format and size bracket."""
    root, reserved, moves = Path(dest_root), set(), []
    for p in file_paths:
        src = Path(p)
        cat = category_for(src.suffix)
        try:
            sz = src.stat().st_size
            sz_str = format_size_human(sz)
            sz_cat = size_category_for(sz)
        except OSError:
            sz_str = "unknown"
            sz_cat = "Small (under 1MB)"

        target_dir = root / cat / sz_cat
        if src.parent == target_dir:
            continue
        dst = unique_destination(target_dir / src.name, reserved)
        reserved.add(str(dst))
        moves.append(Move(str(src), str(dst), f"{cat} ({sz_str}) -> {cat}/{sz_cat}"))
    return moves


def apply_moves(
    conn: sqlite3.Connection,
    moves: list[Move],
    mode: str = "move",
) -> ApplyResult:
    """Execute approved file operations (mode='move' or 'copy').

    Records every operation in SQLite.
    Returns an ApplyResult with batch_id, succeeded count, and failed count.
    """
    batch = uuid.uuid4().hex[:12]
    action_type = "copy" if mode == "copy" else "move"
    succeeded = 0
    failed = 0

    for m in moves:
        src, dst = Path(m.src), Path(m.dst)
        try:
            if not src.exists():
                raise FileNotFoundError(f"source missing: {src}")
            dst = unique_destination(dst)  # re-check at execution time
            dst.parent.mkdir(parents=True, exist_ok=True)

            if action_type == "copy":
                shutil.copy2(str(src), str(dst))
            else:
                shutil.move(str(src), str(dst))

            conn.execute(
                "INSERT INTO operations(batch_id,action,src,dst,status) VALUES(?,?,?,?,?)",
                (batch, action_type, str(src), str(dst), "done"),
            )
            succeeded += 1
        except OSError as e:
            conn.execute(
                "INSERT INTO operations(batch_id,action,src,dst,status,error) VALUES(?,?,?,?,?,?)",
                (batch, action_type, str(src), str(dst), "failed", str(e)),
            )
            failed += 1

    conn.commit()
    return ApplyResult(batch_id=batch, succeeded=succeeded, failed=failed)


def undo_batch(conn: sqlite3.Connection, batch_id: str | ApplyResult) -> tuple[int, int]:
    """Reverse an operation batch (last operation first).

    For moves: restores the file back to src.
    For copies: removes the copied file at dst.
    Returns (restored, failed).
    """
    bid = batch_id.batch_id if hasattr(batch_id, "batch_id") else str(batch_id)
    rows = conn.execute(
        "SELECT id,action,src,dst FROM operations WHERE batch_id=? AND status='done' "
        "ORDER BY id DESC",
        (bid,),
    ).fetchall()
    ok = bad = 0
    for r in rows:
        action = r["action"]
        src, dst = Path(r["src"]), Path(r["dst"])
        try:
            if action == "copy":
                # Undo copy = delete the copy at dst (if it still exists)
                if dst.exists():
                    dst.unlink()
            else:
                # Undo move = restore from dst back to src.
                # Only abort if src already exists (would overwrite original work).
                if src.exists():
                    raise FileExistsError(
                        f"Cannot restore: a file already exists at the original path: {src}"
                    )
                if not dst.exists():
                    # File was already moved/deleted manually; skip gracefully.
                    conn.execute(
                        "UPDATE operations SET status='undone', error=? WHERE id=?",
                        ("source missing at dst — skipped", r["id"]),
                    )
                    ok += 1
                    continue
                src.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dst), str(src))

            conn.execute("UPDATE operations SET status='undone' WHERE id=?", (r["id"],))
            ok += 1
        except OSError as e:
            conn.execute(
                "UPDATE operations SET status='undo_failed', error=? WHERE id=?",
                (str(e), r["id"]),
            )
            bad += 1
    conn.commit()
    return ok, bad
