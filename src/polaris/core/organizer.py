"""Rule-based organization with preview, conflict-safe apply, and undo."""
from __future__ import annotations
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

CATEGORY_BY_EXT = {
    "Documents": {".pdf", ".docx", ".doc", ".txt", ".md", ".rtf", ".odt"},
    "Presentations": {".pptx", ".ppt", ".key"},
    "Spreadsheets": {".xlsx", ".xls", ".csv", ".tsv"},
    "Images": {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg", ".ico", ".psd"},
    "Videos": {".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".flv"},
    "Audio": {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"},
    "Archives": {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"},
    "Source Code": {".py", ".java", ".js", ".ts", ".c", ".cpp", ".cs", ".html", ".css", ".sql", ".json", ".xml", ".yaml", ".yml", ".ipynb"},
    "Creative Projects": {".aep", ".prproj", ".blend", ".unity", ".fig"},
}


@dataclass
class Move:
    src: str
    dst: str
    reason: str = ""


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
    """Never overwrite: add ' (1)', ' (2)'... until the path is free."""
    reserved = reserved or set()
    candidate, n = dst, 1
    while candidate.exists() or str(candidate) in reserved:
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


def apply_moves(conn: sqlite3.Connection, moves: list[Move]) -> str:
    """Execute approved moves. Records every operation. Returns batch_id."""
    batch = uuid.uuid4().hex[:12]
    for m in moves:
        src, dst = Path(m.src), Path(m.dst)
        try:
            if not src.exists():
                raise FileNotFoundError("source missing")
            dst = unique_destination(dst)  # re-check at execution time
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            conn.execute("INSERT INTO operations(batch_id,action,src,dst,status) VALUES(?,?,?,?,?)",
                         (batch, "move", str(src), str(dst), "done"))
        except OSError as e:
            conn.execute("INSERT INTO operations(batch_id,action,src,dst,status,error) VALUES(?,?,?,?,?,?)",
                         (batch, "move", str(src), str(dst), "failed", str(e)))
    conn.commit()
    return batch


def undo_batch(conn: sqlite3.Connection, batch_id: str) -> tuple[int, int]:
    """Reverse a batch (last move first). Returns (restored, failed)."""
    rows = conn.execute("SELECT id,src,dst FROM operations WHERE batch_id=? AND status='done' "
                        "ORDER BY id DESC", (batch_id,)).fetchall()
    ok = bad = 0
    for r in rows:
        src, dst = Path(r["src"]), Path(r["dst"])
        try:
            if not dst.exists() or src.exists():
                raise FileExistsError("cannot restore safely")
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst), str(src))
            conn.execute("UPDATE operations SET status='undone' WHERE id=?", (r["id"],))
            ok += 1
        except OSError as e:
            conn.execute("UPDATE operations SET status='undo_failed', error=? WHERE id=?", (str(e), r["id"]))
            bad += 1
    conn.commit()
    return ok, bad
