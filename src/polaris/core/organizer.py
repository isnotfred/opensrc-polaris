"""Rule-based organization with preview, conflict-safe apply (move/copy), and undo."""
from __future__ import annotations

import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

CATEGORY_BY_EXT = {
    "Documents": {".pdf", ".docx", ".doc", ".txt", ".md", ".rtf", ".odt", ".pptx"},
    "Images": {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"},
    "Source Code": {".py", ".java", ".js", ".ts", ".c", ".cpp", ".cs", ".html", ".css", ".sql", ".ipynb"},
    "Archives": {".zip", ".rar", ".7z", ".tar", ".gz"},
    "Spreadsheets": {".xlsx", ".xls", ".csv"},
    "Audio & Video": {".mp3", ".wav", ".mp4", ".mkv", ".mov"},
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


def apply_moves(conn: sqlite3.Connection, moves: list[Move], mode: str = "move") -> str:
    """
    Execute approved file operations (mode='move' or 'copy').
    Records every operation in SQLite. Returns batch_id.
    """
    batch = uuid.uuid4().hex[:12]
    action_type = "copy" if mode == "copy" else "move"

    for m in moves:
        src, dst = Path(m.src), Path(m.dst)
        try:
            if not src.exists():
                raise FileNotFoundError("source missing")
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
        except OSError as e:
            conn.execute(
                "INSERT INTO operations(batch_id,action,src,dst,status,error) VALUES(?,?,?,?,?,?)",
                (batch, action_type, str(src), str(dst), "failed", str(e)),
            )
    conn.commit()
    return batch


def undo_batch(conn: sqlite3.Connection, batch_id: str) -> tuple[int, int]:
    """
    Reverse an operation batch (last operation first).
    For moves: restores file back to src.
    For copies: removes the copied file at dst.
    Returns (restored, failed).
    """
    rows = conn.execute(
        "SELECT id,action,src,dst FROM operations WHERE batch_id=? AND status='done' "
        "ORDER BY id DESC",
        (batch_id,),
    ).fetchall()
    ok = bad = 0
    for r in rows:
        action = r["action"]
        src, dst = Path(r["src"]), Path(r["dst"])
        try:
            if action == "copy":
                # For copy, undo means deleting the copied file
                if dst.exists():
                    dst.unlink()
            else:
                # For move, undo means restoring to original location
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
