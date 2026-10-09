from pathlib import Path
import pytest
from sortpilot.core.scanner import scan
from sortpilot.core.organizer import plan_by_type, apply_moves, undo_batch
from sortpilot.db.database import connect
from sortpilot.ai.planner_schema import validate_plan, PlanError


def make(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.pdf").write_bytes(b"same content")
    (tmp_path / "sub" / "a_copy.pdf").write_bytes(b"same content")
    (tmp_path / "diff.pdf").write_bytes(b"same conten!")   # same size, different bytes
    (tmp_path / "empty.txt").write_bytes(b"")
    (tmp_path / "big.bin").write_bytes(b"x" * 5000)
    return tmp_path


def test_scan_finds_exact_duplicates_only(tmp_path):
    r = scan(str(make(tmp_path)), large_threshold=4000)
    assert len(r.duplicate_groups) == 1
    names = {Path(f.path).name for f in r.duplicate_groups[0].files}
    assert names == {"a.pdf", "a_copy.pdf"}
    assert [Path(f.path).name for f in r.large_files] == ["big.bin"]
    assert [Path(f.path).name for f in r.empty_files] == ["empty.txt"]


def test_scan_never_modifies(tmp_path):
    make(tmp_path)
    before = sorted((p.name, p.stat().st_mtime) for p in tmp_path.rglob("*") if p.is_file())
    scan(str(tmp_path))
    assert before == sorted((p.name, p.stat().st_mtime) for p in tmp_path.rglob("*") if p.is_file())


def test_organize_conflict_safe_and_undo(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    out = tmp_path / "out"; (out / "Documents").mkdir(parents=True)
    (out / "Documents" / "n.txt").write_text("existing")   # conflict target
    (src / "n.txt").write_text("new")
    conn = connect(tmp_path / "t.db")
    moves = plan_by_type([str(src / "n.txt")], str(out))
    assert moves[0].dst.endswith("n (1).txt")
    batch = apply_moves(conn, moves)
    assert (out / "Documents" / "n.txt").read_text() == "existing"   # not overwritten
    assert (out / "Documents" / "n (1).txt").read_text() == "new"
    assert undo_batch(conn, batch) == (1, 0)
    assert (src / "n.txt").read_text() == "new"


def test_planner_rejects_bad_plans(tmp_path):
    import json
    roots = [tmp_path]
    raw_json = json.dumps({
        "action": "move_files",
        "source_directory": str(tmp_path / "a"),
        "destination_directory": str(tmp_path / "b"),
        "requires_confirmation": False,
    })
    ok = validate_plan(f"```json\n{raw_json}\n```", roots)
    assert ok["requires_confirmation"] is True            # model cannot disable confirmation
    with pytest.raises(PlanError):
        validate_plan({"action": "run_shell", "cmd": "rm -rf /"}, roots)
    with pytest.raises(PlanError):
        validate_plan({"action": "find_duplicates", "source_directory": "/etc"}, roots)
    with pytest.raises(PlanError):
        validate_plan("not json at all", roots)
