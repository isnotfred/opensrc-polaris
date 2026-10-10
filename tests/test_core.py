from pathlib import Path
import pytest
from polaris.core.scanner import scan
from polaris.core.organizer import plan_by_type, apply_moves, undo_batch
from polaris.db.database import connect
from polaris.ai.planner_schema import validate_plan, PlanError


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
    result = apply_moves(conn, moves)
    assert result.succeeded == 1
    assert result.failed == 0
    assert (out / "Documents" / "n.txt").read_text() == "existing"   # not overwritten
    assert (out / "Documents" / "n (1).txt").read_text() == "new"
    assert undo_batch(conn, result.batch_id) == (1, 0)
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


def test_plan_by_type_and_size(tmp_path):
    from polaris.core.organizer import plan_by_type_and_size
    src = tmp_path / "files"
    src.mkdir()
    doc = src / "doc.docx"
    doc.write_text("small document")
    img = src / "pic.png"
    img.write_bytes(b"x" * (1024 * 1024 * 2))  # 2MB medium file

    out = tmp_path / "organized"
    moves = plan_by_type_and_size([str(doc), str(img)], str(out))
    assert len(moves) == 2

    # docx should go to Documents/Small (under 1MB)
    assert any("Documents" in m.dst and "Small (under 1MB)" in m.dst and "doc.docx" in m.dst for m in moves)
    # png should go to Images/Medium (1MB-50MB)
    assert any("Images" in m.dst and "Medium (1MB-50MB)" in m.dst and "pic.png" in m.dst for m in moves)

