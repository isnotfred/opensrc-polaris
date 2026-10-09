"""Validates LLM-produced operation plans. The model proposes; this code decides."""
from __future__ import annotations
import json
import re
from pathlib import Path

# action -> (required keys, keys that hold paths)
ALLOWED_ACTIONS: dict[str, tuple[set[str], set[str]]] = {
    "move_files": ({"source_directory", "destination_directory"}, {"source_directory", "destination_directory"}),
    "find_duplicates": ({"source_directory"}, {"source_directory"}),
    "find_large_files": ({"source_directory"}, {"source_directory"}),
    "summarize_files": ({"source_directory"}, {"source_directory"}),
}
MUTATING = {"move_files"}


class PlanError(ValueError):
    pass


def parse_model_json(text: str) -> dict:
    """Tolerate code fences / chatter around the JSON; fail safely otherwise."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise PlanError("No JSON object in model output")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise PlanError(f"Malformed JSON: {e}") from e
    if not isinstance(obj, dict):
        raise PlanError("Plan must be a JSON object")
    return obj


def validate_plan(raw: str | dict, allowed_roots: list[Path], base: Path | None = None) -> dict:
    plan = parse_model_json(raw) if isinstance(raw, str) else dict(raw)
    action = plan.get("action")
    if action not in ALLOWED_ACTIONS:
        raise PlanError(f"Action not allowed: {action!r}")
    required, path_keys = ALLOWED_ACTIONS[action]
    missing = required - plan.keys()
    if missing:
        raise PlanError(f"Missing fields: {sorted(missing)}")
    base = (base or Path.home()).resolve()
    roots = [r.resolve() for r in allowed_roots]
    for key in path_keys:
        p = Path(str(plan[key])).expanduser()
        p = (p if p.is_absolute() else base / p).resolve()
        if not any(p == r or p.is_relative_to(r) for r in roots):
            raise PlanError(f"{key} is outside the folders you granted: {p}")
        plan[key] = str(p)
    plan["requires_confirmation"] = action in MUTATING  # never trust the model on this
    return plan
