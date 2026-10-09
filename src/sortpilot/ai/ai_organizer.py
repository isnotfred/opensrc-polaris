"""AI-powered file organizer with CPU-optimized prompting and JSON parsing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..config import Settings
from ..core.organizer import Move, plan_by_type, unique_destination
from .ollama_client import chat
from .planner_schema import parse_model_json


def plan_with_ai(
    file_paths: list[str],
    dest_root: str,
    user_instruction: str = "",
    settings: Settings | None = None,
) -> list[Move]:
    """
    Uses Ollama to categorize files with compact prompts and CPU-friendly token budgets.
    """
    settings = settings or Settings()
    root = Path(dest_root)
    file_names = [Path(p).name for p in file_paths]

    if not file_names:
        return []

    # Use batches of 25 for quick CPU generation
    batch_size = 25
    all_moves: list[Move] = []
    reserved_destinations: set[str] = set()

    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i:i + batch_size]
        batch_names = [Path(p).name for p in batch_paths]

        if user_instruction.strip():
            instruction_text = f"USER RULE: {user_instruction.strip()}"
        else:
            instruction_text = "RULE: Group into concise subfolders by topic, date, or type (e.g. Documents, Invoices, Photos, Code)."

        prompt = (
            f"Organize these files into clean subfolders.\n"
            f"{instruction_text}\n\n"
            f"FILES:\n{json.dumps(batch_names)}\n\n"
            f'Output ONLY JSON:\n{{"moves": [{{"filename": "name.ext", "folder": "Subfolder", "reason": "short why"}}]}}'
        )

        try:
            resp_text = chat(
                settings,
                [
                    {"role": "system", "content": "You are a fast JSON-only file classifier. Be concise."},
                    {"role": "user", "content": prompt}
                ],
                options={"num_predict": 350, "num_ctx": 2048, "temperature": 0.1}
            )
            parsed = parse_model_json(resp_text)
            moves_data = parsed.get("moves", [])

            path_map = {Path(p).name: p for p in batch_paths}
            handled_names = set()

            for item in moves_data:
                fname = item.get("filename", "")
                folder = item.get("folder", "").strip("/\\ ")
                reason = item.get("reason", "AI categorized")

                if fname in path_map and folder:
                    src = Path(path_map[fname])
                    target_dir = root / folder
                    dst = unique_destination(target_dir / src.name, reserved_destinations)
                    reserved_destinations.add(str(dst))
                    all_moves.append(Move(src=str(src), dst=str(dst), reason=reason))
                    handled_names.add(fname)

            # Fallback for anything unhandled
            missing = [p for p in batch_paths if Path(p).name not in handled_names]
            if missing:
                rule_moves = plan_by_type(missing, dest_root)
                for rm in rule_moves:
                    dst = unique_destination(Path(rm.dst), reserved_destinations)
                    reserved_destinations.add(str(dst))
                    all_moves.append(Move(src=rm.src, dst=str(dst), reason=f"Fallback: {rm.reason}"))

        except Exception:
            fallback = plan_by_type(batch_paths, dest_root)
            for rm in fallback:
                dst = unique_destination(Path(rm.dst), reserved_destinations)
                reserved_destinations.add(str(dst))
                all_moves.append(Move(src=rm.src, dst=str(dst), reason=f"Rule-based (fallback): {rm.reason}"))

    return all_moves
