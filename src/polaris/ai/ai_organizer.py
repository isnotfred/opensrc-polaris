"""AI-powered file organizer with CPU-optimized prompting, batching, and robust JSON schema parsing."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from ..config import Settings
from ..core.organizer import (
    Move,
    category_for,
    format_size_human,
    plan_by_type,
    plan_by_type_and_size,
    size_category_for,
    unique_destination,
)
from .ollama_client import chat
from .planner_schema import parse_model_json


def _sanitize_folder_component(name: str) -> str:
    """Sanitize folder names to prevent path traversal and Windows invalid character crashes."""
    # Disallow absolute paths and directory traversal
    parts = [p.strip() for p in re.split(r"[\\/]+", name) if p.strip() and p.strip() != ".."]
    if not parts:
        return "Organized_Files"

    cleaned_parts = []
    for part in parts:
        # Strip Windows reserved chars: < > : " / \ | ? *
        clean = re.sub(r'[<>:"/\\|?*]', '_', part)
        clean = clean.strip(". ")
        if clean:
            cleaned_parts.append(clean)

    return "/".join(cleaned_parts) if cleaned_parts else "Organized_Files"


def _extract_file_metadata(p: Path) -> dict:
    """Extract filesystem metadata safely for local LLM classification."""
    try:
        st = p.stat()
        sz = st.st_size
        sz_str = format_size_human(sz)
        sz_bracket = size_category_for(sz)
        from datetime import datetime
        mtime = datetime.fromtimestamp(st.st_mtime)
        year_str = str(mtime.year)
        date_str = mtime.strftime("%Y-%m-%d")
    except OSError:
        sz_str = "unknown"
        sz_bracket = "Small (under 1MB)"
        year_str = "unknown"
        date_str = "unknown"

    return {
        "filename": p.name,
        "extension": p.suffix.lower() if p.suffix else "none",
        "type_hint": category_for(p.suffix),
        "size": sz_str,
        "size_bracket": sz_bracket,
        "year": year_str,
        "date": date_str,
    }


def plan_with_ai(
    file_paths: list[str],
    dest_root: str,
    user_instruction: str = "",
    settings: Settings | None = None,
) -> list[Move]:
    """
    Uses Ollama to categorize files with rich file metadata, CPU-optimized prompts,
    robust JSON schema parsing, and strict adherence to user-defined multi-criteria sorting.
    """
    settings = settings or Settings()
    root = Path(dest_root)

    if not file_paths:
        return []

    # Use batches of 12 files: highly reliable for small local LLMs without token budget starvation
    batch_size = 12
    all_moves: list[Move] = []
    reserved_destinations: set[str] = set()

    norm_instruction = user_instruction.strip()
    is_size_requested = bool(
        norm_instruction
        and any(kw in norm_instruction.lower() for kw in ["size", "bracket", "small", "large", "mb", "kb"])
    )

    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i : i + batch_size]
        batch_meta = [_extract_file_metadata(Path(p)) for p in batch_paths]
        include_dates = not norm_instruction or any(
            kw in norm_instruction.lower() for kw in ["year", "date", "time", "month", "day", "chronolog"]
        )
        prompt_meta = []
        for m in batch_meta:
            item = {
                "filename": m["filename"],
                "extension": m["extension"],
                "type_hint": m["type_hint"],
                "size": m["size"],
                "size_bracket": m["size_bracket"],
            }
            if include_dates:
                item["year"] = m["year"]
                item["date"] = m["date"]
            prompt_meta.append(item)

        if norm_instruction:
            instruction_text = (
                f"STRICT USER INSTRUCTION:\n{norm_instruction}\n\n"
                f"You MUST strictly follow the user instruction above. Only use the criteria specified in the instruction."
            )
        else:
            instruction_text = (
                "AUTONOMOUS AI INSTRUCTION:\n"
                "Group files into clean, logical subfolders based on project, topic, date, or type (e.g. Documents, Invoices, Photos, Code)."
            )

        prompt = (
            f"You are an intelligent file organizer. Categorize the following files into subfolders.\n\n"
            f"{instruction_text}\n\n"
            f"ORGANIZATION RULES:\n"
            f"1. Multi-level hierarchy: If organizing by multiple criteria (such as 'file type, then size', 'by project, then date', or 'by category, then year'), "
            f"create nested subfolders using forward slashes (e.g. 'Documents/Small (under 1MB)', 'Images/Medium (1MB-50MB)', 'Project_Alpha/2026'). "
            f"Do NOT create extraneous unrequested subfolders.\n"
            f"2. Size grouping: If organizing by size, use the provided 'size_bracket' (e.g. 'Small (under 1MB)', 'Medium (1MB-50MB)', 'Large (over 50MB)'). "
            f"Do NOT use '<' or '>' characters because Windows file paths forbid them.\n"
            f"3. Reason: In the 'reason' field, concisely explain why this destination was chosen adhering to the rule (e.g. 'Word doc, 24 KB -> Documents/Small (under 1MB)').\n\n"
            f"FILES TO ORGANIZE:\n{json.dumps(prompt_meta, indent=2)}\n\n"
            f"Output ONLY valid JSON with this exact structure:\n"
            f'{{"moves": [{{"filename": "exact_name.ext", "folder": "Subfolder/OptionalNestedFolder", "reason": "concise explanation"}}]}}'
        )

        try:
            resp_text = chat(
                settings,
                [
                    {
                        "role": "system",
                        "content": "You are a fast, precise file classifier. Return only valid JSON adhering to the moves schema.",
                    },
                    {"role": "user", "content": prompt},
                ],
                schema="json",
                options={
                    "num_predict": 600,   # Ample token room for 12 files (prevents cutoffs)
                    "num_ctx": 2048,
                    "temperature": 0.1,
                },
            )
            parsed = parse_model_json(resp_text)
            moves_data = parsed.get("moves", [])

            # Exact path lookup and lowercase fallback lookup
            path_map = {Path(p).name: p for p in batch_paths}
            lower_map = {Path(p).name.lower().strip(): p for p in batch_paths}
            meta_map = {m["filename"]: m for m in batch_meta}
            meta_lower_map = {m["filename"].lower().strip(): m for m in batch_meta}
            handled_names = set()

            for item in moves_data:
                fname = item.get("filename", "")
                raw_folder = item.get("folder", "")
                reason = item.get("reason", "AI categorized")

                # Resolve file path with exact or case-insensitive fallback
                src_path_str = path_map.get(fname) or lower_map.get(fname.lower().strip())
                if src_path_str and raw_folder:
                    src = Path(src_path_str)
                    meta = meta_map.get(src.name) or meta_lower_map.get(src.name.lower().strip(), {})

                    # Guardrail: If user explicitly asked for size sorting and model omitted size subfolder, augment it
                    raw_folder_lower = raw_folder.lower()
                    has_size_in_folder = any(
                        kw in raw_folder_lower
                        for kw in ["small", "medium", "large", "mb", "kb", "under", "over", "byte"]
                    )
                    if is_size_requested and not has_size_in_folder and meta.get("size_bracket"):
                        raw_folder = f"{raw_folder}/{meta['size_bracket']}"
                        if "size" not in reason.lower():
                            reason = f"{reason} | Size: {meta.get('size', '')} ({meta['size_bracket']})"

                    folder = _sanitize_folder_component(raw_folder)
                    target_dir = root / folder
                    dst = unique_destination(target_dir / src.name, reserved_destinations)
                    reserved_destinations.add(str(dst))
                    all_moves.append(Move(src=str(src), dst=str(dst), reason=reason))
                    handled_names.add(src.name)

            # Rule-based fallback for any unhandled files in this batch
            missing = [p for p in batch_paths if Path(p).name not in handled_names]
            if missing:
                if is_size_requested:
                    rule_moves = plan_by_type_and_size(missing, dest_root)
                else:
                    rule_moves = plan_by_type(missing, dest_root)

                for rm in rule_moves:
                    dst = unique_destination(Path(rm.dst), reserved_destinations)
                    reserved_destinations.add(str(dst))
                    all_moves.append(
                        Move(src=rm.src, dst=str(dst), reason=f"Rule-based (unmatched): {rm.reason}")
                    )

        except Exception:
            # Fallback to deterministic rule-based sorting on any model error
            if is_size_requested:
                fallback = plan_by_type_and_size(batch_paths, dest_root)
            else:
                fallback = plan_by_type(batch_paths, dest_root)

            for rm in fallback:
                dst = unique_destination(Path(rm.dst), reserved_destinations)
                reserved_destinations.add(str(dst))
                all_moves.append(
                    Move(src=rm.src, dst=str(dst), reason=f"Rule-based (fallback): {rm.reason}")
                )

    return all_moves
