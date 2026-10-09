"""AI-powered file organizer with content-aware snippet inspection and CPU-optimized prompting."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..config import Settings
from ..core.extractor import extract_text_from_file
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
    Uses Ollama to categorize files with compact prompts, content snippets, and CPU-friendly token budgets.
    """
    settings = settings or Settings()
    root = Path(dest_root)

    if not file_paths:
        return []

    # Process in batches of 20 for responsive CPU inference
    batch_size = 20
    all_moves: list[Move] = []
    reserved_destinations: set[str] = set()

    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i:i + batch_size]
        items_payload = []

        # Content-aware inspection: grab first 120 chars for documents/text files
        for p_str in batch_paths:
            p = Path(p_str)
            item_info: dict[str, str] = {"filename": p.name}
            ext = p.suffix.lower()

            if ext in (".pdf", ".docx", ".doc", ".txt", ".md", ".py", ".json", ".csv", ".html", ".js", ".ts"):
                try:
                    pages = extract_text_from_file(p)
                    if pages and pages[0][1].strip():
                        # Clean whitespace and truncate
                        clean_snip = " ".join(pages[0][1].split())[:120]
                        item_info["snippet"] = clean_snip
                except Exception:
                    pass

            items_payload.append(item_info)

        if user_instruction.strip():
            instruction_text = f"USER RULE: {user_instruction.strip()}"
        else:
            instruction_text = "RULE: Group into concise subfolders by topic, date, or type (e.g. Documents, Invoices, Photos, Code)."

        prompt = (
            f"Organize these files into clean subfolders using their filenames and snippets.\n"
            f"{instruction_text}\n\n"
            f"FILES:\n{json.dumps(items_payload)}\n\n"
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


def suggest_single_file_placement(
    file_path: str | Path,
    dest_root: str | Path | None = None,
    settings: Settings | None = None,
) -> dict:
    """
    Analyzes an incoming single file and suggests:
    - Clean, standardized filename
    - Target subfolder
    - Reason
    """
    settings = settings or Settings()
    p = Path(file_path)
    if not p.exists():
        return {}

    snippet = ""
    ext = p.suffix.lower()
    if ext in (".pdf", ".docx", ".doc", ".txt", ".md", ".py", ".json", ".csv", ".html"):
        try:
            pages = extract_text_from_file(p)
            if pages and pages[0][1].strip():
                snippet = " ".join(pages[0][1].split())[:180]
        except Exception:
            pass

    prompt = (
        f"A user just downloaded or saved this file:\n"
        f"Original Name: {p.name}\n"
    )
    if snippet:
        prompt += f"Document Content Preview: {snippet}\n"

    prompt += (
        "\nProvide a clean standardized filename (keep the same extension) and a logical subfolder "
        "(e.g. Documents/Invoices, Photos/2024, Software/Code, etc.).\n"
        "Output ONLY JSON in this format:\n"
        '{"suggested_filename": "clean_name.ext", "suggested_folder": "Documents/Invoices", "reason": "why"}'
    )

    try:
        resp = chat(
            settings,
            [
                {"role": "system", "content": "You are an intelligent desktop file organizer. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            options={"num_predict": 250, "num_ctx": 2048, "temperature": 0.1}
        )
        parsed = parse_model_json(resp)
        new_name = parsed.get("suggested_filename", p.name).strip()
        folder = parsed.get("suggested_folder", "Organized").strip("/\\ ")
        reason = parsed.get("reason", "Smart auto-placement").strip()

        # Ensure extension isn't dropped by model
        if not Path(new_name).suffix and p.suffix:
            new_name += p.suffix

        return {
            "original_path": str(p),
            "suggested_filename": new_name,
            "suggested_folder": folder,
            "reason": reason,
        }
    except Exception:
        from ..core.organizer import category_for
        cat = category_for(p.suffix)
        return {
            "original_path": str(p),
            "suggested_filename": p.name,
            "suggested_folder": cat,
            "reason": f"Standard {cat} categorization",
        }

