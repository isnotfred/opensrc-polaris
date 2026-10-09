"""AI-powered file organizer with content-aware snippet inspection and CPU-optimized prompting."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Callable, List

from ..config import Settings
from ..core.extractor import extract_text_from_file
from ..core.organizer import Move, plan_by_type, unique_destination
from .ollama_client import chat
from .planner_schema import parse_model_json

logger = logging.getLogger(__name__)


def plan_with_ai(
    file_paths: list[str],
    dest_root: str,
    user_instruction: str = "",
    settings: Settings | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Move]:
    """
    Uses Ollama to categorize files with compact prompts, content snippets, and CPU-friendly token budgets.

    Args:
        progress_callback: optional callable(current_batch, total_batches) called before each batch.
    """
    settings = settings or Settings()
    root = Path(dest_root)

    if not file_paths:
        return []

    # Process in batches of 20 for responsive CPU inference
    batch_size = 20
    total_batches = max(1, (len(file_paths) + batch_size - 1) // batch_size)
    all_moves: list[Move] = []
    reserved_destinations: set[str] = set()

    for batch_num, i in enumerate(range(0, len(file_paths), batch_size), start=1):
        if progress_callback:
            progress_callback(batch_num, total_batches)
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

        # Few-shot example teaches the model the exact output shape to use.
        few_shot = '{"moves": [{"filename": "report_q3.pdf", "folder": "Finance", "reason": "quarterly report"}]}'
        prompt = (
            f"Organize these files into clean subfolders using their filenames and snippets.\n"
            f"{instruction_text}\n\n"
            f"FILES:\n{json.dumps(items_payload)}\n\n"
            f"Output ONLY a JSON object in exactly this shape (do NOT output a bare array):\n"
            f"{few_shot}"
        )

        try:
            resp_text = chat(
                settings,
                [
                    {"role": "system", "content": "You are a fast JSON-only file classifier. Output a JSON object with a 'moves' key. Be concise."},
                    {"role": "user", "content": prompt}
                ],
                options={"num_predict": 400, "num_ctx": 2048, "temperature": 0.1}
            )
            parsed = parse_model_json(resp_text)

            # Handle both {"moves": [...]} and a bare list (model sometimes ignores the wrapper)
            if isinstance(parsed, list):
                moves_data = parsed
            else:
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

        except Exception as exc:
            logger.warning(
                "plan_with_ai batch failed (falling back to rule-based): %s: %s",
                type(exc).__name__, exc,
            )
            fallback = plan_by_type(batch_paths, dest_root)
            for rm in fallback:
                dst = unique_destination(Path(rm.dst), reserved_destinations)
                reserved_destinations.add(str(dst))
                all_moves.append(Move(src=rm.src, dst=str(dst), reason=f"Rule-based (fallback): {rm.reason}"))

    return all_moves


# Timeout (seconds) for the Ollama round-trip in the watcher path.
# Keeps AnalyzeIncomingWorker from blocking indefinitely if the model is slow.
_WATCHER_AI_TIMEOUT_SEC = 15


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

    The Ollama call is wrapped in a threading timeout so this function always
    returns within ~15 s even if the model is busy or unreachable.
    """
    settings = settings or Settings()
    p = Path(file_path)
    if not p.exists():
        return {}

    try:
        file_size_bytes = p.stat().st_size
    except OSError:
        file_size_bytes = 0

    # --- Build content snippet ------------------------------------------------
    snippet = ""
    ext = p.suffix.lower()
    if ext in (".pdf", ".docx", ".doc", ".txt", ".md", ".py", ".json", ".csv", ".html"):
        try:
            pages = extract_text_from_file(p)
            if pages and pages[0][1].strip():
                snippet = " ".join(pages[0][1].split())[:180]
        except Exception:
            pass

    # --- Build prompt ---------------------------------------------------------
    # Use a few-shot example so small models don't echo the placeholder.
    # The example uses a completely different filename + folder so the model
    # learns the *pattern*, not the specific values.
    few_shot_example = (
        '{"suggested_filename": "quarterly_sales_report_q3.pdf", '
        '"suggested_folder": "Documents/Reports", '
        '"reason": "Financial report grouped with other reports"}'
    )

    prompt_lines = [
        f"File detected: {p.name}",
    ]
    if snippet:
        prompt_lines.append(f"Content preview: {snippet}")

    prompt_lines += [
        "",
        "Suggest a clean, descriptive filename for this specific file (keep the same extension)"
        " and the best subfolder to save it in (e.g. Documents/Reports, Photos/2024,"
        " Software/Installers, Work/Reviews, etc.).",
        "Output ONLY a JSON object. Do NOT use placeholder values.",
        f"Example output for a different file: {few_shot_example}",
        "Now output JSON for the file above:",
    ]
    prompt = "\n".join(prompt_lines)

    # --- Call Ollama with a hard timeout so we never hang --------------------
    result: dict = {}
    exc_holder: list[Exception] = []

    def _do_chat() -> None:
        try:
            resp = chat(
                settings,
                [
                    {
                        "role": "system",
                        "content": (
                            "You are an intelligent desktop file organizer. "
                            "Always output valid JSON only. "
                            "Never copy placeholder values like 'clean_name' — "
                            "always use the actual filename provided by the user."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                options={"num_predict": 150, "num_ctx": 2048, "temperature": 0.15},
            )
            parsed = parse_model_json(resp)
            new_name = parsed.get("suggested_filename", "").strip()
            folder = parsed.get("suggested_folder", "").strip("/\\ ")
            reason = parsed.get("reason", "Smart auto-placement").strip()

            # --- Sanity checks -----------------------------------------------
            # Reject literal placeholder names that small models sometimes echo
            _bad_stems = {"clean_name", "filename", "name", "file", "example"}
            if not new_name or Path(new_name).stem.lower() in _bad_stems:
                new_name = p.name  # Fall back to original

            # Reject placeholder folders
            if not folder or folder.lower() in {"subfolder", "folder", "path"}:
                from ..core.organizer import category_for
                folder = category_for(p.suffix)

            # Ensure extension isn't dropped by the model
            if not Path(new_name).suffix and p.suffix:
                new_name += p.suffix

            result.update({
                "original_path": str(p),
                "suggested_filename": new_name,
                "suggested_folder": folder,
                "reason": reason,
                "file_size_bytes": file_size_bytes,
            })
        except Exception as e:  # noqa: BLE001
            exc_holder.append(e)

    t = threading.Thread(target=_do_chat, daemon=True)
    t.start()
    t.join(timeout=_WATCHER_AI_TIMEOUT_SEC)

    if result:
        return result

    # Timed-out or errored — fall back to rule-based category
    from ..core.organizer import category_for
    cat = category_for(p.suffix)
    return {
        "original_path": str(p),
        "suggested_filename": p.name,
        "suggested_folder": cat,
        "reason": f"Quick suggestion ({cat}) — AI response timed out or unavailable",
        "file_size_bytes": file_size_bytes,
    }

