"""Fast AI document summarizer with real-time streaming and CPU-optimized context."""
from __future__ import annotations

from pathlib import Path
from typing import Generator, Sequence
from ..config import Settings
from ..core.extractor import extract_text_from_file
from .ollama_client import chat_stream


SUMMARY_PROMPTS = {
    "key_points": (
        "Provide 3-5 concise bullet points highlighting key takeaways, findings, and any action items."
    ),
    "executive": (
        "Write a concise executive summary (1-2 paragraphs) of the main findings and conclusions."
    ),
    "detailed": (
        "Provide a structured summary highlighting the main ideas, key numbers, and conclusions."
    ),
}

# Human-readable labels exported so the UI can display them on the slider.
LENGTH_LABELS: dict[int, str] = {
    0: "Brief",
    1: "Moderate",
    2: "Comprehensive",
}

# Per-level inference + excerpt parameters.
# 'excerpt_limit' is the threshold above which trimming kicks in.
# 'begin'/'middle'/'end' are the char budgets for each excerpt section.
_LENGTH_PROFILES: dict[int, dict] = {
    0: {  # Brief — fastest, highest-density
        "num_predict": 400,
        "num_ctx": 3072,
        "excerpt_limit": 4500,
        "begin": 3500,
        "middle": 0,
        "end": 0,
        "suffix": (
            " Be concise and direct. Focus on key takeaways in crisp bullet points. "
            "Write complete sentences and do not leave thoughts unfinished."
        ),
    },
    1: {  # Moderate — balanced default
        "num_predict": 800,
        "num_ctx": 4096,
        "excerpt_limit": 8000,
        "begin": 3500,
        "middle": 2000,
        "end": 2500,
        "suffix": (
            " Keep the output well-structured, direct, and complete. "
            "Ensure all sections and sentences are fully written to the end."
        ),
    },
    2: {  # Comprehensive — thorough in-depth analysis
        "num_predict": 1400,
        "num_ctx": 4096,
        "excerpt_limit": 12000,
        "begin": 5000,
        "middle": 3000,
        "end": 4000,
        "suffix": (
            " Be thorough — include all significant points, key figures, dates, "
            "named entities, and full concluding synthesis. Finish all thoughts cleanly."
        ),
    },
}


def _build_excerpt(full_text: str, profile: dict) -> str:
    """Trim document text to the char budget defined by the length profile.

    If the text is already within the limit it is returned unchanged (no
    unnecessary string allocations).  Otherwise three optional windows —
    beginning, middle, and end — are stitched together.  Brief mode passes
    middle=0 / end=0 so only the opening region is used, which is the fastest
    and typically the most information-dense section.
    """
    if len(full_text) <= profile["excerpt_limit"]:
        return full_text

    parts: list[str] = [full_text[: profile["begin"]]]

    if profile["middle"]:
        mid = len(full_text) // 2
        parts.append(
            "\n\n[... middle section ...]\n\n"
            + full_text[mid : mid + profile["middle"]]
        )

    if profile["end"]:
        parts.append(
            "\n\n[... ending section ...]\n\n"
            + full_text[-profile["end"] :]
        )

    return "".join(parts)


def summarize_document_stream(
    file_path: Path | str,
    preset: str = "key_points",
    settings: Settings | None = None,
    length_level: int = 1,
) -> Generator[str, None, None]:
    """Stream summary tokens in real-time.

    Args:
        file_path:    Path to the document to summarise.
        preset:       Summary style — 'key_points', 'executive', or 'detailed'.
        settings:     Polaris Settings (uses defaults if None).
        length_level: Depth/length of the output.
                      0 = Brief  (fast, minimal tokens)
                      1 = Moderate  (default)
                      2 = Comprehensive  (thorough, accepts higher latency)
    """
    settings = settings or Settings()
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    pages = extract_text_from_file(path)
    if not pages:
        yield (
            "Could not extract text from this file "
            "(it may be empty, an image-only scan without OCR, or an unsupported format)."
        )
        return

    # Guard: clamp level to valid range so bad input never breaks the profile lookup
    level = max(0, min(length_level, 2))
    profile = _LENGTH_PROFILES[level]

    # Build document text from pages then trim to the excerpt budget
    full_text = "\n\n".join(f"[Page {p}]\n{txt}" for p, txt in pages)
    full_text = _build_excerpt(full_text, profile)

    # Combine format instruction (from preset) + depth instruction (from level)
    base_instruction = SUMMARY_PROMPTS.get(preset, SUMMARY_PROMPTS["key_points"])
    instruction = base_instruction + profile["suffix"]

    prompt = (
        f"Analyze this document ({path.name}) and perform this task: {instruction}\n\n"
        f"DOCUMENT:\n{full_text}"
    )

    messages = [
        {"role": "system", "content": "You are a concise, analytical executive assistant. Deliver complete, cohesive summaries without cutting off."},
        {"role": "user", "content": prompt},
    ]

    for token in chat_stream(
        settings,
        messages,
        options={
            "num_predict": profile["num_predict"],
            "num_ctx": profile["num_ctx"],
            "temperature": 0.1,
        },
    ):
        yield token


def summarize_document(
    file_path: Path | str,
    preset: str = "key_points",
    settings: Settings | None = None,
    length_level: int = 1,
) -> str:
    """Non-streaming wrapper — collects all tokens and returns the full summary."""
    tokens: list[str] = []
    for t in summarize_document_stream(file_path, preset, settings, length_level):
        tokens.append(t)
    return "".join(tokens)


# ---------------------------------------------------------------------------
# Entity & Action Items Extractor
# ---------------------------------------------------------------------------

# Structured extraction prompt with focused limits per category to guarantee
# complete JSON generation within token budgets.
_ENTITY_EXTRACT_PROMPT = (
    "You are a precise data-extraction assistant.\n"
    "Extract structured information from the document below and return ONLY a "
    "valid JSON object with exactly these four keys:\n\n"
    '  "action_items"      : list of key actionable tasks or to-dos (up to 8 items)\n'
    '  "deadlines"         : list of specific dates, due dates, or schedules (up to 8 items)\n'
    '  "financial_figures" : list of monetary amounts, budgets, or revenue numbers (up to 8 items)\n'
    '  "key_entities"      : list of important people, organisations, products, or places (up to 10 items)\n\n'
    "Rules:\n"
    "- Each list item must be a brief, self-contained string (max 15 words).\n"
    "- If a category has no items return an empty list [].\n"
    "- Return ONLY the JSON object — no explanation, no markdown fences.\n\n"
    "DOCUMENT ({filename}):\n"
    "{text}"
)

# Fallback JSON returned when text extraction yields nothing.
_EMPTY_ENTITIES = (
    '{"action_items":[],"deadlines":[],"financial_figures":[],"key_entities":[]}'
)


def extract_entities_stream(
    file_path: Path | str,
    settings: Settings | None = None,
) -> Generator[str, None, None]:
    """Stream JSON tokens for entity & figure extraction.

    The caller must accumulate all yielded tokens and parse the result as JSON
    when the generator is exhausted. Streaming is used so the UI can show a
    live progress indicator while the model runs.
    """
    settings = settings or Settings()
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    pages = extract_text_from_file(path)
    if not pages:
        yield _EMPTY_ENTITIES
        return

    full_text = "\n\n".join(txt for _, txt in pages)
    if len(full_text) > 6000:
        mid = len(full_text) // 2
        text_excerpt = (
            full_text[:2500]
            + "\n\n[... middle section ...]\n\n"
            + full_text[mid : mid + 1500]
            + "\n\n[... ending section ...]\n\n"
            + full_text[-2000:]
        )
    else:
        text_excerpt = full_text

    prompt = _ENTITY_EXTRACT_PROMPT.format(
        filename=path.name,
        text=text_excerpt,
    )

    messages = [
        {
            "role": "system",
            "content": "You are a precise data-extraction assistant. Output only valid JSON with all four keys.",
        },
        {"role": "user", "content": prompt},
    ]

    for token in chat_stream(
        settings,
        messages,
        options={
            "num_predict": 900,   # Generous token budget prevents truncated JSON
            "num_ctx": 4096,      # 4k context gives ample room for document + output
            "temperature": 0.0,   # deterministic — consistent extraction every run
        },
        schema="json",            # Ollama JSON mode — enforces JSON structure
    ):
        yield token


# ---------------------------------------------------------------------------
# Multi-Document Comparative Summary
# ---------------------------------------------------------------------------

# Inference options per length level for multi-doc mode.
_MULTI_DOC_PROFILES: dict[int, dict] = {
    0: {"num_predict": 650, "num_ctx": 4096, "chars_budget": 6500},
    1: {"num_predict": 1100, "num_ctx": 4096, "chars_budget": 10500},
    2: {"num_predict": 1600, "num_ctx": 6144, "chars_budget": 15000},
}

# Style instructions with strong comparative framing.
_MULTI_DOC_STYLE: dict[str, str] = {
    "key_points": (
        "Provide 4-8 crisp bullet points comparing and contrasting the documents. "
        "Highlight key points of agreement, divergence, and unique takeaways across all documents."
    ),
    "executive": (
        "Write an executive comparative synthesis (2-4 paragraphs). "
        "Contrast the main findings, highlight strategic implications, and synthesize collective conclusions."
    ),
    "detailed": (
        "Provide a detailed, structured comparative breakdown: cross-cutting themes, individual document "
        "contributions, notable metrics or figures, and overarching synthesis."
    ),
}


def _extract_doc_excerpt(path: Path, char_budget: int) -> str:
    """Extract text from one file and trim to char_budget characters.

    If the text exceeds char_budget, extracts a balanced sample: 65% from the
    beginning (introduction & core topics) and 35% from the end (conclusions,
    summaries, or closing numbers), ensuring the model understands the full document.
    """
    pages = extract_text_from_file(path)
    if not pages:
        return ""
    full_text = "\n\n".join(txt for _, txt in pages).strip()
    if len(full_text) <= char_budget:
        return full_text

    head_len = int(char_budget * 0.65)
    tail_len = max(120, char_budget - head_len)
    return (
        full_text[:head_len]
        + "\n\n[... content omitted for brevity ...]\n\n"
        + full_text[-tail_len:]
    )


def comparative_summary_stream(
    file_paths: Sequence[Path | str] | list[str],
    preset: str = "key_points",
    settings: Settings | None = None,
    length_level: int = 1,
) -> Generator[str, None, None]:
    """Stream a comparative summary of 2-5 documents in real-time.

    Args:
        file_paths:   List of 2-5 paths.  Raises ``ValueError`` if outside range.
        preset:       'key_points', 'executive', or 'detailed'.
        settings:     Polaris Settings object (defaults if None).
        length_level: 0 = Brief, 1 = Moderate, 2 = Comprehensive.
    """
    if not (2 <= len(file_paths) <= 5):
        raise ValueError(
            f"comparative_summary_stream requires 2-5 documents; got {len(file_paths)}."
        )

    settings = settings or Settings()
    level = max(0, min(length_level, 2))
    profile = _MULTI_DOC_PROFILES[level]

    num_docs = len(file_paths)
    # Distribute total char budget equally across documents (minimum 600 chars each).
    per_doc_budget = max(600, profile["chars_budget"] // num_docs)

    # Build labelled document sections & manifest
    doc_sections: list[str] = []
    skipped: list[str] = []
    doc_manifest: list[str] = []

    for i, fp in enumerate(file_paths, start=1):
        path = Path(fp)
        excerpt = _extract_doc_excerpt(path, per_doc_budget)
        if not excerpt:
            skipped.append(path.name)
            continue
        doc_manifest.append(f"{i}. {path.name}")
        doc_sections.append(
            f"=== Document {i}: {path.name} ===\n{excerpt}"
        )

    if not doc_sections:
        yield (
            "Could not extract text from any of the selected documents. "
            "Ensure the files are readable PDFs, Word documents, or text files."
        )
        return

    skip_note = ""
    if skipped:
        skip_note = (
            f"> **Note:** The following file(s) could not be read and were skipped: "
            + ", ".join(f"`{s}`" for s in skipped)
            + "\n\n"
        )

    docs_text = "\n\n".join(doc_sections)
    manifest_str = "\n".join(doc_manifest)
    names_summary = ", ".join(
        f"Document {i} ({Path(fp).name})"
        for i, fp in enumerate(file_paths, 1)
        if Path(fp).name not in skipped
    )
    style_instruction = _MULTI_DOC_STYLE.get(preset, _MULTI_DOC_STYLE["key_points"])
    suffix = _LENGTH_PROFILES[level]["suffix"]

    prompt = (
        f"You are comparing and synthesizing {len(doc_sections)} documents:\n"
        f"{manifest_str}\n\n"
        f"{docs_text}\n\n"
        f"MANDATORY COMPARATIVE INSTRUCTIONS:\n"
        f"You MUST examine, contrast, and reference ALL {len(doc_sections)} documents listed above ({names_summary}). "
        f"Do NOT stop after discussing only one or two documents.\n\n"
        f"Perform this comparative task: {style_instruction}{suffix}\n\n"
        f"Structure your response to cover:\n"
        f"1. Executive Synthesis & Key Differences (Cross-cutting comparison across all documents)\n"
        f"2. Individual Document Contributions (Explicitly cite each document by name: {names_summary})\n"
        f"3. Conclusions, Decisions & Key Takeaways\n"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior analyst skilled at synthesising insights across "
                "multiple documents. Be precise, thorough, and reference every document explicitly."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    # Yield the skip note first (if any) so it appears at top of the output
    if skip_note:
        yield skip_note

    for token in chat_stream(
        settings,
        messages,
        options={
            "num_predict": profile["num_predict"],
            "num_ctx": profile["num_ctx"],
            "temperature": 0.1,
        },
    ):
        yield token
