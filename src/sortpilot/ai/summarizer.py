"""Fast AI document summarizer with real-time streaming and CPU-optimized context."""
from __future__ import annotations

from pathlib import Path
from typing import Generator
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


def summarize_document_stream(
    file_path: Path | str,
    preset: str = "key_points",
    settings: Settings | None = None,
) -> Generator[str, None, None]:
    """
    Streams summary tokens in real-time.
    Uses a compact context window (~6,000 chars) for instant CPU processing.
    """
    settings = settings or Settings()
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    pages = extract_text_from_file(path)
    if not pages:
        yield "Could not extract text from this file (it may be empty, an image scan without OCR, or an unsupported format)."
        return

    # Condense document text to high-value excerpts (up to 6,000 characters) for fast CPU inference
    full_text = "\n\n".join([f"[Page {p}]\n{txt}" for p, txt in pages])
    if len(full_text) > 7000:
        # Take beginning, middle, and conclusion
        half = len(full_text) // 2
        full_text = (
            full_text[:3000]
            + "\n\n[... middle section ...]\n\n"
            + full_text[half:half + 1500]
            + "\n\n[... ending section ...]\n\n"
            + full_text[-2000:]
        )

    instruction = SUMMARY_PROMPTS.get(preset, SUMMARY_PROMPTS["key_points"])
    prompt = (
        f"Analyze this document ({path.name}) and perform this task: {instruction}\n"
        f"Keep the output direct, concise, and well-structured.\n\n"
        f"DOCUMENT:\n{full_text}"
    )

    messages = [
        {"role": "system", "content": "You are a concise, analytical executive assistant."},
        {"role": "user", "content": prompt}
    ]

    for token in chat_stream(settings, messages, options={"num_predict": 350, "num_ctx": 2048, "temperature": 0.1}):
        yield token


def summarize_document(
    file_path: Path | str,
    preset: str = "key_points",
    settings: Settings | None = None,
) -> str:
    tokens = []
    for t in summarize_document_stream(file_path, preset, settings):
        tokens.append(t)
    return "".join(tokens)
