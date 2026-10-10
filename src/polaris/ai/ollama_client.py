"""Ollama client with real-time streaming, keep-alive warm cache, and CPU-optimized inference settings."""
from __future__ import annotations

import json
import os
from typing import Generator
import urllib.request
from urllib.error import URLError

from ..config import Settings


class OllamaError(RuntimeError):
    pass


def _get_optimal_threads(settings: Settings) -> int:
    """Returns the optimal number of CPU threads for inference."""
    if settings.num_threads > 0:
        return settings.num_threads
    try:
        import psutil  # type: ignore[import-untyped,import-not-found]
        cores = psutil.cpu_count(logical=False)
        if cores and cores > 0:
            return max(1, min(cores, 8))
    except Exception:
        pass
    cores = os.cpu_count() or 4
    return max(1, min(cores // 2 if cores > 2 else cores, 8))


def _format_error(e: Exception, url: str) -> str:
    err_str = str(e)
    if "10061" in err_str or "refused" in err_str.lower() or "connection" in err_str.lower():
        return f"Cannot connect to Ollama at {url}. Please ensure Ollama is running ('ollama serve')."
    if "timed out" in err_str.lower():
        return f"Ollama request timed out at {url}. The model or system may be under heavy load."
    return f"Ollama error: {err_str}"


def is_available(settings: Settings) -> bool:
    try:
        urllib.request.urlopen(settings.ollama_url + "/api/tags", timeout=2)
        return True
    except (URLError, TimeoutError):
        return False


def chat_stream(
    settings: Settings,
    messages: list[dict],
    options: dict | None = None,
    schema: dict | str | None = None,
) -> Generator[str, None, None]:
    """
    Streams tokens in real-time.
    Defaults are tuned for fast CPU inference (capped context & output length, keep_alive warm cache).
    """
    opts = {
        "temperature": 0.1,
        "num_predict": 400,   # Keep responses crisp on CPU
        "num_ctx": 2048,      # Context ceiling for fast evaluation
        "num_thread": _get_optimal_threads(settings),
    }
    if options:
        opts.update(options)

    payload = {
        "model": settings.chat_model,
        "messages": messages,
        "stream": True,
        "options": opts,
        "keep_alive": settings.ollama_keep_alive,
    }
    if schema:
        payload["format"] = schema

    req = urllib.request.Request(
        settings.ollama_url + "/api/chat",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                if line:
                    data = json.loads(line.decode("utf-8"))
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done", False):
                        break
    except Exception as e:
        raise OllamaError(_format_error(e, settings.ollama_url)) from e


def chat(
    settings: Settings,
    messages: list[dict],
    schema: dict | str | None = None,
    options: dict | None = None,
) -> str:
    """Non-streaming wrapper over chat_stream."""
    tokens = []
    for t in chat_stream(settings, messages, options=options, schema=schema):
        tokens.append(t)
    return "".join(tokens)


def embed(settings: Settings, texts: list[str]) -> list[list[float]]:
    payload = {
        "model": settings.embed_model,
        "input": texts,
        "keep_alive": settings.ollama_keep_alive,
    }
    req = urllib.request.Request(
        settings.ollama_url + "/api/embed",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["embeddings"]
    except Exception as e:
        raise OllamaError(_format_error(e, settings.ollama_url)) from e
