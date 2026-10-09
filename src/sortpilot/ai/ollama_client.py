"""Ollama client with real-time streaming and CPU-optimized inference settings."""
from __future__ import annotations

import json
from typing import Generator
import urllib.request
from urllib.error import URLError

from ..config import Settings


class OllamaError(RuntimeError):
    pass


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
    schema: dict | None = None,
) -> Generator[str, None, None]:
    """
    Streams tokens in real-time.
    Defaults are tuned for fast CPU inference (capped context & output length).
    """
    opts = {
        "temperature": 0.1,
        "num_predict": 350,   # Keep responses crisp on CPU
        "num_ctx": 2048,       # 2k context prevents huge CPU memory overhead
    }
    if options:
        opts.update(options)

    payload = {
        "model": settings.chat_model,
        "messages": messages,
        "stream": True,
        "options": opts,
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
        raise OllamaError(str(e)) from e


def chat(
    settings: Settings,
    messages: list[dict],
    schema: dict | None = None,
    options: dict | None = None,
) -> str:
    """Non-streaming wrapper over chat_stream."""
    tokens = []
    for t in chat_stream(settings, messages, options=options, schema=schema):
        tokens.append(t)
    return "".join(tokens)


def embed(settings: Settings, texts: list[str]) -> list[list[float]]:
    req = urllib.request.Request(
        settings.ollama_url + "/api/embed",
        json.dumps({"model": settings.embed_model, "input": texts}).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["embeddings"]
    except (URLError, TimeoutError, json.JSONDecodeError) as e:
        raise OllamaError(str(e)) from e
