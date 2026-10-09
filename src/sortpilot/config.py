"""Central settings. Models are config, not code: swap them after benchmarking."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    ollama_url: str = os.getenv("SORTPILOT_OLLAMA_URL", "http://localhost:11434")
    chat_model: str = os.getenv("SORTPILOT_CHAT_MODEL", "qwen2.5:1.5b")
    embed_model: str = os.getenv("SORTPILOT_EMBED_MODEL", "nomic-embed-text")
    data_dir: Path = field(default_factory=lambda: Path(
        os.getenv("SORTPILOT_DATA", str(Path.home() / ".sortpilot"))))
    large_file_mb: int = 500
    chunk_chars: int = 1200
    chunk_overlap: int = 150

    @property
    def db_path(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "sortpilot.db"
