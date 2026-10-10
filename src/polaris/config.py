"""Central settings for Polaris AI."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    ollama_url: str = os.getenv("POLARIS_OLLAMA_URL", os.getenv("SORTPILOT_OLLAMA_URL", "http://localhost:11434"))
    chat_model: str = os.getenv("POLARIS_CHAT_MODEL", os.getenv("SORTPILOT_CHAT_MODEL", "qwen2.5:1.5b"))
    embed_model: str = os.getenv("POLARIS_EMBED_MODEL", os.getenv("SORTPILOT_EMBED_MODEL", "nomic-embed-text"))
    data_dir: Path = field(default_factory=lambda: Path(
        os.getenv("POLARIS_DATA", os.getenv("SORTPILOT_DATA", str(Path.home() / ".polaris")))))
    large_file_mb: int = 500
    chunk_chars: int = 800
    chunk_overlap: int = 100
    ollama_keep_alive: str = os.getenv("POLARIS_KEEP_ALIVE", "30m")
    num_threads: int = int(os.getenv("POLARIS_NUM_THREADS", "0"))  # 0 = auto-detect physical cores


    @property
    def db_path(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "polaris.db"
