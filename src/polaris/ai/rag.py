"""Local RAG search & document chat engine with streaming and CPU-optimized retrieval."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Generator, List, Tuple
import numpy as np
import faiss

from ..config import Settings
from ..core.extractor import DocumentChunk, chunk_file
from .ollama_client import chat_stream, embed


class RagEngine:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.chunks: list[DocumentChunk] = []
        self.index: faiss.IndexFlatIP | None = None
        self.indexed_folder: str = ""

    def index_folder(
        self,
        folder_path: str,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> int:
        folder = Path(folder_path)
        if not folder.is_dir():
            raise ValueError(f"Folder not found: {folder_path}")

        self.indexed_folder = str(folder)
        supported_exts = {
            ".pdf", ".docx", ".doc", ".txt", ".md", ".py", ".js", ".ts",
            ".html", ".css", ".json", ".csv", ".xml", ".yaml", ".yml", ".sql"
        }

        target_files = [
            f for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in supported_exts and not f.name.startswith(".")
        ]

        if not target_files:
            self.chunks = []
            self.index = None
            return 0

        # Chunk files with 800 chars for light and fast context
        all_chunks: list[DocumentChunk] = []
        for i, file_p in enumerate(target_files):
            if progress_cb:
                progress_cb(i, len(target_files), f"Reading {file_p.name}...")
            all_chunks.extend(chunk_file(file_p, chunk_chars=800, overlap=100))

        if not all_chunks:
            self.chunks = []
            self.index = None
            return 0

        # Generate embeddings in batches
        batch_size = 32
        embeddings_list = []
        total_chunks = len(all_chunks)

        for b_idx in range(0, total_chunks, batch_size):
            batch = all_chunks[b_idx:b_idx + batch_size]
            texts = [c.text for c in batch]
            if progress_cb:
                progress_cb(b_idx, total_chunks, f"Embedding chunks {b_idx + 1} to {min(b_idx + batch_size, total_chunks)} of {total_chunks}...")
            batch_vecs = embed(self.settings, texts)
            embeddings_list.extend(batch_vecs)

        embeddings_np = np.array(embeddings_list, dtype=np.float32)
        faiss.normalize_L2(embeddings_np)
        dim = embeddings_np.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings_np)
        self.chunks = all_chunks

        if progress_cb:
            progress_cb(total_chunks, total_chunks, f"Indexed {len(all_chunks)} chunks from {len(target_files)} files!")

        return len(all_chunks)

    def search(self, query: str, top_k: int = 3) -> list[tuple[DocumentChunk, float]]:
        if not self.index or not self.chunks or not query.strip():
            return []

        q_vec = np.array(embed(self.settings, [query]), dtype=np.float32)
        faiss.normalize_L2(q_vec)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(q_vec, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results

    def chat_with_docs_stream(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int = 3,
    ) -> Generator[str, None, list[DocumentChunk]]:
        """
        Streams answers token-by-token.
        Returns the list of cited chunks at completion.
        """
        matched = self.search(question, top_k=top_k)
        retrieved_chunks = [chunk for chunk, _score in matched]

        if not retrieved_chunks:
            messages = [
                {"role": "system", "content": "You are a helpful desktop assistant. Keep answers concise."},
                {"role": "user", "content": question}
            ]
            for token in chat_stream(self.settings, messages, options={"num_predict": 250, "num_ctx": 2048}):
                yield token
            return []

        # Keep context concise so the CPU can ingest it in < 1 second
        context_str = "\n\n".join([
            f"[Source: {Path(c.doc_path).name} (Page {c.page})]\n{c.text}"
            for c in retrieved_chunks
        ])

        system_prompt = (
            "You are Polaris AI. Answer the question accurately using ONLY the context below.\n"
            "Cite sources like [filename, Page X]. If the answer isn't in the context, say so.\n"
            "Keep the answer concise and direct.\n\n"
            f"CONTEXT:\n{context_str}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-2:])  # keep last 2 turns
        messages.append({"role": "user", "content": question})

        for token in chat_stream(self.settings, messages, options={"num_predict": 300, "num_ctx": 2048}):
            yield token

        return retrieved_chunks

    def chat_with_docs(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int = 3,
    ) -> tuple[str, list[DocumentChunk]]:
        gen = self.chat_with_docs_stream(question, history, top_k)
        tokens = []
        try:
            while True:
                tokens.append(next(gen))
        except StopIteration as e:
            cited = e.value or []
        return "".join(tokens), cited
