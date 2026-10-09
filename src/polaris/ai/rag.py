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
        self.indexed_files: list[str] = []

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

        self.indexed_files = [f.name for f in target_files]

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
        min_score: float = 0.22,
    ) -> Generator[str, None, list[DocumentChunk]]:
        """
        Streams answers token-by-token.
        Returns the list of cited chunks at completion.
        """
        matched = self.search(question, top_k=top_k)
        # Filter chunks by relevance score floor to eliminate hallucination from irrelevant files
        relevant_matches = [(chunk, score) for chunk, score in matched if score >= min_score]
        retrieved_chunks = [chunk for chunk, _ in relevant_matches]

        manifest_clause = ""
        if self.indexed_files:
            file_names_list = ", ".join(f"`{name}`" for name in self.indexed_files)
            manifest_clause = (
                f"\nFOLDER FILE MANIFEST:\n"
                f"The indexed folder contains exactly {len(self.indexed_files)} physical file(s): {file_names_list}.\n"
                f"URLs, links, or online profiles (e.g. LinkedIn, GitHub) mentioned inside a document's text are NOT files in the folder.\n"
            )

        if not retrieved_chunks:
            fallback_info = (
                f"The folder currently contains {len(self.indexed_files)} file(s): {', '.join(self.indexed_files)}."
                if self.indexed_files
                else "No relevant files or chunks were found in the folder."
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Polaris AI, an intelligent desktop assistant. "
                        f"{fallback_info} "
                        "Answer the question directly and concisely."
                    ),
                },
                {"role": "user", "content": question},
            ]
            for token in chat_stream(
                self.settings,
                messages,
                options={"num_predict": 450, "num_ctx": 2048, "temperature": 0.2},
            ):
                yield token
            return []

        # Build clean citation context
        context_str = "\n\n".join([
            f"[Source: {Path(c.doc_path).name} (Page {c.page})]\n{c.text}"
            for c in retrieved_chunks
        ])

        system_prompt = (
            "You are Polaris AI, an intelligent desktop assistant.\n"
            f"{manifest_clause}\n"
            "Answer the question accurately using the context below.\n"
            "Cite sources explicitly like [filename, Page X]. If the answer cannot be determined from the context, state that clearly.\n"
            "Keep the answer direct, well-structured, and complete without cutting off.\n\n"
            f"DOCUMENT CONTEXT:\n{context_str}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-2:])  # keep last 2 turns for context continuity
        messages.append({"role": "user", "content": question})

        for token in chat_stream(
            self.settings,
            messages,
            options={"num_predict": 650, "num_ctx": 3584, "temperature": 0.1},
        ):
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
        cited: list[DocumentChunk] = []
        try:
            while True:
                tokens.append(next(gen))
        except StopIteration as e:
            cited = e.value or []
        return "".join(tokens), cited

