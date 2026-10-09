"""Local RAG search & document chat engine with streaming, hybrid search, and CPU-optimized retrieval."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Callable, Generator, List, Sequence, Set, Tuple
import numpy as np
import faiss

from ..config import Settings
from ..core.extractor import DocumentChunk, chunk_file
from .ollama_client import chat_stream, embed

# Common stopwords to ignore during keyword matching
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "shouldn't", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "wasn't", "we", "were", "weren't", "what",
    "when", "where", "which", "while", "who", "whom", "why", "with", "won't", "would", "wouldn't",
    "you", "your", "yours", "yourself", "yourselves"
}


def format_chat_export(history: list[dict], title: str = "Polaris Document Chat Transcript") -> str:
    """Exports chat session history into a cleanly formatted Markdown document."""
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {title}",
        f"*Exported on: {timestamp_str}*",
        "",
        "---",
        "",
    ]
    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"### 👤 User\n\n{content}\n")
        elif role == "assistant":
            lines.append(f"### 🤖 Polaris Assistant\n\n{content}\n")
            cited = msg.get("cited", [])
            if cited:
                lines.append("#### 📄 Sources & Citations")
                for c in cited:
                    lines.append(f"- **{c.file_name}** (Page {c.page}) — `{c.doc_path}`")
                lines.append("")
            lines.append("---\n")
    return "\n".join(lines)


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
            ".pdf", ".docx", ".doc", ".txt", ".md", ".rst", ".py", ".js", ".ts",
            ".html", ".css", ".json", ".csv", ".xml", ".yaml", ".yml", ".toml", ".ini", ".sql"
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

    def search(
        self,
        query: str,
        top_k: int = 3,
        file_types: Sequence[str] | Set[str] | None = None,
        hybrid: bool = True,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Retrieves top relevant chunks using Hybrid Search (BM25/Keyword + FAISS Vector)
        combined with Reciprocal Rank Fusion (RRF), with optional file-type filtering.
        """
        if not self.index or not self.chunks or not query.strip():
            return []

        # 1. Filter candidate chunk indices by file extension if requested
        candidate_indices = set(range(len(self.chunks)))
        if file_types:
            norm_exts = {ft.lower() if ft.startswith(".") else f".{ft.lower()}" for ft in file_types}
            candidate_indices = {i for i in candidate_indices if self.chunks[i].extension in norm_exts}
            if not candidate_indices:
                return []

        # 2. Vector search (FAISS IndexFlatIP Cosine Similarity)
        q_vec = np.array(embed(self.settings, [query]), dtype=np.float32)
        faiss.normalize_L2(q_vec)

        search_k = min(len(self.chunks), max(top_k * 5, 25))
        scores, indices = self.index.search(q_vec, search_k)

        vector_scores: dict[int, float] = {}
        vector_rank_map: dict[int, int] = {}
        rank_v = 1
        for idx, score in zip(indices[0], scores[0]):
            if idx in candidate_indices:
                vector_scores[idx] = float(score)
                vector_rank_map[idx] = rank_v
                rank_v += 1

        # 3. Keyword / Lexical Scoring
        kw_rank_map: dict[int, int] = {}
        query_tokens = [tok for tok in re.findall(r"\w+", query.lower()) if tok not in STOPWORDS and len(tok) >= 2]

        if hybrid and query_tokens:
            kw_scores: dict[int, float] = {}
            for idx in candidate_indices:
                chunk = self.chunks[idx]
                text_lower = chunk.text.lower()
                doc_name_lower = chunk.file_name.lower()

                score = 0.0
                for tok in query_tokens:
                    c_count = text_lower.count(tok)
                    if c_count > 0:
                        score += 1.0 + min(c_count, 5) * 0.2
                    if tok in doc_name_lower:
                        score += 2.0  # boost matches in filename

                if score > 0:
                    kw_scores[idx] = score

            if kw_scores:
                sorted_kw = sorted(kw_scores.items(), key=lambda x: x[1], reverse=True)
                for rank_kw, (idx, _) in enumerate(sorted_kw[:search_k], start=1):
                    kw_rank_map[idx] = rank_kw

        # 4. Reciprocal Rank Fusion (RRF)
        # RRF formula: RRF_score = sum(1.0 / (k_rrf + rank))
        k_rrf = 60
        combined_candidates = set(vector_rank_map.keys()) | set(kw_rank_map.keys())
        if not combined_candidates:
            # Fallback to candidates by vector scores if any
            combined_candidates = set(vector_scores.keys())

        scored_results: list[tuple[int, float]] = []
        for idx in combined_candidates:
            rrf_score = 0.0
            if idx in vector_rank_map:
                rrf_score += 1.0 / (k_rrf + vector_rank_map[idx])
            if idx in kw_rank_map:
                rrf_score += 1.0 / (k_rrf + kw_rank_map[idx])

            # Use vector score directly if hybrid produced nothing, otherwise RRF score
            final_score = rrf_score if (hybrid and (vector_rank_map or kw_rank_map)) else vector_scores.get(idx, 0.0)
            scored_results.append((idx, final_score))

        # Sort descending by score
        scored_results.sort(key=lambda item: item[1], reverse=True)

        results = []
        for idx, score in scored_results[:top_k]:
            results.append((self.chunks[idx], score))
        return results

    def chat_with_docs_stream(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int = 3,
        file_types: Sequence[str] | Set[str] | None = None,
        hybrid: bool = True,
    ) -> Generator[str, None, list[DocumentChunk]]:
        """
        Streams answers token-by-token.
        Returns the list of cited chunks at completion.
        """
        matched = self.search(question, top_k=top_k, file_types=file_types, hybrid=hybrid)
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
            f"[Source: {c.file_name} (Page {c.page})]\n{c.text}"
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
        file_types: Sequence[str] | Set[str] | None = None,
        hybrid: bool = True,
    ) -> tuple[str, list[DocumentChunk]]:
        gen = self.chat_with_docs_stream(question, history, top_k, file_types=file_types, hybrid=hybrid)
        tokens = []
        try:
            while True:
                tokens.append(next(gen))
        except StopIteration as e:
            cited = e.value or []
        return "".join(tokens), cited
