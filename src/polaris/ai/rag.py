"""Local RAG search & document chat engine with streaming, BM25 + FAISS hybrid search, incremental caching, HyDE query expansion, and CPU-optimized retrieval."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Generator, List, Sequence, Set, Tuple
import numpy as np
import faiss

from ..config import Settings
from ..core.extractor import DocumentChunk, chunk_file, compute_file_hash
from .ollama_client import chat_stream, chat, embed

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

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".rst", ".rtf", ".log",
    ".html", ".htm", ".css", ".scss", ".js", ".jsx", ".ts", ".tsx",
    ".py", ".pyw", ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".cs", ".java",
    ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".kts", ".dart", ".lua",
    ".r", ".scala", ".pl", ".pm", ".sh", ".bash", ".zsh", ".bat", ".cmd", ".ps1",
    ".json", ".jsonl", ".csv", ".tsv", ".xml", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".sql", ".env"
}


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """
    Combines multiple ranked lists using Reciprocal Rank Fusion (RRF).
    Formula: RRF_score(d) = sum_m( w_m / (k + rank_m(d)) )
    Returns list of (doc_index, rrf_score) sorted descending by score.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    rrf_scores: dict[int, float] = {}
    for r_list, weight in zip(ranked_lists, weights):
        for rank, doc_idx in enumerate(r_list, start=1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (weight / (k + rank))

    sorted_scores = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_scores


def expand_query_terms(query: str, hypothetical_text: str, max_terms: int = 5) -> str:
    """
    Augments the original query with relevant high-signal keywords extracted
    from the generated hypothetical document snippet for BM25 lexical expansion.
    """
    if not hypothetical_text:
        return query

    query_tokens = set(re.findall(r"\w+", query.lower()))
    hyp_tokens = re.findall(r"\w+", hypothetical_text.lower())

    extra_tokens = []
    for tok in hyp_tokens:
        if len(tok) >= 3 and tok not in STOPWORDS and tok not in query_tokens:
            if tok not in extra_tokens:
                extra_tokens.append(tok)
                if len(extra_tokens) >= max_terms:
                    break

    if extra_tokens:
        return f"{query} {' '.join(extra_tokens)}"
    return query


def compress_history(
    history: list[dict],
    max_recent_turns: int = 2,
    max_summary_chars: int = 600,
) -> list[dict]:
    """
    Compresses older conversation history into a concise distillation
    to keep prompt token count well within CPU inference limits.
    """
    if not history or len(history) <= max_recent_turns * 2:
        return list(history)

    split_idx = len(history) - (max_recent_turns * 2)
    older = history[:split_idx]
    recent = history[split_idx:]

    summary_points: list[str] = []
    for msg in older:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        if not content:
            continue
        first_line = content.split("\n")[0]
        snippet = first_line[:120].strip()
        if role == "user":
            summary_points.append(f"User: {snippet}")
        elif role == "assistant":
            summary_points.append(f"AI: {snippet}")

    summary_text = " | ".join(summary_points)
    if len(summary_text) > max_summary_chars:
        summary_text = summary_text[:max_summary_chars].rstrip() + "..."

    compressed: list[dict] = [
        {"role": "system", "content": f"[Summary of earlier discussion: {summary_text}]"}
    ]
    compressed.extend(recent)
    return compressed


class BM25Index:
    """
    In-memory Okapi BM25 lexical index with inverted postings, document length
    normalization (k1=1.5, b=0.75), and filename boosting.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.avgdl: float = 0.0
        self.doc_len: list[int] = []
        self.doc_filenames: list[str] = []
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        self.idf: dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        tokens = re.findall(r"\w+", text.lower())
        return [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]

    def build(self, chunks: list[DocumentChunk]):
        self.corpus_size = len(chunks)
        self.doc_len = []
        self.doc_filenames = []
        self.inverted_index = {}
        self.idf = {}

        if not chunks:
            self.avgdl = 0.0
            return

        total_tokens = 0
        df: dict[str, int] = {}

        for doc_idx, chunk in enumerate(chunks):
            self.doc_filenames.append(chunk.file_name.lower())
            tokens = self.tokenize(chunk.text)
            n_tokens = len(tokens)
            self.doc_len.append(n_tokens)
            total_tokens += n_tokens

            tf_map: dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for t, count in tf_map.items():
                if t not in self.inverted_index:
                    self.inverted_index[t] = []
                self.inverted_index[t].append((doc_idx, count))
                df[t] = df.get(t, 0) + 1

        self.avgdl = (total_tokens / self.corpus_size) if self.corpus_size > 0 else 1.0

        for t, freq in df.items():
            self.idf[t] = float(np.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5)))

    def search(
        self,
        query: str,
        candidate_indices: set[int] | None = None,
        top_k: int = 50,
    ) -> list[tuple[int, float]]:
        if not self.corpus_size:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores: dict[int, float] = {}
        k1 = self.k1
        b = self.b
        avgdl = self.avgdl if self.avgdl > 0 else 1.0

        for t in set(query_tokens):
            idf = self.idf.get(t, 0.0)
            if idf <= 0.0:
                idf = 0.1

            postings = self.inverted_index.get(t, [])
            for doc_idx, tf in postings:
                if candidate_indices is not None and doc_idx not in candidate_indices:
                    continue

                dl = self.doc_len[doc_idx]
                tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (dl / avgdl)))
                scores[doc_idx] = scores.get(doc_idx, 0.0) + (idf * tf_norm)

            # Boost matches occurring in the filename
            for doc_idx in (candidate_indices if candidate_indices is not None else range(self.corpus_size)):
                if t in self.doc_filenames[doc_idx]:
                    scores[doc_idx] = scores.get(doc_idx, 0.0) + 2.0

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def format_chat_export(
    history: list[dict],
    title: str = "Polaris Document Chat Transcript",
    include_snippets: bool = True,
) -> str:
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
                    section_info = f" (§ {c.section})" if getattr(c, "section", "") else ""
                    lines.append(f"- **{c.file_name}**{section_info} (Page {c.page}) — `{c.doc_path}`")
                    if include_snippets and hasattr(c, "text") and c.text:
                        preview = c.text.strip().replace("\n", "\n> ")
                        lines.append(f"> {preview}\n")
                lines.append("")
            lines.append("---\n")
    return "\n".join(lines)


class RagEngine:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.chunks: list[DocumentChunk] = []
        self.index: faiss.IndexFlatIP | None = None
        self.embeddings: np.ndarray | None = None
        self.bm25: BM25Index = BM25Index()
        self.indexed_folder: str = ""

    def get_cache_dir(self, folder_path: str) -> Path:
        """Determines unique persistent cache directory for a target document folder."""
        norm_path = str(Path(folder_path).resolve())
        folder_hash = hashlib.sha256(norm_path.encode("utf-8")).hexdigest()[:16]
        folder_name = re.sub(r"[^\w\-]", "_", Path(folder_path).name) or "root"
        cache_dir = self.settings.data_dir / "rag_cache" / f"{folder_name}_{folder_hash}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def has_cache(self, folder_path: str) -> bool:
        """Checks whether a valid persistent index exists for the folder."""
        cache_dir = self.get_cache_dir(folder_path)
        meta_file = cache_dir / "metadata.json"
        index_file = cache_dir / "faiss.index"
        embed_file = cache_dir / "embeddings.npy"
        return meta_file.is_file() and index_file.is_file() and embed_file.is_file()

    def save_cache(self, folder_path: str | None = None) -> bool:
        """Saves current FAISS index, embeddings array, and metadata to disk."""
        target_folder = folder_path or self.indexed_folder
        if not target_folder or self.index is None or not self.chunks:
            return False

        cache_dir = self.get_cache_dir(target_folder)
        try:
            faiss.write_index(self.index, str(cache_dir / "faiss.index"))

            if self.embeddings is not None and len(self.embeddings) == len(self.chunks):
                np.save(str(cache_dir / "embeddings.npy"), self.embeddings)

            file_records: dict[str, dict] = {}
            for chunk in self.chunks:
                p_str = chunk.doc_path
                if p_str not in file_records:
                    p = Path(p_str)
                    stat = p.stat() if p.exists() else None
                    file_records[p_str] = {
                        "mtime": stat.st_mtime if stat else 0.0,
                        "size": stat.st_size if stat else 0,
                        "hash": compute_file_hash(p) if stat else "",
                        "chunks_count": 0,
                    }
                file_records[p_str]["chunks_count"] += 1

            meta = {
                "folder_path": target_folder,
                "indexed_at": datetime.now().isoformat(),
                "dim": int(self.index.d),
                "chunks": [c.to_dict() for c in self.chunks],
                "files": file_records,
            }
            (cache_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return True
        except Exception:
            return False

    def load_cache(self, folder_path: str) -> bool:
        """Loads persistent FAISS index, embeddings, and BM25 postings from disk."""
        if not self.has_cache(folder_path):
            return False

        cache_dir = self.get_cache_dir(folder_path)
        try:
            meta = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
            chunks = [DocumentChunk.from_dict(d) for d in meta.get("chunks", [])]
            index = faiss.read_index(str(cache_dir / "faiss.index"))
            embed_file = cache_dir / "embeddings.npy"
            embeddings = np.load(str(embed_file)) if embed_file.is_file() else None

            self.chunks = chunks
            self.index = index
            self.embeddings = embeddings
            self.indexed_folder = folder_path
            self.bm25.build(chunks)
            return True
        except Exception:
            return False

    def clear_cache(self, folder_path: str) -> bool:
        """Removes the persistent index files for a folder."""
        cache_dir = self.get_cache_dir(folder_path)
        for fname in ("faiss.index", "embeddings.npy", "metadata.json"):
            f = cache_dir / fname
            if f.is_file():
                try:
                    f.unlink()
                except Exception:
                    pass
        return True

    def generate_hypothetical_answer(self, query: str, max_tokens: int = 50) -> str:
        """
        Generates a 1-sentence hypothetical excerpt to perform HyDE
        (Hypothetical Document Embeddings) retrieval.
        """
        prompt = (
            "Write a brief 1-sentence direct factual excerpt as it would appear in a technical report answering:\n"
            f"Question: {query}\n"
            "Excerpt:"
        )
        messages = [
            {"role": "system", "content": "You are a concise factual document excerpt generator."},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = chat(
                self.settings,
                messages,
                options={"num_predict": max_tokens, "temperature": 0.2, "num_ctx": 1024},
            )
            return resp.strip().strip('"')
        except Exception:
            return ""

    def index_folder(
        self,
        folder_path: str,
        progress_cb: Callable[[int, int, str], None] | None = None,
        force_reindex: bool = False,
    ) -> int:
        """
        Indexes a folder with Incremental Indexing, Header-Aware Markdown/Code parsing,
        and Persistent Caching.
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise ValueError(f"Folder not found: {folder_path}")

        self.indexed_folder = str(folder)
        target_files = sorted([
            f for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS and not f.name.startswith(".")
        ], key=lambda p: str(p))

        if not target_files:
            self.chunks = []
            self.index = None
            self.embeddings = None
            self.bm25.build([])
            self.clear_cache(folder_path)
            return 0

        chunk_chars = getattr(self.settings, "chunk_chars", 800)
        chunk_overlap = getattr(self.settings, "chunk_overlap", 100)

        # Check incremental cache
        cache_dir = self.get_cache_dir(folder_path)
        cached_meta = None
        cached_embeddings = None
        if not force_reindex and self.has_cache(folder_path):
            try:
                cached_meta = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
                cached_embeddings = np.load(str(cache_dir / "embeddings.npy"))
            except Exception:
                cached_meta = None
                cached_embeddings = None

        if cached_meta and cached_embeddings is not None:
            cached_files: dict[str, dict] = cached_meta.get("files", {})
            raw_cached_chunks = [DocumentChunk.from_dict(d) for d in cached_meta.get("chunks", [])]

            file_chunks_map: dict[str, list[DocumentChunk]] = {}
            file_vecs_map: dict[str, list[np.ndarray]] = {}
            for chunk_idx, c in enumerate(raw_cached_chunks):
                file_chunks_map.setdefault(c.doc_path, []).append(c)
                if chunk_idx < len(cached_embeddings):
                    file_vecs_map.setdefault(c.doc_path, []).append(cached_embeddings[chunk_idx])

            unchanged_files: list[Path] = []
            modified_or_new_files: list[Path] = []

            for f in target_files:
                f_str = str(f)
                stat = f.stat()
                rec = cached_files.get(f_str)
                if rec and f_str in file_chunks_map:
                    if stat.st_mtime == rec.get("mtime") and stat.st_size == rec.get("size"):
                        unchanged_files.append(f)
                        continue
                    current_hash = compute_file_hash(f)
                    if current_hash and current_hash == rec.get("hash"):
                        unchanged_files.append(f)
                        continue
                modified_or_new_files.append(f)

            if not modified_or_new_files and len(unchanged_files) == len(target_files) and len(cached_files) == len(target_files):
                if progress_cb:
                    progress_cb(len(target_files), len(target_files), "✓ Up-to-date: loaded directly from cache!")
                self.load_cache(folder_path)
                return len(self.chunks)

            final_chunks: list[DocumentChunk] = []
            final_embeddings_list: list[np.ndarray] = []

            # Carry over unchanged
            for f in unchanged_files:
                f_str = str(f)
                final_chunks.extend(file_chunks_map.get(f_str, []))
                final_embeddings_list.extend(file_vecs_map.get(f_str, []))

            # Extract and embed modified/new files
            new_chunks: list[DocumentChunk] = []
            for i, f in enumerate(modified_or_new_files):
                if progress_cb:
                    progress_cb(i, len(modified_or_new_files), f"Reading modified {f.name}...")
                new_chunks.extend(chunk_file(f, chunk_chars=chunk_chars, overlap=chunk_overlap))

            if new_chunks:
                batch_size = 32
                for b_idx in range(0, len(new_chunks), batch_size):
                    batch = new_chunks[b_idx:b_idx + batch_size]
                    texts = [c.text for c in batch]
                    if progress_cb:
                        progress_cb(b_idx, len(new_chunks), f"Embedding new chunks {b_idx + 1}-{min(b_idx + batch_size, len(new_chunks))}...")
                    batch_vecs = embed(self.settings, texts)
                    for vec in batch_vecs:
                        final_embeddings_list.append(np.array(vec, dtype=np.float32))
                final_chunks.extend(new_chunks)

            for idx, c in enumerate(final_chunks):
                c.chunk_index = idx

            if final_chunks and final_embeddings_list:
                embeddings_np = np.vstack(final_embeddings_list).astype(np.float32)
                faiss.normalize_L2(embeddings_np)
                dim = embeddings_np.shape[1]
                self.index = faiss.IndexFlatIP(dim)
                self.index.add(embeddings_np)
                self.chunks = final_chunks
                self.embeddings = embeddings_np
                self.bm25.build(final_chunks)
                self.save_cache(folder_path)

                if progress_cb:
                    progress_cb(len(target_files), len(target_files), f"✓ Updated {len(modified_or_new_files)} modified file(s). Total: {len(final_chunks)} chunks.")
                return len(final_chunks)

        # Full index from scratch
        all_chunks: list[DocumentChunk] = []
        for i, file_p in enumerate(target_files):
            if progress_cb:
                progress_cb(i, len(target_files), f"Reading {file_p.name}...")
            all_chunks.extend(chunk_file(file_p, chunk_chars=chunk_chars, overlap=chunk_overlap))

        if not all_chunks:
            self.chunks = []
            self.index = None
            self.embeddings = None
            self.bm25.build([])
            return 0

        self.bm25.build(all_chunks)

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
        self.embeddings = embeddings_np

        self.save_cache(folder_path)

        if progress_cb:
            progress_cb(total_chunks, total_chunks, f"Indexed {len(all_chunks)} chunks from {len(target_files)} files!")

        return len(all_chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
        file_types: Sequence[str] | Set[str] | None = None,
        hybrid: bool = True,
        score_threshold: float = 0.0,
        use_hyde: bool = False,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Retrieves top relevant chunks using Hybrid Search (Okapi BM25 + FAISS Vector)
        combined with Reciprocal Rank Fusion (RRF), optional file-type filtering,
        score thresholding, and optional HyDE query expansion.
        """
        if not self.chunks or not query.strip():
            return []

        # 1. Filter candidate chunk indices by file extension if requested
        candidate_indices = set(range(len(self.chunks)))
        if file_types:
            norm_exts = {ft.lower() if ft.startswith(".") else f".{ft.lower()}" for ft in file_types}
            candidate_indices = {i for i in candidate_indices if self.chunks[i].extension in norm_exts}
            if not candidate_indices:
                return []

        search_k = min(len(self.chunks), max(top_k * 5, 25))

        # 2. HyDE (Hypothetical Document Embeddings) Expansion if requested
        hyp_text = ""
        if use_hyde:
            hyp_text = self.generate_hypothetical_answer(query)

        # 3. Vector search (FAISS IndexFlatIP Cosine Similarity)
        vector_ranked: list[int] = []
        vector_scores: dict[int, float] = {}

        if self.index is not None:
            try:
                if use_hyde and hyp_text:
                    vecs = embed(self.settings, [query, hyp_text])
                    q_vec1 = np.array(vecs[0], dtype=np.float32)
                    q_vec2 = np.array(vecs[1], dtype=np.float32)
                    fused_q = (0.5 * q_vec1) + (0.5 * q_vec2)
                    q_vec = np.expand_dims(fused_q, axis=0)
                else:
                    q_vec = np.array(embed(self.settings, [query]), dtype=np.float32)

                faiss.normalize_L2(q_vec)
                scores, indices = self.index.search(q_vec, search_k)
                for idx, score in zip(indices[0], scores[0]):
                    if idx in candidate_indices and idx != -1:
                        vector_ranked.append(int(idx))
                        vector_scores[int(idx)] = float(score)
            except Exception:
                pass

        # 4. BM25 Lexical search (with optional expanded query)
        bm25_ranked: list[int] = []
        if hybrid:
            bm25_query = expand_query_terms(query, hyp_text) if hyp_text else query
            bm25_results = self.bm25.search(bm25_query, candidate_indices=candidate_indices, top_k=search_k)
            bm25_ranked = [doc_idx for doc_idx, _score in bm25_results]

        # 5. Rank Fusion or Vector / BM25 fallback
        scored_candidates: list[tuple[int, float]] = []

        if hybrid and (vector_ranked or bm25_ranked):
            ranked_lists = []
            if vector_ranked:
                ranked_lists.append(vector_ranked)
            if bm25_ranked:
                ranked_lists.append(bm25_ranked)

            rrf_results = reciprocal_rank_fusion(ranked_lists, k=60)
            scored_candidates = rrf_results
        elif vector_ranked:
            scored_candidates = [(idx, vector_scores[idx]) for idx in vector_ranked]
        elif bm25_ranked:
            scored_candidates = [(idx, 1.0 / (rank + 1)) for rank, idx in enumerate(bm25_ranked)]

        # 6. Apply score thresholding
        filtered = [
            (self.chunks[idx], score)
            for idx, score in scored_candidates
            if score >= score_threshold
        ]

        return filtered[:top_k]

    def chat_with_docs_stream(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int = 3,
        file_types: Sequence[str] | Set[str] | None = None,
        hybrid: bool = True,
        score_threshold: float = 0.0,
        compress_history_enabled: bool = True,
        use_hyde: bool = False,
    ) -> Generator[str, None, list[DocumentChunk]]:
        """
        Streams answers token-by-token with grounded citations, context compression,
        and optional HyDE expansion. Returns the list of cited chunks at completion.
        """
        matched = self.search(
            question,
            top_k=top_k,
            file_types=file_types,
            hybrid=hybrid,
            score_threshold=score_threshold,
            use_hyde=use_hyde,
        )
        retrieved_chunks = [chunk for chunk, _score in matched]

        if not retrieved_chunks:
            messages = [
                {"role": "system", "content": "You are a helpful desktop assistant. Keep answers concise."},
            ]
            if history:
                messages.extend(compress_history(history, max_recent_turns=2) if compress_history_enabled else history[-2:])
            messages.append({"role": "user", "content": question})

            for token in chat_stream(self.settings, messages, options={"num_predict": 250, "num_ctx": 2048}):
                yield token
            return []

        # Keep context concise, structured, and section-grounded
        context_parts = []
        for i, c in enumerate(retrieved_chunks, 1):
            sec_label = f" | Section: {c.section}" if getattr(c, "section", "") else ""
            context_parts.append(
                f"[Source {i}: {c.file_name}{sec_label} (Page {c.page})]\nPath: {c.doc_path}\n{c.text}"
            )
        context_str = "\n\n".join(context_parts)

        system_prompt = (
            "You are Polaris AI, an intelligent and precise local desktop document assistant.\n"
            "Answer the question accurately using ONLY the context provided below.\n"
            "Cite sources using [filename, Page X] or [Source X]. If the answer is not in the context, "
            "explicitly state that the indexed documents do not contain that information.\n"
            "Keep the answer direct, structured, and factual.\n\n"
            f"CONTEXT:\n{context_str}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            if compress_history_enabled:
                messages.extend(compress_history(history, max_recent_turns=2))
            else:
                messages.extend(history[-2:])
        messages.append({"role": "user", "content": question})

        for token in chat_stream(self.settings, messages, options={"num_predict": 350, "num_ctx": 2048}):
            yield token

        return retrieved_chunks

    def chat_with_docs(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int = 3,
        file_types: Sequence[str] | Set[str] | None = None,
        hybrid: bool = True,
        score_threshold: float = 0.0,
        compress_history_enabled: bool = True,
        use_hyde: bool = False,
    ) -> tuple[str, list[DocumentChunk]]:
        gen = self.chat_with_docs_stream(
            question,
            history,
            top_k,
            file_types=file_types,
            hybrid=hybrid,
            score_threshold=score_threshold,
            compress_history_enabled=compress_history_enabled,
            use_hyde=use_hyde,
        )
        tokens = []
        try:
            while True:
                tokens.append(next(gen))
        except StopIteration as e:
            cited = e.value or []
        return "".join(tokens), cited
