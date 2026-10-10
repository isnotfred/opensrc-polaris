"""Local RAG search & document chat engine with streaming, BM25 + FAISS hybrid search, incremental caching, HyDE query expansion, and CPU-optimized retrieval."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sqlite3
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

# Generic document, search command, and intent words that should not overshadow specific entities
GENERIC_QUERY_TERMS = {
    "report", "reports", "document", "documents", "doc", "docs", "file", "files",
    "text", "page", "pages", "summary", "summaries", "overview", "notes", "note",
    "record", "records", "data", "sheet", "sheets", "info", "information",
    "elaborate", "explain", "describe", "show", "tell", "summarize", "find",
    "search", "list", "give", "get", "detail", "details", "content", "contents",
    "section", "sections", "paragraph", "paragraphs"
}


def extract_substantive_query_terms(query: str) -> list[str]:
    """
    Extracts high-information entity/subject terms from a user query,
    filtering out stopwords and generic query/document command words
    (e.g., 'report', 'elaborate', 'summary', 'file').
    If the entire query consists of generic terms (e.g. 'show all reports'),
    falls back to all non-stopword tokens.
    """
    tokens = [w.lower() for w in re.findall(r"\w+", query) if len(w) >= 2]
    substantive = [w for w in tokens if w not in STOPWORDS and w not in GENERIC_QUERY_TERMS]
    if substantive:
        return substantive
    return [w for w in tokens if w not in STOPWORDS]

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


def clean_history_messages(history: list[dict] | None, max_turns: int = 2) -> list[dict]:
    """
    Extracts clean {'role', 'content'} message dicts from history,
    stripping internal metadata like DocumentChunk that cannot be serialized to JSON by Ollama.
    """
    if not history:
        return []
    slice_count = max_turns * 2 if max_turns > 0 else len(history)
    selected = history[-slice_count:]
    cleaned = []
    for m in selected:
        content = m.get("content", "")
        if content:
            cleaned.append({
                "role": str(m.get("role", "user")),
                "content": str(content),
            })
    return cleaned


def compress_history(
    history: list[dict],
    max_recent_turns: int = 2,
    max_summary_chars: int = 600,
) -> list[dict]:
    """
    Compresses older conversation history into a concise distillation
    to keep prompt token count well within CPU inference limits.
    Guarantees all output messages contain ONLY clean 'role' and 'content' string keys.
    """
    if not history:
        return []

    if len(history) <= max_recent_turns * 2:
        return clean_history_messages(history, max_turns=max_recent_turns)

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
    compressed.extend(clean_history_messages(recent, max_turns=max_recent_turns))
    return compressed


def cross_encoder_score(
    query: str,
    chunk: DocumentChunk,
    base_score: float = 0.0,
    q_words: list[str] | None = None,
    substantive_words: list[str] | None = None,
) -> float:
    """
    Computes a fast multi-signal cross-feature score between query and chunk:
    - Exact query phrase containment (+3.0)
    - Keyword overlap & coverage ratio (+2.5 * ratio)
    - Substantive entity match & distractor penalty
    - Section breadcrumb & filename token match (+1.5)
    - Window proximity of query terms (+1.0)
    - Base retrieval score prior (+1.0 * normalized_base)
    """
    if not query or not getattr(chunk, "text", ""):
        return 0.0

    q_lower = query.lower().strip()
    text_lower = chunk.text.lower()

    score = 0.0

    # 1. Exact phrase match
    if len(q_lower) > 3 and q_lower in text_lower:
        score += 3.0

    # 2. Token overlap & coverage
    words = q_words if q_words is not None else [w for w in re.findall(r"\w+", q_lower) if w not in STOPWORDS and len(w) >= 2]
    if words:
        text_words = set(re.findall(r"\w+", text_lower))
        matches = [w for w in words if w in text_words]
        coverage = len(matches) / len(words)
        score += 2.5 * coverage

        # 3. Section and file match
        sec_lower = getattr(chunk, "section", "").lower()
        file_lower = getattr(chunk, "file_name", "").lower()
        sec_matches = [w for w in words if w in sec_lower or w in file_lower]
        if sec_matches:
            score += 1.5 * (len(sec_matches) / len(words))

        # 4. Proximity density: if >= 2 words match, check distance between first and last match
        if len(matches) >= 2:
            indices = [text_lower.find(w) for w in matches if text_lower.find(w) != -1]
            if indices and (max(indices) - min(indices) <= 250):
                score += 1.0

        # 5. Substantive entity match & distractor penalty
        sub_terms = substantive_words if substantive_words is not None else extract_substantive_query_terms(query)
        if sub_terms and any(t not in GENERIC_QUERY_TERMS for t in sub_terms):
            sub_matches = [w for w in sub_terms if w in text_words or w in file_lower or w in sec_lower]
            if sub_matches:
                score += 3.0 * (len(sub_matches) / len(sub_terms))
            else:
                score *= 0.1

    # 6. Base score prior
    norm_base = min(1.0, max(0.0, float(base_score)))
    score += norm_base * 1.0

    return score


def llm_rerank_candidates(
    settings: Settings,
    query: str,
    candidates: list[tuple[DocumentChunk, float]],
    top_n: int = 10,
) -> dict[int, float]:
    """
    Prompts the local LLM to score the relevance of top candidates from 0 to 10.
    Returns a dict mapping candidate index (0-based) to normalized score (0.0 - 1.0).
    Falls back gracefully if LLM is unavailable or fails to return JSON.
    """
    if not candidates or not query.strip():
        return {}

    eval_candidates = candidates[:top_n]
    prompt_lines = [
        "Rate the relevance of each candidate text to the search query from 0 to 10.",
        f'Query: "{query}"',
        "",
        "Candidates:",
    ]
    for i, (chunk, _) in enumerate(eval_candidates, 1):
        clean_text = chunk.preview(max_chars=200).replace("\n", " ")
        prompt_lines.append(f"[{i}] {clean_text}")

    prompt_lines.append("")
    prompt_lines.append('Return STRICTLY a JSON list of objects with "id" and "score", e.g. [{"id": 1, "score": 9.0}, {"id": 2, "score": 2.5}]')
    prompt = "\n".join(prompt_lines)

    try:
        raw_resp = chat(
            settings,
            [{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 120, "num_ctx": 2048},
        )
        cleaned_json = re.sub(r"^```(?:json)?|```$", "", raw_resp.strip(), flags=re.MULTILINE).strip()
        data = json.loads(cleaned_json)
        scores: dict[int, float] = {}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "id" in item and "score" in item:
                    cand_idx = int(item["id"]) - 1
                    if 0 <= cand_idx < len(eval_candidates):
                        raw_score = float(item["score"])
                        scores[cand_idx] = max(0.0, min(1.0, raw_score / 10.0))
        return scores
    except Exception:
        return {}


def rerank_chunks(
    query: str,
    candidates: list[tuple[DocumentChunk, float]],
    top_k: int = 3,
    settings: Settings | None = None,
    use_llm: bool = False,
    rerank_pool_size: int = 10,
) -> list[tuple[DocumentChunk, float]]:
    """
    Re-ranks top candidate chunks using fast cross-scoring and optional local LLM evaluation.
    Sorts by final relevance score descending and returns the top_k entries.
    """
    if not candidates:
        return []

    pool = candidates[:rerank_pool_size]
    llm_scores: dict[int, float] = {}
    if use_llm and settings is not None:
        llm_scores = llm_rerank_candidates(settings, query, pool, top_n=rerank_pool_size)

    # Pre-tokenize query once for ultra-fast candidate scoring
    q_lower = query.lower().strip()
    q_words = [w for w in re.findall(r"\w+", q_lower) if w not in STOPWORDS and len(w) >= 2]
    sub_words = extract_substantive_query_terms(query)

    reranked: list[tuple[DocumentChunk, float]] = []
    for i, (chunk, base_score) in enumerate(pool):
        cross_score = cross_encoder_score(
            query,
            chunk,
            base_score=base_score,
            q_words=q_words,
            substantive_words=sub_words,
        )
        if i in llm_scores:
            final_score = (cross_score * 0.6) + (llm_scores[i] * 5.0 * 0.4)
        else:
            final_score = cross_score
        reranked.append((chunk, final_score))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]


def deduplicate_chunks(
    chunks_with_scores: list[tuple[DocumentChunk, float]],
    overlap_threshold: float = 0.70,
) -> list[tuple[DocumentChunk, float]]:
    """
    Suppresses redundant overlapping chunks from the same file to maximize
    context diversity and prevent wasting LLM prompt budget on repeated lines.
    """
    if not chunks_with_scores:
        return []

    unique_results: list[tuple[DocumentChunk, float]] = []
    seen_token_sets: list[tuple[str, set[str]]] = []

    for chunk, score in chunks_with_scores:
        chunk_tokens = set(re.findall(r"\w+", chunk.text.lower()))
        if not chunk_tokens:
            continue

        is_duplicate = False
        for prev_path, prev_tokens in seen_token_sets:
            if prev_path == chunk.doc_path:
                intersection = len(chunk_tokens & prev_tokens)
                union = len(chunk_tokens | prev_tokens)
                jaccard = intersection / union if union > 0 else 0.0
                if jaccard >= overlap_threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_results.append((chunk, score))
            seen_token_sets.append((chunk.doc_path, chunk_tokens))

    return unique_results


def assemble_prompt_context(
    chunks: list[DocumentChunk],
    max_context_chars: int = 4000,
) -> tuple[str, list[DocumentChunk]]:
    """
    Assembles grounded source context string, enforcing character/token limits
    to keep prompts well within CPU inference memory budgets and prevent context truncation.
    Returns (context_str, accepted_chunks).
    """
    if not chunks:
        return "", []

    accepted: list[DocumentChunk] = []
    context_parts: list[str] = []
    current_chars = 0

    for i, c in enumerate(chunks, 1):
        sec_label = f" | Section: {c.section}" if getattr(c, "section", "") else ""
        line_label = f" | Line: {c.start_line}" if getattr(c, "start_line", 1) > 1 else ""
        part = f"[Source {i}: {c.file_name}{sec_label}{line_label} (Page {c.page})]\nPath: {c.doc_path}\n{c.text}"
        part_len = len(part)

        if current_chars + part_len > max_context_chars and accepted:
            break

        context_parts.append(part)
        accepted.append(c)
        current_chars += part_len

    return "\n\n".join(context_parts), accepted


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
        self.filename_inverted_index: dict[str, set[int]] = {}
        self.idf: dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        parts = re.findall(r"[a-zA-Z0-9]+", text.lower())
        tokens = [p for p in parts if len(p) >= 2 and p not in STOPWORDS]
        compounds = re.findall(r"[a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)+", text.lower())
        for c in compounds:
            if len(c) >= 3 and c not in STOPWORDS:
                tokens.append(c)
        return tokens

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
            fn_lower = chunk.file_name.lower()
            self.doc_filenames.append(fn_lower)
            for ft in self.tokenize(fn_lower):
                if ft not in self.filename_inverted_index:
                    self.filename_inverted_index[ft] = set()
                self.filename_inverted_index[ft].add(doc_idx)

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

        sub_words = set(extract_substantive_query_terms(query))
        has_sub_words = len(sub_words) > 0 and any(w not in GENERIC_QUERY_TERMS for w in sub_words)

        for t in set(query_tokens):
            idf = self.idf.get(t, 0.0)
            if idf <= 0.0:
                idf = 0.1

            postings = self.inverted_index.get(t, [])
            term_weight = 1.5 if (has_sub_words and t in sub_words) else 1.0
            for doc_idx, tf in postings:
                if candidate_indices is not None and doc_idx not in candidate_indices:
                    continue

                dl = self.doc_len[doc_idx]
                tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (dl / avgdl)))
                scores[doc_idx] = scores.get(doc_idx, 0.0) + (idf * tf_norm * term_weight)

            # Boost matches occurring in the filename (instant O(1) set lookup)
            fn_boost = 2.5 if (has_sub_words and t in sub_words) else (0.2 if (has_sub_words and t in GENERIC_QUERY_TERMS) else 2.0)
            for doc_idx in self.filename_inverted_index.get(t, set()):
                if candidate_indices is None or doc_idx in candidate_indices:
                    scores[doc_idx] = scores.get(doc_idx, 0.0) + fn_boost

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


class SQLiteFTSIndex:
    """
    Persistent, disk-backed full-text search index powered by SQLite FTS5.
    Provides instant cold starts, zero Python RAM overhead for postings lists,
    and native BM25 ranking for large document collections.
    """
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = str(db_path) if db_path else ":memory:"
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                        chunk_id UNINDEXED,
                        doc_path UNINDEXED,
                        file_name,
                        section,
                        content,
                        tokenize='unicode61'
                    );
                """)
                conn.commit()
        except Exception:
            pass

    def build(self, chunks: list[DocumentChunk]):
        if not chunks:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM chunk_fts;")
                conn.executemany(
                    "INSERT INTO chunk_fts(chunk_id, doc_path, file_name, section, content) VALUES (?, ?, ?, ?, ?);",
                    [
                        (i, c.doc_path, c.file_name, getattr(c, "section", ""), c.text)
                        for i, c in enumerate(chunks)
                    ]
                )
                conn.commit()
        except Exception:
            pass

    def save_to_disk(self, disk_path: Path | str, chunks: list[DocumentChunk]):
        try:
            disk_fts = SQLiteFTSIndex(disk_path)
            disk_fts.build(chunks)
            self.db_path = str(disk_path)
        except Exception:
            pass

    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        tokens = [w for w in re.findall(r"\w+", query.lower()) if w not in STOPWORDS and len(w) >= 2]
        if not tokens:
            return []

        fts_query = " OR ".join(f'"{t}"' for t in tokens)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT chunk_id, bm25(chunk_fts) FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY rank LIMIT ?;",
                    (fts_query, top_k)
                )
                rows = cur.fetchall()
                results = []
                for cid, raw_score in rows:
                    pos_score = max(0.001, -float(raw_score))
                    results.append((int(cid), pos_score))
                return results
        except Exception:
            return []


class RagEngine:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.chunks: list[DocumentChunk] = []
        self.index: faiss.IndexFlatIP | None = None
        self.embeddings: np.ndarray | None = None
        self.bm25: BM25Index = BM25Index()
        self.fts: SQLiteFTSIndex = SQLiteFTSIndex()
        self.indexed_folder: str = ""
        self.indexed_files: list[str] = []

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

            # Persist SQLite FTS5 index to disk
            fts_file = cache_dir / "fts5.db"
            self.fts.save_to_disk(fts_file, self.chunks)

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

            fts_file = cache_dir / "fts5.db"
            if fts_file.is_file():
                self.fts = SQLiteFTSIndex(fts_file)
            else:
                self.fts = SQLiteFTSIndex()
                self.fts.build(chunks)

            return True
        except Exception:
            return False

    def clear_cache(self, folder_path: str) -> bool:
        """Removes the persistent index files for a folder."""
        cache_dir = self.get_cache_dir(folder_path)
        for fname in ("faiss.index", "embeddings.npy", "metadata.json", "fts5.db"):
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

        self.indexed_files = [f.name for f in target_files]

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
                self.fts.build(final_chunks)
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
            self.fts.build([])
            return 0

        self.bm25.build(all_chunks)
        self.fts.build(all_chunks)

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
        use_reranker: bool = False,
        use_llm_reranker: bool = False,
        rerank_pool_size: int = 10,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Retrieves top relevant chunks using Hybrid Search (Okapi BM25 + FAISS Vector)
        combined with Reciprocal Rank Fusion (RRF), optional file-type filtering,
        score thresholding, optional HyDE query expansion, and optional Cross-Encoder/LLM Re-ranking.
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

        # 5b. Substantive entity prioritization to demote false-positive distractors
        sub_words = extract_substantive_query_terms(query)
        if sub_words and any(w not in GENERIC_QUERY_TERMS for w in sub_words):
            sub_set = set(sub_words)
            def sub_match_priority(idx: int) -> int:
                c = self.chunks[idx]
                has_m = any(
                    w in c.text.lower()
                    or w in c.file_name.lower()
                    or w in getattr(c, "section", "").lower()
                    for w in sub_set
                )
                return 1 if has_m else 0

            has_sub_candidates = any(sub_match_priority(idx) == 1 for idx, _ in scored_candidates)
            if has_sub_candidates:
                scored_candidates = sorted(
                    scored_candidates,
                    key=lambda item: (sub_match_priority(item[0]), item[1]),
                    reverse=True
                )

        # 6. Optional Cross-Encoder / LLM Re-ranker
        if use_reranker:
            candidate_chunks = [(self.chunks[idx], score) for idx, score in scored_candidates]
            reranked = rerank_chunks(
                query,
                candidate_chunks,
                top_k=top_k,
                settings=self.settings,
                use_llm=use_llm_reranker,
                rerank_pool_size=rerank_pool_size,
            )
            filtered = [
                (chunk, score)
                for chunk, score in reranked
                if score >= score_threshold
            ]
            return filtered[:top_k]

        # 7. Standard score thresholding without re-ranking
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
        use_reranker: bool = False,
        use_llm_reranker: bool = False,
    ) -> Generator[str, None, list[DocumentChunk]]:
        """
        Streams answers token-by-token with grounded citations, context compression,
        optional HyDE expansion, and optional Re-ranking. Returns the list of cited chunks at completion.
        """
        matched = self.search(
            question,
            top_k=top_k,
            file_types=file_types,
            hybrid=hybrid,
            score_threshold=score_threshold,
            use_hyde=use_hyde,
            use_reranker=use_reranker,
            use_llm_reranker=use_llm_reranker,
        )

        # Prune zero-substantive distractors for prompt synthesis
        sub_words = extract_substantive_query_terms(question)
        if sub_words and any(w not in GENERIC_QUERY_TERMS for w in sub_words):
            sub_set = set(sub_words)
            has_sub = any(
                any(
                    w in chunk.text.lower()
                    or w in chunk.file_name.lower()
                    or w in getattr(chunk, "section", "").lower()
                    for w in sub_set
                )
                for chunk, _ in matched
            )
            if has_sub:
                matched = [
                    (chunk, score)
                    for chunk, score in matched
                    if any(
                        w in chunk.text.lower()
                        or w in chunk.file_name.lower()
                        or w in getattr(chunk, "section", "").lower()
                        for w in sub_set
                    )
                ]

        # Deduplicate overlapping chunks from same file
        deduped_matched = deduplicate_chunks(matched, overlap_threshold=0.70)
        cand_chunks = [chunk for chunk, _score in deduped_matched]

        # Assemble context within token/char budget (keeps prompt crisp on CPU)
        context_str, retrieved_chunks = assemble_prompt_context(cand_chunks, max_context_chars=4000)

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
            ]
            if history:
                messages.extend(
                    compress_history(history, max_recent_turns=2)
                    if compress_history_enabled
                    else clean_history_messages(history, max_turns=2)
                )
            messages.append({"role": "user", "content": question})

            for token in chat_stream(
                self.settings,
                messages,
                options={"num_predict": 450, "num_ctx": 2048, "temperature": 0.2},
            ):
                yield token
            return []

        system_prompt = (
            "You are Polaris AI, an intelligent, precise, and rigorously grounded local desktop document assistant.\n"
            f"{manifest_clause}\n"
            "STRICT GROUNDING & ENTITY INTEGRITY RULES:\n"
            "1. ENTITY INTEGRITY & NO CONFLATION: Never attribute facts, metrics, revenue, dates, or statements from one document "
            "to another company, person, or topic. Each [Source X] is an independent document. If a document belongs to 'Polaris Technologies' "
            "and the user asked about 'Delta Airlines', NEVER mix their data or claim that Delta Airlines had Polaris's revenue.\n"
            "2. DOCUMENT TYPE FIDELITY: If the retrieved documents do not contain the specific type of document requested "
            "(e.g., user asks for a 'report' about Delta Airlines, but only an 'expense receipt' exists), explicitly clarify what document "
            "actually exists (e.g., 'The indexed files contain an expense receipt for Delta Airlines, but no corporate financial report.').\n"
            "3. INLINE CITATIONS: Attribute factual statements with inline citations like [Source X] or [1] so every claim is verified.\n"
            "4. NO HALLUCINATIONS: If the context does not contain the answer, explicitly state that the indexed documents do not contain that information.\n"
            "5. Keep the answer structured, clear, and professional without cutting off.\n\n"
            f"DOCUMENT CONTEXT:\n{context_str}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(
                compress_history(history, max_recent_turns=2)
                if compress_history_enabled
                else clean_history_messages(history, max_turns=2)
            )
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
        file_types: Sequence[str] | Set[str] | None = None,
        hybrid: bool = True,
        score_threshold: float = 0.0,
        compress_history_enabled: bool = True,
        use_hyde: bool = False,
        use_reranker: bool = False,
        use_llm_reranker: bool = False,
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
            use_reranker=use_reranker,
            use_llm_reranker=use_llm_reranker,
        )
        tokens = []
        cited: list[DocumentChunk] = []
        try:
            while True:
                tokens.append(next(gen))
        except StopIteration as e:
            cited = e.value or []
        return "".join(tokens), cited

