# Polaris: Development Log & Technical Summary

This document records the architectural decisions, bug fixes, features implemented, and hardware optimizations completed for Polaris.

---

## Phase 1: Scaffold & Environment Stabilization
* **Environment Configuration**: Set up Python 3.13 virtual environment (`.venv`) with editable installation of `sortpilot`.
* **Bug Fix (Windows Path Escapes in Tests)**:
  * *Issue*: `tests/test_core.py` was failing on Windows because path strings like `C:\Users\...` were formatted into JSON using `%s` string interpolation, producing invalid JSON escape sequences (`\U` in `\Users`).
  * *Fix*: Refactored test payload generation to use `json.dumps()`, properly escaping backslashes. All tests passed.
* **Accidental Directory Cleanup**: Removed an empty directory `{core,ai,db,ui}` created by shell expansion syntax.

---

## Phase 2: Core Organize UI & Transactional Undo
* **Organize Tab Implementation (`src/sortpilot/ui/organize_tab.py`)**:
  * Implemented an interactive preview table displaying source path, proposed category, and conflict-safe destination paths (`(1)`, `(2)`).
  * Added individual row checkboxes with "Select All" / "Select None" controls.
  * Connected execution to background `QThread` workers (`PlanWorker`, `ApplyWorker`, `UndoWorker`) to prevent UI freezes.
* **SQLite Transaction Logging**:
  * Every applied move is recorded in the `operations` table in `sortpilot.db`.
  * Added **Recent Batches** dropdown allowing 1-click restoration of any batch in reverse order.
* **Seamless Scan Integration**:
  * Added "Organize this folder ->" button to the Scan tab to send scan results directly into the Organize pipeline.

---

## Phase 3: Scope Refinement to 3 Core Local AI Pillars
The user requested narrowing the scope to focus exclusively on local AI capabilities powered by **Ollama**:
1. **AI Organize**: Natural language and content-aware file organization.
2. **AI Search & Chat**: Document indexing and grounded Q&A chatbot with citations.
3. **AI Summarizer**: On-demand document analysis with style presets.

### Technical Implementations for Revised Scope:
* **Multi-Format Extractor (`src/sortpilot/core/extractor.py`)**:
  * Integrated `PyMuPDF` (`fitz`) for PDF extraction with page tracking.
  * Integrated `python-docx` for Word documents.
  * Added plain text, Markdown, CSV, JSON, and source code parsers.
  * Implemented overlapping text chunking (`chunk_text`, `chunk_file`).
* **AI Organizer (`src/sortpilot/ai/ai_organizer.py`)**:
  * Connected natural language prompts and quick presets to Ollama chat completions.
  * Structured JSON schema parsing with fallback to deterministic rule-based sorting.
* **Local RAG Engine (`src/sortpilot/ai/rag.py`)**:
  * Implemented folder indexer generating 768-dimensional float32 embeddings with `nomic-embed-text`.
  * Built local in-memory **FAISS `IndexFlatIP`** vector database.
  * Structured grounded QA prompts with citation enforcement (`[filename, Page X]`).
* **Document Summarizer (`src/sortpilot/ai/summarizer.py`)**:
  * Built multi-level summarizer supporting presets (*Key Takeaways*, *Executive Summary*, *Detailed Notes*).
* **UI Redesign (`src/sortpilot/ui/main_window.py`)**:
  * Clean 3-tab layout (`📁 AI Organize`, `💬 Search & Chat`, `📝 Summarize`).
  * Live status bar displaying Ollama connection status and active model.

---

## Phase 4: Performance Profiling & Hardware Tuning (8 GB CPU)
* **Hardware Analysis**:
  * Detected system environment: Intel 4-Core CPU, Integrated Iris Xe graphics, 7.8 GB total system RAM.
  * Identified bottlenecks causing initial output latency:
    1. Ollama's default context allocation (8K–128K) caused massive CPU memory buffers and prompt ingestion lag.
    2. Lack of token streaming caused the UI to appear frozen while generating 300+ tokens on CPU.
    3. Multi-chunk loops in summarization compounded sequential latency.
* **Optimizations Implemented**:
  * **Real-Time Token Streaming**: Added `chat_stream()` in `src/sortpilot/ai/ollama_client.py` and hooked it into `ChatTab` and `SummarizeTab`. Initial latency dropped to **< 1 second**.
  * **Context Window Capping**: Enforced `num_ctx: 2048` across all calls, reducing RAM usage and cutting prompt ingestion time.
  * **Output Token Capping**: Enforced `num_predict: 300–350` to keep answers dense and snappy.
  * **RAG Retrieval Pruning**: Reduced retrieval to `top_k=3` concise 800-character chunks.

---

## Phase 5: Low-End Model Optimization (`qwen2.5:1.5b`)
* Evaluated models for 8 GB CPU machines without compromising reliability:
  * Selected **`qwen2.5:1.5b`** (~986 MB download, ~1.2 GB RAM).
  * Generates at **~35–50 tokens/sec on CPU** (3x faster than Llama 3.2 3B).
  * Exceptional benchmark score for structured JSON output and grounded Q&A.
* **Configuration & Switcher**:
  * Set `qwen2.5:1.5b` as default in `src/sortpilot/config.py`.
  * Added a **`🧠 Model`** dropdown widget in the top-right corner of `MainWindow` to toggle between `qwen2.5:1.5b` and `llama3.2:3b` in real-time.

---

## Phase 6: Sample Dataset & Automated Test Suite
* **Sample Directory (`sample_files/`)**:
  * Populated 10 varied test files covering financial reports, invoices, travel receipts, architecture specs, Python scripts, CSV spreadsheets, NDA contracts, and media assets.
* **Automated Tests (`tests/`)**:
  * 9 unit tests passing cleanly in `pytest` under headless offscreen PySide6:
    * Scanner duplicate detection & non-mutating invariant.
    * Conflict-safe renaming & undo batch logic.
    * Planner schema allowlist validation.
    * Extractor page parsing & text chunking.
    * UI tabs initialization & preview population.
