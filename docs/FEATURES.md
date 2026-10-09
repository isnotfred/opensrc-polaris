# SortPilot AI: Features & Capabilities Guide

SortPilot AI is a local-first, privacy-focused desktop assistant built with **PySide6** and powered 100% on-device by **Ollama**.

---

## 1. 📁 AI-Powered File Organizing

SortPilot organizes messy directories into structured folders using natural language instructions, while enforcing deterministic filesystem safeguards.

### Key Capabilities
* **Natural Language File Grouping**:
  * Give plain-English prompts like *"Organize my research papers by topic"*, *"Sort invoices and receipts by year & client"*, or *"Separate source code and docs"*.
  * Choose from one-click presets:
    * **Smart Auto-Group**: Groups files logically based on topic, project, and file type.
    * **Sort by Year & Date**: Identifies timestamps or dates in filenames to group by period.
    * **Sort by Project / Client**: Discovers project entities and consolidates matching files.
* **Deterministic Safety Layer**:
  * The LLM *only* proposes moves; it never directly touches files.
  * Destination conflicts are automatically resolved with collision-safe suffixes (`filename (1).ext`, `filename (2).ext`), ensuring files are **never overwritten**.
  * Paths are sandboxed to the target directory.
* **Interactive Preview Table**:
  * Displays proposed moves in a table before any action is executed.
  * Individual checkboxes allow you to selectively approve or exclude specific files.
  * Shows filename, proposed destination, reason, and original path.
* **1-Click Transactional Undo**:
  * Every applied move is logged in a local SQLite database (`operations` table).
  * The **History & Undo** dropdown lists past batches with item counts and timestamps.
  * Clicking **Undo Selected Batch** moves files back to their exact original locations in reverse order.
* **Rule-Based Fallback**:
  * A "Fast Rule-based (by Type)" button allows instant extension-based sorting (`Documents/`, `Images/`, `Source Code/`, `Archives/`, etc.) without running LLM inference.

---

## 2. 💬 AI Search & Document Chatbot

A grounded desktop search and question-answering engine that lets you chat with your local documents with verifiable citations.

### Key Capabilities
* **Multi-Format Document Ingestion**:
  * Extracts text and preserves page numbers across:
    * **PDFs** (via `PyMuPDF`)
    * **Word Documents** (`.docx` via `python-docx`)
    * **Plain Text & Markdown** (`.txt`, `.md`)
    * **Source Code & Data** (`.py`, `.js`, `.ts`, `.html`, `.css`, `.json`, `.csv`, `.sql`)
* **Vector Indexing with Local Embeddings**:
  * Breaks documents into 800-character overlapping chunks.
  * Computes 768-dimensional float32 vector embeddings using Ollama's `nomic-embed-text`.
  * Builds a local **FAISS `IndexFlatIP`** (inner product over normalized vectors for cosine similarity) in RAM.
  * Indexes dozens of files in just a few seconds.
* **Grounded Retrieval-Augmented Generation (RAG)**:
  * When you ask a question, SortPilot retrieves the top 3 most relevant chunks.
  * The LLM is instructed to answer strictly based on the provided context.
  * Source documents and page numbers are embedded in the answer (e.g., `📄 report.md (Page 1)`).
  * If the answer is not present in your files, the model explicitly states so rather than hallucinating.
* **Real-Time Token Streaming**:
  * Generates answers token-by-token directly into the chat view with sub-second initial response latency.
* **Chat Management**:
  * Conversation history tracking with context carryover across questions.
  * One-click **Clear Chat** button to reset session state.

---

## 3. 📝 AI Document Summarizer

An on-demand document analysis tool that generates crisp, structured summaries from complex files.

### Key Capabilities
* **Flexible Style Presets**:
  * **Key Takeaways & Action Items**: Generates 3–5 high-impact bullet points focusing on decisions, deliverables, and conclusions.
  * **Executive Summary**: Produces a polished 1–2 paragraph high-level overview.
  * **Detailed Notes**: Structured sectional breakdown highlighting key figures, dates, and conclusions.
* **CPU-Tuned Excerpting**:
  * Evaluates multi-page documents using prioritized beginning, middle, and conclusion excerpts, keeping prompt tokens under 2,000 for near-instant CPU processing.
* **Live Streaming Output**:
  * Summary text streams onto the screen in real-time as it is generated.
  * Automatically renders clean GitHub-flavored Markdown once complete.
* **1-Click Clipboard Export**:
  * **📋 Copy Summary to Clipboard** button transfers formatted plain text to your clipboard for easy pasting into emails, notes, or tickets.

---

## 4. 🧠 Adaptive Model Architecture & CPU Optimizations

SortPilot is engineered specifically to run efficiently on entry-level hardware (tested on an Intel 4-Core CPU with 8 GB RAM and integrated graphics).

### Optimization Techniques
1. **Dynamic Model Switcher**:
   * A dropdown widget in the top-right corner allows switching models on the fly without restarting:
     * **`qwen2.5:1.5b` (Default)**: Ultra-fast (~35–50 tok/s on CPU), uses ~1.2 GB RAM, exceptional JSON formatting reliability.
     * **`llama3.2:3b`**: Higher reasoning capacity (~10–15 tok/s on CPU), uses ~2.5 GB RAM.
2. **Context Window Capping (`num_ctx: 2048`)**:
   * Avoids Ollama's default 8K–128K context allocations, saving significant system RAM and eliminating prompt ingestion lag.
3. **Output Token Limits (`num_predict: 300–350`)**:
   * Ensures the model produces concise, high-density outputs without getting bogged down in verbose filler text.
4. **Non-Blocking Architecture**:
   * All long-running operations (file scanning, document extraction, embedding generation, vector indexing, and LLM inference) run inside dedicated PySide6 `QThread` workers. The user interface remains 100% fluid and responsive at all times.
