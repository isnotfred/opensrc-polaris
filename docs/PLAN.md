# SortPilot AI: Local AI Assistant (Revised Scope)

Focused local-first desktop file assistant powered 100% on your machine via **Ollama**. No cloud calls, zero data leakage.

## Core Pillars

1. **📁 AI-Powered Organizing**:
   - Instruct the AI in plain English (*"Group my research papers by topic"*, *"Sort invoices by year & client"*, or smart auto-grouping).
   - Ollama (`llama3.2:3b`) analyzes files and proposes structured destination subfolders.
   - Deterministic safety layer prevents overwrites (`(1)`, `(2)`) and restricts moves to permitted paths.
   - Interactive preview table with individual checkboxes.
   - Batch transaction history with 1-click **Undo**.

2. **💬 AI Search & Document Chat (Chatbot)**:
   - Index local documents (PDF, DOCX, TXT, Markdown, Source Code).
   - Generate embeddings using Ollama's `nomic-embed-text` (768 dimensions) stored in a local FAISS vector index.
   - Grounded Q&A chatbot: ask questions across your documents.
   - Strict citation formatting: answers include verified source files and page numbers.

3. **📝 AI Document Summarizer**:
   - Select any document file (PDF, Word doc, Markdown, text, code).
   - Style presets:
     - **Key Takeaways & Action Items** (bulleted highlights)
     - **Executive Summary** (concise 2-3 paragraph overview)
     - **Detailed Notes** (structured section-by-section breakdown)
   - Map-reduce processing for long documents.
   - Formatted Markdown viewer with 1-click copy to clipboard.

---

## Architecture & Safety Rules

- **100% Local Inference**: Runs via Ollama at `http://localhost:11434`. Models: `llama3.2:3b` for chat/reasoning, `nomic-embed-text` for vector search.
- **Safety**: The LLM only proposes file moves. Deterministic code verifies paths, calculates unique destinations, and performs actual filesystem mutations.
- **Non-destructive & Undoable**: Every file organization batch is recorded in SQLite (`operations` table) and can be restored in reverse order with 1 click.
- **Responsive UI**: All text extraction, embedding generation, vector indexing, and LLM inference run in background `QThread` workers. The PySide6 UI never freezes.
