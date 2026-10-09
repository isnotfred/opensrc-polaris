# Polaris AI: Team Roles & Simultaneous Collaboration Guide

This guide establishes the division of responsibilities for a **3-person development team** to build features in parallel with **zero git merge conflicts**.

---

## 🛡️ Core Principles for Conflict-Free Development

1. **Strict File Ownership**: Each developer works exclusively in their assigned UI and backend files.
2. **Dedicated Test Files**: Developers do not share a single test file. Each role maintains their own test suite (`test_organize.py`, `test_rag.py`, `test_summarize.py`).
3. **No Direct Commits to `main`**: All work occurs on dedicated feature branches (`feature/ai-organize`, `feature/search-chat`, `feature/summarizer`).
4. **Frozen Shared Files**: Core scaffolding files (`main_window.py`, `config.py`, `ollama_client.py`) remain untouched unless agreed upon by the entire team.

---

## 👥 Architecture & Role Breakdown

```
                                  [ Polaris Application ]
                                             │
             ┌───────────────────────────────┼───────────────────────────────┐
             ▼                               ▼                               ▼
       [ ROLE 1 ]                       [ ROLE 2 ]                       [ ROLE 3 ]
    AI Organize Lead                Search & Chat Lead                Summarizer Lead
  📁 ai_organize_tab.py             💬 chat_tab.py                  📝 summarize_tab.py
  ⚙️ ai_organizer.py                ⚙️ rag.py                       ⚙️ summarizer.py
  🧪 test_organize.py               🧪 test_rag.py                  🧪 test_summarize.py
```

---

### 👤 Role 1: AI Organize & Filesystem Specialist
> **Mission**: Enhance intelligent file sorting, preview table ergonomics, custom rule configuration, and safe filesystem operations.

* **Exclusively Owned Files (Only Role 1 edits these):**
  * `src/polaris/ui/ai_organize_tab.py` *(UI layout, interactive preview table, buttons, progress bar)*
  * `src/polaris/ai/ai_organizer.py` *(Classification prompts, JSON output formatting)*
  * `src/polaris/core/organizer.py` *(Path sandboxing, conflict-safe rename logic `(1)`, `(2)`)*
  * `tests/test_organize.py` *(Dedicated test file for organization logic and UI)*

* **Recommended Backlog Tasks:**
  - [ ] **Drag & Drop Folder Input**: Allow users to drag a folder directly onto the source folder input field.
  - [ ] **Custom Rules Builder**: UI modal allowing users to save manual sorting rules (e.g., *"Move all `.raw` and `.cr2` files into Photos/Raw"*).
  - [ ] **Preview Table Search/Filter**: A search box above the preview table to filter proposed moves before execution.
  - [ ] **File Size & Metadata Columns**: Add file size and last modified date columns to the preview table.

---

### 👤 Role 2: Search & Document Chat (RAG) Specialist
> **Mission**: Advance document text extraction, vector database indexing (embeddings + FAISS), semantic retrieval, and grounded chatbot Q&A.

* **Exclusively Owned Files (Only Role 2 edits these):**
  * `src/polaris/ui/chat_tab.py` *(Chat conversation display, real-time token streaming, citation badges)*
  * `src/polaris/ai/rag.py` *(FAISS index management, embedding generation, prompt context assembly)*
  * `src/polaris/core/extractor.py` *(Document text extraction for PDF, DOCX, TXT, MD, Code & text chunking)*
  * `tests/test_rag.py` *(Dedicated test file for extraction, embeddings, and chat)*

* **Recommended Backlog Tasks:**
  - [ ] **Hybrid Search (BM25 + FAISS)**: Merge SQLite FTS5 keyword search with vector embeddings using Reciprocal Rank Fusion (RRF).
  - [ ] **Chat Session Export**: Add an "Export Chat" button to save Q&A logs as Markdown or PDF.
  - [ ] **Source Viewer Drawer / Modal**: Clicking a citation badge (`📄 report.md (Page 1)`) opens a preview of the exact source paragraph.
  - [ ] **File Type Search Filters**: Checkboxes to filter search scope (e.g., *"Search code only"* or *"Search PDFs only"*).

---

### 👤 Role 3: Document Summarizer & Insights Specialist
> **Mission**: Build on-demand document analysis, multi-format summaries, key metrics extraction, and report generation tools.

* **Exclusively Owned Files (Only Role 3 edits these):**
  * `src/polaris/ui/summarize_tab.py` *(Document picker, style preset selectors, streaming markdown viewer)*
  * `src/polaris/ai/summarizer.py` *(Summarization prompts, excerpting strategies, token budgeting)*
  * `tests/test_summarize.py` *(Dedicated test file for summarization features)*

* **Recommended Backlog Tasks:**
  - [ ] **Multi-Document Comparative Summary**: Select 2–5 documents simultaneously and generate a consolidated brief.
  - [ ] **Length & Tone Controls**: UI sliders to adjust summary detail (*Brief / 1-Paragraph* vs *Comprehensive / Multi-Section*).
  - [ ] **Entity & Action Items Extractor**: A dedicated panel for extracting deadlines, financial figures, and action items.
  - [ ] **Export to File**: Allow users to save generated summaries directly as `.md` or `.txt` files on disk.

---

## 🚫 Shared Files (Do Not Modify Independently)

These files connect the application components together. Any modification requires team alignment:

| Shared File | Purpose | Why Independent Edits Are Restricted |
|---|---|---|
| `src/polaris/ui/main_window.py` | Mounts the 3 tabs and the model selector | Simultaneous edits here will cause git merge conflicts. |
| `src/polaris/config.py` | Global settings, default models, paths | All modules rely on these definitions. |
| `src/polaris/ai/ollama_client.py` | Low-level Ollama HTTP streaming wrapper | Already fully functional for all three tabs. |
| `pyproject.toml` | Dependencies and project packaging | Only update when adding new pip dependencies. |

---

## 🌿 Step-by-Step Git Collaboration Workflow

Follow this procedure daily to guarantee conflict-free branches:

### 1. Synchronize Local `main`
```powershell
git checkout main
git pull origin main
```

### 2. Create Your Feature Branch
```powershell
# Developer 1 (Organize):
git checkout -b feature/ai-organize

# Developer 2 (Search & Chat):
git checkout -b feature/search-chat

# Developer 3 (Summarizer):
git checkout -b feature/summarizer
```

### 3. Run Your Dedicated Test Suite
Run your specific test file during development for rapid feedback:
```powershell
pytest tests/test_organize.py    # (Role 1)
pytest tests/test_rag.py         # (Role 2)
pytest tests/test_summarize.py   # (Role 3)

# Or run the entire test suite before submitting:
pytest
```

### 4. Commit and Push to Your Branch
```powershell
git add .
git commit -m "feat(organize): add drag-and-drop folder support"
git push -u origin feature/<your-branch-name>
```

### 5. Open a Pull Request (PR) on GitHub
* Open a PR from your feature branch into `main`.
* Because each developer edited strictly their owned files, GitHub will report **"Able to merge automatically"** with zero merge conflicts.
