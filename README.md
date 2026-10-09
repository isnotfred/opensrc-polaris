# Polaris: Local-First Intelligent File Assistant

Polaris is a private, on-premise desktop assistant built with **PySide6** and powered 100% on-device by **Ollama**. No cloud calls, zero external API keys, and zero data leakage.

Designed to run smoothly on standard laptops and desktops (including CPU-only 8 GB RAM machines).

---

## 3 Core Capabilities

### 1. 📁 AI-Powered File Organizing
- **Natural Language Sorting**: Tell the AI how to organize your files (e.g. *"Sort invoices by year & client"*, *"Group research papers by topic"*, or use one-click presets like *Smart Auto-Group*).
- **Deterministic Safety Guarantee**: The model only proposes moves. Paths are validated, and destination collisions are automatically avoided using incremental suffixes (`(1)`, `(2)`) so files are **never overwritten**.
- **Interactive Preview Table**: Review every proposed move with checkboxes to include or exclude files before applying.
- **1-Click Transactional Undo**: Every operation batch is recorded in SQLite (`operations` table) and can be restored to its exact original state at any time.
- **Rule-Based Fallback**: Instant extension-based sorting (`Documents/`, `Images/`, `Source Code/`, etc.) without LLM inference.

### 2. 💬 AI Search & Document Chatbot
- **Multi-Format Ingestion**: Extracts text with page tracking from **PDFs** (via `PyMuPDF`), **Word Documents** (`.docx`), **Markdown**, **Plain Text**, and **Source Code**.
- **Local FAISS Vector Index**: Chunks text and computes 768-dimensional float32 embeddings with `nomic-embed-text`.
- **Grounded Q&A with Citations**: Answers questions using only verified context from your files with explicit source citations (e.g. `📄 report.md (Page 1)`).
- **Real-Time Token Streaming**: Words stream onto the screen in real-time as they are generated.

### 3. 📝 AI Document Summarizer
- **Style Presets**: Generate *Key Takeaways & Action Items* (bullets), *Executive Summary* (1–2 paragraphs), or *Detailed Notes*.
- **CPU-Tuned Context**: Prioritized excerpting ensures multi-page documents process in seconds on CPU.
- **1-Click Export**: Formatted Markdown display with a **📋 Copy Summary to Clipboard** button.

---

## Quick Start & Installation

### 1. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install Polaris with all dependencies
pip install -e ".[dev,docs,search]"
```

### 2. Local AI Setup (via Ollama)
Install Ollama from [ollama.com](https://ollama.com), then pull the lightweight recommended models:
```powershell
# Default fast LLM (~986 MB, ~35-50 tok/s on CPU)
ollama pull qwen2.5:1.5b

# High-quality embedding model (~274 MB)
ollama pull nomic-embed-text

# (Optional) Alternative 3B model
ollama pull llama3.2:3b
```

### 3. Launch Application
```powershell
python -m polaris
# or:
polaris
```

*(You can toggle between `qwen2.5:1.5b` and `llama3.2:3b` live via the **🧠 Model** dropdown in the top-right corner of the application window).*

---

## Trying the Included Demo Dataset

A sample folder is included at `sample_files/` containing 10 varied documents:
- `q3_financial_performance_report.md` (Financial report with revenue figures)
- `invoice_acme_corp_nov2024.txt` (Hardware server invoice)
- `receipt_delta_airlines_oct2024.txt` (Travel expense receipt)
- `quarterly_budget_2024.csv` (Departmental budget spreadsheet)
- `nda_quantum_ventures.txt` (Legal confidentiality contract)
- `project_apollo_architecture_spec.md` (Engineering architecture spec)
- `clean_customer_records.py` (Data pipeline script)
- `backup_server_logs_2024.zip` (Archive file)
- `vacation_beach_sunset.png` & `family_reunion_2023.jpg` (Photos)

**Test workflows:**
1. In **📁 AI Organize**: Select `sample_files`, click **Smart Auto-Group**, review preview table, click **Apply**, and test **Undo**.
2. In **💬 Search & Chat**: Select `sample_files`, click **⚡ Index Folder**, and ask: *"What was the total revenue in the Q3 report?"* or *"When does the NDA expire?"*.
3. In **📝 Summarize**: Select `sample_files/q3_financial_performance_report.md` and generate an **Executive Summary**.

---

## Automated Test Suite

Run unit and UI tests (supported in headless offscreen mode):
```powershell
pytest
```

---

## Documentation
* [docs/TEAM_ROLES_WORKFLOW.md](docs/TEAM_ROLES_WORKFLOW.md) — Simultaneous collaboration guide, developer roles, and git workflow.
* [docs/FEATURES.md](docs/FEATURES.md) — Comprehensive guide to all 3 core features, safety guarantees, and settings.
* [docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md) — Technical history of bug fixes, architecture evolution, and CPU tuning.
* [docs/PLAN.md](docs/PLAN.md) — High-level architecture and hardware design guidelines.
