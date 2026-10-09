# Project Apollo System Architecture Spec
Version: 1.2 | Lead Architect: Elena Vance

## Overview
Project Apollo is an on-premise document indexing and intelligence pipeline designed to run on constrained desktop hardware.

## Core Components
1. **File Watcher & Extractor**:
   - Listens to filesystem events using OS notification hooks.
   - Converts multi-page documents (PDF, DOCX, TXT) into chunked textual blocks with metadata.
2. **Local Embedding Engine**:
   - Uses `nomic-embed-text` running via Ollama.
   - Computes 768-dimensional float32 vectors.
3. **Vector Database**:
   - FAISS `IndexFlatIP` utilizing inner-product similarity over L2-normalized vectors.
   - Operates entirely in RAM with disk persistence to SQLite.

## Hardware Constraints
- Target RAM: <= 8 GB system RAM.
- CPU: Optimized for 4-core / 8-thread modern x86_64 architectures.
- Zero network egress: All network calls bound strictly to localhost (`127.0.0.1:11434`).
