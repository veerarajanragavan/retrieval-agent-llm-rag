# retrieval-agent-llm-rag

Retrieval-Augmented Generation (RAG) demo agent using sentence-transformers + FAISS and a local LLM stub for offline demo.

## Quickstart (browser-only / local optional)
1. This repository is a demo skeleton. To run locally you need Python and the listed requirements.
2. If you cannot run locally, you can still showcase this repo on GitHub — the code and usage examples are visible here.

## What is included
- `src/` : core modules (embeddings, vectordb, retriever, agent, llm stub)
- `scripts/` : helper scripts to build a demo corpus and index
- `docs/sample_docs/` : small demo docs to ingest
- `tests/` : minimal pytest

## How to run locally (optional)
```bash
python -m venv .venv
source .venv/bin/activate    # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
python scripts/init_corpus.py
python scripts/build_faiss.py
python -c "from src.agent import RetrievalAgent; a=RetrievalAgent(); a.ingest_docs([{'text':'hello world','meta':{}}]); print(a.answer('hello'))"
