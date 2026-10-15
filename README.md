# frontdesk-rag-core

A small, dependency-free RAG bot for business FAQs. It's pure Python 3.9+ standard library: no pip installs and no vector database.

- **Retrieval:** BM25 over 80-word overlapping chunks from a folder of `.md`/`.txt` files
- **Answers:** any OpenAI-compatible endpoint (OpenAI, Groq, Together, local Ollama), or with no LLM at all, an extractive fallback that quotes your docs
- **Citations:** every answer names its source file
- **Refusal:** when nothing relevant is retrieved (or the LLM finds no answer in the context) it hands off to a human contact instead of guessing
- **Resilience:** retries, timeouts and automatic fallback to extractive mode if the LLM server fails

## Quick start
```bash
mkdir docs && cp your-faq.md docs/
python bot.py ask "What are your opening hours?"     # CLI, prints JSON
python bot.py serve                                   # JSON API on :8000
curl -s localhost:8000/chat -d '{"message":"Do you have parking?"}'
```
Optional LLM mode (local Ollama example):
```bash
export LLM_API_KEY=ollama LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3.1
```
Settings: `DOCS_DIR`, `BUSINESS_NAME`, `CONTACT`, `PORT`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`.

API: `GET /health` and `POST /chat {"message": "..."}` return `{"answer", "sources", "mode"}`.

## Tests
```bash
python -m unittest discover tests
```
These use a small fictional fixture and a mock OpenAI-compatible server. Live providers haven't been tested.

## Want the full kit?
The paid **FrontDesk RAG Kit** adds an embeddable website chat widget, a Dockerfile, sample business data and an agency/client licence: https://ritikster36.gumroad.com/l/rmxrkw

## Disclosure
Written by Ritik Kumar with AI assistance, then reviewed and tested by me. MIT licence (see LICENSE). No warranty.
