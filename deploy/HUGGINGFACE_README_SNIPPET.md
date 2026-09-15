# Hugging Face Space — README header

Copy the YAML below to the **top** of your Space `README.md` (create the Space with SDK **Streamlit**), then push this repo.

```yaml
---
title: SPARK AI
emoji: ⚡
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
license: mit
---
```

## Secrets (Space settings → Repository secrets)

Configure the following secrets for the Multi-Cloud and Langfuse architecture:

- `OLLAMA_API_KEY` — Ollama Cloud key (router, security, workers, evaluator)
- `GEMINI_API_KEY` — Google Gemini key (planner, coding worker, embeddings)
- `NVIDIA_API_KEY` — Nvidia NIM key (inference & embeddings)
- `SEARXNG_URL` — SearXNG search engine instance URL
- `LANGFUSE_PUBLIC_KEY` — Langfuse project public key (`pk-lf-...`)
- `LANGFUSE_SECRET_KEY` — Langfuse project secret key (`sk-lf-...`)
- `LANGFUSE_HOST` — Langfuse host (`https://cloud.langfuse.com`)

Optional variables (same as `.env`):

- `OLLAMA_BASE_URL` (defaults to `https://ollama.com/v1`)
- `CORS_ORIGINS` (use `*` on HF unless restricted)

## Single entrypoint

This repo's root [`app.py`](../app.py) starts **uvicorn** (`backend.main:app`) on `127.0.0.1:7861`, then **Streamlit** on `$PORT`. The UI reads `BACKEND_URL` automatically.

If you prefer two Spaces (API + UI), run only Streamlit and set `SKIP_EMBEDDED_FASTAPI=1` and `BACKEND_URL` to your API URL.

## Hardware

GPU is optional; NIM calls are remote. Choose a **CPU basic** Space for demos.
