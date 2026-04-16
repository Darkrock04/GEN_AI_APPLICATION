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

Add at least:

- `NVIDIA_API_KEY` — or switch `LLM_PROVIDER` in variables and set the matching keys (see `.env.example`).

Optional variables (same as `.env`):

- `LLM_PROVIDER`
- `CORS_ORIGINS` (use `*` on HF unless you restrict)

## Single entrypoint

This repo's root [`app.py`](../app.py) starts **uvicorn** (`backend.main:app`) on `127.0.0.1:7861`, then **Streamlit** on `$PORT`. The UI reads `BACKEND_URL` automatically.

If you prefer two Spaces (API + UI), run only Streamlit and set `SKIP_EMBEDDED_FASTAPI=1` and `BACKEND_URL` to your API URL.

## Hardware

GPU is optional; NIM calls are remote. Choose a **CPU basic** Space for demos.
