# 03 — Models & Agent Roles

## Why Multiple Models?

SPARK AI uses **specialized model assignments** instead of one general-purpose model. Lightweight 7B models handle classification tasks (fast, cheap), while the large models (like DeepSeek-V3) handle generation (high quality).

## Model Assignment Table

| Agent | Model | Why This Model? |
|---|---|---|
| **Security Gate** | `Qwen/Qwen2.5-7B-Instruct` | Fast binary SAFE/UNSAFE classification |
| **Planner** | `Qwen/Qwen2.5-7B-Instruct` | Task decomposition needs reasoning |
| **Router** | `Qwen/Qwen2.5-7B-Instruct` | Simple 3-way classification |
| **Worker (General)** | `deepseek-ai/DeepSeek-V3` | Best overall for analysis, Q&A |
| **Worker (Coding)** | `deepseek-ai/DeepSeek-V3` | Code generation |
| **Worker (Creative)** | `deepseek-ai/DeepSeek-V3` | Creative writing |
| **Validator** | `Qwen/Qwen2.5-7B-Instruct` | Quality assessment |
| **Evaluator** | `deepseek-ai/DeepSeek-V3` | Final polish and formatting |
| **Embeddings** | `Qwen/Qwen3-Embedding-0.6B` | Document RAG vectors |

## API Access

All models are accessed through the SiliconFlow API:
- **Endpoint:** `https://api.siliconflow.cn/v1`
- **Auth:** Bearer token via `SILICONFLOW_API_KEY`
- **Provider setup:** `LLM_PROVIDER=siliconflow`

## Configuration

- Each `ChatOpenAI` instance has `timeout=45` and `max_retries=1`
- Security and Router use 7B model with `max_tokens=64` and `32` respectively
- Workers use `max_tokens=4096` for full responses

## How to Change Models

Edit `backend/llm_factory.py` — the `SILICONFLOW_MODELS` dictionary:

```python
SILICONFLOW_MODELS = {
    "security": "Qwen/Qwen2.5-7B-Instruct",
    "planner": "Qwen/Qwen2.5-7B-Instruct",
    ...
}
```

Browse available models on the SiliconFlow documentation.
