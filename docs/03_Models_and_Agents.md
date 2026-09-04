# 03 — Models & Agent Roles (Multi-Cloud Architecture)

## Why Multiple Models?

SPARK AI uses a **Multi-Cloud Mixture of Agents**. Instead of relying on a single provider and hitting their arbitrary free-tier bottlenecks, we distribute the workload across three different tech giants. Each agent is routed to the specific cloud provider that offers the best free-tier advantage for that exact task.

## Multi-Cloud Assignments

| Agent | Cloud Provider | Model | Why This Assignment? |
|---|---|---|---|
| **Security Gate** | **Ollama Cloud** | `nemotron-3-nano:30b` | Fast logic checking with Nvidia Nemotron on Ollama. |
| **Planner** | **Google (Gemini)** | `gemini-3.1-flash-lite` | Strong reasoning, 15 RPM, 500 requests per day. |
| **Router** | **Ollama Cloud** | `gemma4:31b` | Instant classification with Ollama free cloud tier. |
| `worker_general` | **Ollama Cloud** | `gpt-oss:120b` | Massive generation. Takes advantage of Ollama's free cloud model tier. |
| `worker_creative` | **Ollama Cloud** | `gpt-oss:120b` | High generation capabilities without hitting daily limits. |
| `worker_coding` | **Google (Gemini)** | `gemini-3.5-flash` | Best coding model available on the free tier with 250K TPM. |
| **Validator** | **Google (Gemini)** | `gemini-3.1-flash-lite` | Fast validation pass with huge 500 daily requests limit. |
| **Evaluator** | **Ollama Cloud** | `nemotron-3-super` | Perfect for final text polish with Nemotron 3 Super. |
| **Embeddings** | **Google (Gemini)** | `gemini-embedding-001` | High-throughput RAG embeddings on Gemini API. |

---

## Free Tier Rate Limits (2026 Reference Guide)

To ensure SPARK AI runs completely for free, we engineered it around the following constraints:

### 1. Ollama Cloud
- **Endpoint:** `https://ollama.com/v1` (OpenAI-compatible)
- **Designated Free Models:** `gemma4:31b`, `gpt-oss:120b`, `gpt-oss:20b`, `nemotron-3-nano:30b`, `nemotron-3-super`, `nemotron-3-ultra`
- *Our Usage:* Used for routing (`gemma4:31b`) and primary content generation (`gpt-oss:120b`).

### 2. Google AI Studio (Gemini)
- **Requests Per Minute (RPM):** 15
- **Tokens Per Minute (TPM):** 250,000
- *Our Usage:* Used exclusively for heavy coding tasks. The generous 15 RPM is perfect since this node is only called occasionally.

### 3. Nvidia NIM
- **Requests Per Minute (RPM):** ~40
- *Our Usage:* Used for evaluation and embeddings, staying comfortably beneath the limit.

## API Keys
To run this architecture, you must configure the following `.env` variables:
- `OLLAMA_API_KEY` (and optional `OLLAMA_BASE_URL`)
- `GEMINI_API_KEY`
- `NVIDIA_API_KEY`
