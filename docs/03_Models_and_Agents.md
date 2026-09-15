# 03 — Models & Agent Roles (Multi-Cloud Architecture)

## Why Multiple Models?

SPARK AI uses a **Multi-Cloud Mixture of Agents**. Instead of relying on a single provider and hitting their arbitrary free-tier bottlenecks, we distribute the workload across three different tech giants. Each agent is routed to the specific cloud provider that offers the best free-tier advantage for that exact task.

## Multi-Cloud & Prompt Registry Assignments

| Agent / Role | Cloud Provider | Model | Managed Langfuse Prompt | Why This Assignment? |
|---|---|---|---|---|
| **Security Gate** | **Ollama Cloud** | `nemotron-3-nano:30b` | `security_gate_prompt` | Fast logic checking with Nvidia Nemotron on Ollama. |
| **Quick Greeter** | **Ollama Cloud** | `gpt-oss:120b` | `simple_answer_prompt` | Natural, fast responses for basic greetings. |
| **Planner** | **Google (Gemini)** | `gemini-3.1-flash-lite` | `planner_prompt` | Strong reasoning, 15 RPM, 500 requests per day. |
| **Router** | **Ollama Cloud** | `gemma4:31b` | `router_prompt` | Instant classification with Ollama free cloud tier. |
| `worker_general` | **Ollama Cloud** | `gpt-oss:120b` | `worker_general_prompt` | Massive generation. Takes advantage of Ollama's free cloud tier. |
| `worker_creative` | **Ollama Cloud** | `gpt-oss:120b` | `worker_creative_prompt` | High generation capabilities without hitting daily limits. |
| `worker_coding` | **Google (Gemini)** | `gemini-3.5-flash` | `worker_coding_prompt` | Best coding model available on the free tier with 250K TPM. |
| **Validator** | **Google (Gemini)** | `gemini-3.1-flash-lite` | `validator_prompt` | Fast validation pass with huge 500 daily requests limit. |
| **Evaluator** | **Ollama Cloud** | `nemotron-3-super` | `evaluator_prompt` | Perfect for final text polish with Nemotron 3 Super. |
| **Memory Summarizer** | **Ollama Cloud** | `gemma4:31b` | `history_summarizer_prompt` | Rapid compression of long chat history. |
| **Embeddings** | **Google (Gemini)** | `gemini-embedding-001` | *(Vector pipeline)* | High-throughput RAG embeddings on Gemini API. |

---

## Free Tier Rate Limits (2026 Reference Guide)

To ensure SPARK AI runs completely for free, we engineered it around the following constraints:

### 1. Ollama Cloud
- **Endpoint:** `https://ollama.com/v1` (OpenAI-compatible)
- **Designated Free Models:** `gemma4:31b`, `gpt-oss:120b`, `gpt-oss:20b`, `nemotron-3-nano:30b`, `nemotron-3-super`, `nemotron-3-ultra`
- *Our Usage:* Used for routing (`gemma4:31b`), security, evaluation, and primary content generation (`gpt-oss:120b`).

### 2. Google AI Studio (Gemini)
- **Requests Per Minute (RPM):** 15
- **Tokens Per Minute (TPM):** 250,000
- *Our Usage:* Used exclusively for heavy coding tasks, planning, validation, and embeddings. The generous 15 RPM is perfect for high reliability.

### 3. Nvidia NIM
- **Requests Per Minute (RPM):** ~40
- *Our Usage:* Additional specialized model access and fallback inference.

## Environment Variables & Keys
To run this architecture, configure the following `.env` variables:
- `OLLAMA_API_KEY` (and optional `OLLAMA_BASE_URL`)
- `GEMINI_API_KEY`
- `NVIDIA_API_KEY`
- `SEARXNG_URL`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST`

