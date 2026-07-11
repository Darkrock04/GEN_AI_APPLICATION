# 03 — Models & Agent Roles (Multi-Cloud Architecture)

## Why Multiple Models?

SPARK AI uses a **Multi-Cloud Mixture of Agents**. Instead of relying on a single provider and hitting their arbitrary free-tier bottlenecks, we distribute the workload across three different tech giants. Each agent is routed to the specific cloud provider that offers the best free-tier advantage for that exact task.

## Multi-Cloud Assignments

| Agent | Cloud Provider | Model | Why This Assignment? |
|---|---|---|---|
| **Security Gate** | **Nvidia NIM** | `meta/llama-3.1-8b-instruct` | Fast logic checking. |
| **Planner** | **Google (Gemini)** | `gemini-3.1-flash-lite` | Strong reasoning, 15 RPM, 500 requests per day. |
| **Router** | **Cerebras** | `gemma-4-31b` | Instant classification on Cerebras. |
| `worker_general` | **Cerebras** | `gpt-oss-120b` | Massive generation. Takes advantage of Cerebras's 1,000,000 TPD allowance. |
| `worker_creative` | **Cerebras** | `gpt-oss-120b` | High generation capabilities without hitting daily limits. |
| `worker_coding` | **Google (Gemini)** | `gemini-3.5-flash` | Best coding model available on the free tier with 250K TPM. |
| **Validator** | **Google (Gemini)** | `gemini-3.1-flash-lite` | Fast validation pass with huge 500 daily requests limit. |
| **Evaluator** | **Nvidia NIM** | `meta/llama-3.1-70b-instruct`| Perfect for final text polish. |
| **Embeddings** | **Nvidia NIM** | `nv-embedqa-e5-v5` | Enterprise-grade RAG embeddings. |

---

## Free Tier Rate Limits (2026 Reference Guide)

To ensure SPARK AI runs completely for free, we engineered it around the following constraints:

### 1. Cerebras Inference
- **Requests Per Minute (RPM):** 30
- **Tokens Per Day (TPD):** 1,000,000
- *Our Usage:* 1 request per message (Workers). We heavily utilize their massive 1M daily token allowance for generating huge walls of text.

### 2. Google AI Studio (Gemini)
- **Requests Per Minute (RPM):** 15
- **Tokens Per Minute (TPM):** 250,000
- *Our Usage:* Used exclusively for heavy coding tasks. The generous 15 RPM is perfect since this node is only called occasionally.

### 3. Nvidia NIM
- **Requests Per Minute (RPM):** ~40
- *Our Usage:* Used for evaluation and embeddings, staying comfortably beneath the limit.

## API Keys
To run this architecture, you must configure the following `.env` variables:
- `CEREBRAS_API_KEY`
- `GEMINI_API_KEY`
- `NVIDIA_API_KEY`
