# 02 — Architecture & Multi-Agent Workflow

## The Dual-Brain Cognitive Architecture (System 1 + System 2)

SPARK AI implements a **Dual-Process Cognitive Architecture** inspired by Daniel Kahneman's model of human cognition:

* **System 1 (Reflex Arc — Laya Engine):** An ultra-fast, non-autoregressive decision engine running ModernBERT-large (English) and mmBERT-base (100+ languages) fine-tuned via **Reinforcement Learning for Calibrated Decisions (RLCD)**. It operates in sub-second forward passes without streaming tokens, returning mathematically calibrated probabilities for safety triage, greeting detection, CRAG document relevance scoring, model routing, and Universal Verification.
* **System 2 (Prefrontal Cortex — Generative Ensemble):** Slower, deliberate, multi-agent frontier models (Google Gemini 3.1 Flash-Lite, Gemini 3.5 Flash, Qwen 2.5 Coder 32B, Mistral Small 24B, Llama 3.3 70B, and Nvidia Nemotron) dedicated to task decomposition, query planning, code synthesis, creative writing, and factual critique.

<img width="1024" alt="arch" src="images/architecture_v3.png" />

---

## Decision & Generation Load Distribution

| Pipeline Phase | Primary Engine | Fallback Engine | Primitive Type | Typical Latency |
|---|---|---|---|---|
| **Safety Guardrail** | **Nvidia Nemotron** (`nemotron-3-nano:30b`) | Rule-based heuristics | Binary Audit | ~300ms |
| **Greeting & Triage** | **System 1 (Laya)** | `SIMPLE_PATTERNS` regex | `is_greeting: noul` | ~150ms |
| **Execution Planning** | **System 2 (Gemini)** (`gemini-3.1-flash-lite`) | Direct pass-through | Structured Text Plan | ~350ms |
| **CRAG Document Grading** | **System 1 (Laya)** | **Gemini 3.1 Flash-Lite** | `doc_relevance: choice` & `score` | ~180ms |
| **Worker Routing** | **System 1 (Laya)** | **Gemma-4 31B** (Ollama Cloud) | `task_type: choice` (`coding`, `creative`, `general`) | ~140ms |
| **Code Generation** | **System 2 (Gemini / Qwen)** | Mistral / Llama | Autoregressive Token Stream | 2.0s – 5.0s |
| **Universal Verification** | **System 1 (Laya)** | **Gemini 3.1 Flash-Lite** | `faithfulness: choice` (Claim check) | ~200ms |
| **Final Polishing** | **System 2 (Nemotron 3 Super)** | Draft pass-through | Text Refinement | ~400ms |

---

## Dual-Process Workflow Architecture

![Dual-Process Workflow Architecture](images/architecture_v3.png)

### End-to-End Execution Flow
1. **Security Gate & Fast-Path Triage:** Every incoming prompt is evaluated for safety by `nemotron-3-nano:30b`. In parallel, System 1 Laya runs a non-autoregressive greeting reflex (`is_greeting: noul`). If recognized as a greeting, the workflow executes a sub-second conversational reflex.
2. **Date-Aware Planning & Retrieval:** For actionable tasks, `gemini-3.1-flash-lite` decomposes the goal with temporal date awareness, retrieving relevant context from ChromaDB.
3. **Corrective RAG (CRAG):** Retrieved document chunks are evaluated by System 1 Laya's non-autoregressive relevance head (`doc_relevance: choice` + `score`). If chunks are irrelevant or missing, live internet ground truth is retrieved via SearXNG.
4. **Intent Dispatching & Worker Execution:** System 1 Laya classifies the task type (`coding`, `creative`, `general`) in ~140ms and routes to the specialized worker LLM.
5. **Universal Verification & Grounding:** Generated drafts are verified by System 1 Laya (`faithfulness: choice`) in ~200ms. If hallucination is detected (≥ 85%), live web search grounding triggers re-generation.
6. **Final Polish & Langfuse Scoring:** The draft is polished with standard LaTeX math rendering and clean Markdown by the evaluator, with quality scores published directly to Langfuse.

### 1. Root Trace Instrumentation
- In `process_chat()` and `stream_graph_updates()`, the pipeline initializes the Langfuse `CallbackHandler`.
- Trace attributes are propagated automatically:
  - `langfuse_session_id`: Matches the unique per-request session ID.
  - `langfuse_trace_name`: Set to `"SPARK-AI-Workflow"`.
  - `langfuse_tags`: Tagged with `["production", "agentic-rag"]`.

### 2. Node-Level Generation Linking
- Within `_safe_llm_call()`, when an agent node executes, it requests its prompt template from `get_managed_prompt(prompt_name)`.
- The prompt object is attached to the LangChain invoke metadata:
  ```python
  config={"metadata": {"langfuse_prompt": prompt_obj}}
  ```
- This directly links every LLM generation to its active prompt version in the Langfuse dashboard.

### 3. Automated Quality Scoring
- The `validation_node` evaluates drafts against relevance, factuality, and coherence criteria.
- Once evaluated, `log_validation_score()` logs a numeric metric (`1.0` for approved, `0.0` for failed) along with validator critique comments into the Langfuse trace.

