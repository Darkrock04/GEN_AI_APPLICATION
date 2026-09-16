# 06 — Features Deep Dive

## 1. Semantic Summarization Memory

To prevent context window overflow during long conversations, the system uses a **Semantic Summarization** engine:
- The frontend sends the full history array.
- If the history exceeds **6 messages**, a lightning-fast background LLM (e.g., Llama 3.1 8B) compresses all older messages into a dense semantic summary.
- It actively extracts user preferences, names, and unresolved facts.
- The 6 most recent messages are kept perfectly intact below the summary.

This guarantees the model remembers core context infinitely without crashing due to token limits.

## 1.5. Live Web Search (SearXNG)

The agent has direct access to the live internet:
- The **Planner Node** scans the user's query. If it detects a need for live data (news, current events), it flags `needs_web_search`.
- The graph routes to the **Web Search Node**, which silently queries a self-hosted SearXNG instance.
- The returned web snippets and URLs are injected into the context window, giving the LLM live knowledge far beyond its original training cutoff.

## 1.6. Enterprise Corrective RAG (CRAG) & Self-RAG Loops

To eliminate hallucinations and knowledge cutoff apologies, SPARK AI incorporates industrial self-correction:
- **Document Relevance Grading:** Chunks retrieved from ChromaDB are graded by `grade_documents_node`. If they are off-topic or empty, the pipeline dynamically pivots to SearXNG web search rather than allowing the worker to guess.
- **Dynamic Date Injection:** The execution date (`datetime.now()`) is injected into planner and worker prompts (`{{current_date}}`), making models actively aware of temporal context.
- **Worker Self-Reflection Loop:** If a worker emits `[NEEDS_WEB_SEARCH: query]` or admits a cutoff (*"as of my knowledge cutoff"*, *"up to mid-2024"*), `route_after_worker` intercepts the draft, queries SearXNG, enriches the context, and re-invokes the worker for a grounded answer.
- **Cutoff Gate in Validator:** If a draft contains cutoff disclaimers without live search, the validator fails with `FAIL: CUTOFF_DETECTED`, prompting an automatic web search recovery pass.

## 2. Consolidated Validation (Single LLM Call)

Every complex response passes through a single validation call that checks three criteria simultaneously:

| Criterion | Checks | Fails When |
|---|---|---|
| **Relevance** | Does it answer the question? | Response is off-topic |
| **Factuality** | Does it avoid making up information? | Contains hallucinated facts |
| **Coherence** | Is it well-structured? | Poorly organized or illogical |

The validator returns PASS or FAIL with a brief reason. On FAIL, the draft is sent back to the worker with specific feedback. Maximum 1 retry to prevent loops.

**Why consolidated?** The original design used 3 separate LLM calls (gatekeeper + auditor + strategist). Consolidating into 1 call reduces latency by ~66% — critical for Hugging Face free tier.

**Automated Langfuse Trace Scoring:** Every validation result is automatically pushed to Langfuse via `log_validation_score()`. The trace records a numeric score:
- `quality_validation` = `1.0` on PASS
- `quality_validation` = `0.0` on FAIL
Along with validator feedback comments and session IDs, enabling real-time quality tracking on the Langfuse dashboard.

## 3. Two-Stage Content Safety

**Stage 1 (Python — instant, free):**
Checks if the query contains any of 50+ safe keywords. If yes, immediately marks SAFE. This catches 95%+ of queries without an LLM call.

**Stage 2 (LLM — only for suspicious queries):**
If Stage 1 doesn't match, the security model makes the SAFE/UNSAFE decision.

**Design principle:** *Fail open* — if the security LLM crashes, it defaults to SAFE rather than blocking legitimate users.

## 4. Document Management

Users can:
- **Upload** PDFs and TXT files via the sidebar (auto-processed, no button needed)
- **View** all uploaded documents in the sidebar
- **Delete** individual documents (removes vectors + raw file)
- **Clear session** to wipe everything for a fresh start

## 5. Session Management

On page load, SPARK AI automatically:
1. Calls `/clear_session` to wipe any previous data
2. Initializes a fresh ChromaDB collection
3. Starts with an empty chat history

On "Clear Session" button click:
1. Clears all documents from ChromaDB
2. Creates a new collection with a random UUID name
3. Deletes uploaded files from disk
4. Resets chat history in the frontend

**Windows-safe:** ChromaDB holds a SQLite lock. Instead of deleting the DB file, we rotate to a new collection.

## 6. Fail-Open Error Handling & Dynamic Overrides

Every LLM call is wrapped in `_safe_llm_call()`:
```python
def _safe_llm_call(
    prompt_template: str = "",
    llm = None,
    variables: dict = None,
    fallback: str = "",
    retries: int = 1,
    temperature: float = None,
    top_p: float = None,
    max_tokens: int = None,
    prompt_name: str = None,
    langfuse_prompt: Any = None,
) -> str:
    # 1. Fetch managed prompt from Langfuse Prompt Registry (with TTL cache)
    if prompt_name:
        managed_template, lf_obj = get_managed_prompt(prompt_name)
        if managed_template:
            prompt_template = managed_template
        if lf_obj is not None:
            langfuse_prompt = lf_obj

    # 2. Bind dynamic temperature, top_p, and max_tokens
    # 3. Attach metadata={"langfuse_prompt": langfuse_prompt} for trace linking
    # 4. Invoke with exponential backoff on rate limits (429)
    # 5. Return fallback on fatal failure — never crash the pipeline
```

This means:
- If any agent node encounters an issue, the pipeline continues with a graceful fallback
- If an upstream cloud API encounters a rate-limit (429), exponential retry backoff engages automatically
- The system **never crashes or returns a 500 error** to the user

## 7. Pipeline Streaming

When "Stream Responses" is enabled, the frontend displays real-time pipeline progress:

| Node Name | Display Label |
|---|---|
| `stress_test` | 🔒 Security Check |
| `planner` | 📋 Planning |
| `retrieve` | 🔍 Retrieving Context |
| `router` | 🔀 Routing |
| `worker` | ✍️ Generating Response |
| `validation` | ✅ Validating |
| `evaluation` | ✨ Polishing |
| `simple_answer` | 💬 Responding |

The streaming uses NDJSON format — one JSON object per line, each containing the node name, timing metadata, and state updates.

## 8. Centralized Prompt Management & Hot-Swapping

All 10 system prompts are decoupled from application code and managed via **Langfuse Prompt Registry**:
1. `security_gate_prompt` — Content safety rules
2. `simple_answer_prompt` — Friendly greeting and conversational instructions
3. `planner_prompt` — Task breakdown and web search detection
4. `router_prompt` — Intent classification rules
5. `worker_general_prompt` — General analysis and document synthesis
6. `worker_coding_prompt` — Production-grade coding instructions
7. `worker_creative_prompt` — Creative writing persona
8. `validator_prompt` — Strict relevance/factuality/coherence rubric
9. `evaluator_prompt` — LaTeX math formatting and markdown polishing
10. `history_summarizer_prompt` — Long conversation context compression

**Key benefits:**
- **Zero-downtime prompt changes:** Edit or fine-tune prompts directly in the Langfuse dashboard. The application fetches updated prompts automatically within 300 seconds (TTL cache).
- **Auto-seeding:** If a prompt does not yet exist on your Langfuse project, `ensure_prompts_seeded()` automatically creates it on application startup with the `production` label.
- **Rollback protection:** If Langfuse is unreachable, the system transparently falls back to local default prompt templates.

## 9. Full Observability & Trace Visualizer

SPARK AI integrates deep observability with Langfuse:
- **Trace Graph:** View every execution step from API entry to final response.
- **Latency Breakdown:** Inspect millisecond execution times per agent node and per cloud provider.
- **Token & Cost Tracking:** Track token consumption across Ollama Cloud, Google Gemini, and Nvidia NIM models.
- **Prompt Version Correlation:** Analyze how different prompt iterations impact generation quality, latency, and token consumption.
- **Session Attribution:** Multi-turn conversations are linked under consistent session IDs for longitudinal analysis.
