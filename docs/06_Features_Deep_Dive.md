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

## 2. Consolidated Validation (Single LLM Call)

Every complex response passes through a single validation call that checks three criteria simultaneously:

| Criterion | Checks | Fails When |
|---|---|---|
| **Relevance** | Does it answer the question? | Response is off-topic |
| **Factuality** | Does it avoid making up information? | Contains hallucinated facts |
| **Coherence** | Is it well-structured? | Poorly organized or illogical |

The validator returns PASS or FAIL with a brief reason. On FAIL, the draft is sent back to the worker with specific feedback. Maximum 1 retry to prevent loops.

**Why consolidated?** The original design used 3 separate LLM calls (gatekeeper + auditor + strategist). Consolidating into 1 call reduces latency by ~66% — critical for Hugging Face free tier.

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

## 6. Fail-Open Error Handling

Every LLM call is wrapped in `_safe_llm_call()`:
```python
def _safe_llm_call(prompt, llm, variables, fallback, retries=1):
    try:
        result = (prompt | llm).invoke(variables)
        return result.content.strip() or fallback
    except Exception:
        return fallback  # Never crash — use fallback
```

This means:
- If any agent crashes, the pipeline continues with a fallback
- If the SiliconFlow API has a transient error, the user gets a degraded but functional response
- The system **never shows a 500 error** to the user

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

The streaming uses NDJSON format — one JSON object per line, each containing the node name and its state update.
