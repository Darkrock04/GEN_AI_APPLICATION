# 02 — Architecture & Multi-Agent Workflow

## How the Pipeline Works

Every user message flows through a **LangGraph StateGraph** — a directed graph where each node is a specialized agent. Routing is dynamic based on the content of the request.

<img width="1024" alt="arch" src="images/architecture_v2.png" />

## LLM Calls Per Request Type

| Request Type | Total LLM Calls | Path |
|---|---|---|
| Simple greeting ("hello") | **1** | Security (keyword only) → Quick Response |
| Complex query (local info) | **5** | Security + Planner + Router + Worker + Validator |
| Complex query (live web search) | **6** | Security + Planner + Web Search + Router + Worker + Validator |
| Complex + evaluator | **+1** | Above + Evaluator (long responses only) |
| Complex + 1 retry | **+1** | Above + Worker retry + Validator |

---

## Observability & Prompt Management in the Workflow

Every execution of the LangGraph pipeline is observed and managed through Langfuse:

```mermaid
flowchart TD
    subgraph Client["Client Interaction"]
        User["User Request"] --> Entry["FastAPI /chat or /chat/stream"]
    end

    subgraph LangfusePlane["Langfuse Control Plane"]
        Registry["Prompt Registry (10 Prompts, 300s TTL Cache)"]
        TraceHandler["CallbackHandler (Session ID, Tags, Metadata)"]
        ScoreLogger["Quality Score Logger (quality_validation: 1.0 / 0.0)"]
    end

    subgraph GraphExecution["LangGraph StateGraph"]
        Stress["stress_test_node\n(security_gate_prompt)"]
        Planner["planner_node\n(planner_prompt)"]
        WebSearch["web_search_node\n(SearXNG)"]
        Retrieve["retrieve_context_node\n(ChromaDB + Gemini Embeddings)"]
        Router["router_node\n(router_prompt)"]
        Worker["worker_agent_node\n(worker_*_prompt)"]
        Validator["validation_node\n(validator_prompt)"]
        Evaluator["evaluation_node\n(evaluator_prompt)"]
    end

    Entry --> TraceHandler
    TraceHandler --> Stress
    Registry -.->|Fetch Template & Link Gen| Stress
    Registry -.->|Fetch Template & Link Gen| Planner
    Registry -.->|Fetch Template & Link Gen| Router
    Registry -.->|Fetch Template & Link Gen| Worker
    Registry -.->|Fetch Template & Link Gen| Validator
    Registry -.->|Fetch Template & Link Gen| Evaluator

    Stress --> Planner
    Planner -->|needs_web_search| WebSearch --> Retrieve
    Planner -->|standard| Retrieve
    Retrieve --> Router
    Router --> Worker
    Worker --> Validator
    Validator -.->|Log Score| ScoreLogger
    Validator -->|FAIL & attempt < 2| Worker
    Validator -->|PASS| Evaluator
    Evaluator --> Output["Final Response + Citations + Timings"]
```

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

