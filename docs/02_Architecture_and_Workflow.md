# 02 — Architecture & Multi-Agent Workflow

## How the Pipeline Works

Every user message flows through a **LangGraph StateGraph** — a directed graph where each node is a specialized agent. Routing is dynamic based on the content of the request.

## Architecture Diagram

```
User Message
    │
    ▼
┌──────────────────────────────────────────────┐
│  SECURITY GATE (Qwen2.5-7B)                  │
│  Stage 1: Python keyword pre-filter (free)   │
│  Stage 2: LLM classification (if ambiguous)  │
└──────┬──────────┬──────────┬─────────────────┘
       │          │          │
    UNSAFE     SIMPLE     COMPLEX
       │       (≤3 words)    │
       ▼          │          ▼
   [BLOCKED]      │      PLANNER
                  ▼      (Qwen2.5-7B)
             Quick Reply     │
                  │          ▼
                [END]   ┌──────────┐
                        │ RETRIEVER │  ChromaDB top-5
                        └────┬─────┘
                             ▼
                        ┌────────┐
                        │ ROUTER  │  (Qwen2.5-7B)
                        └─┬──┬──┬┘
                          │  │  │
                     coding creative general
                          │  │  │
                          ▼  ▼  ▼
               ┌────────────────────────┐
               │   SPECIALIZED WORKER    │
               │   (DeepSeek-V3)        │
               └──────────┬─────────────┘
                          ▼
               ┌────────────────────────┐
               │   VALIDATOR (1 call)   │
               │   Relevance + Facts    │
               │   + Coherence          │
               └─────┬──────────┬───────┘
                     │          │
                   PASS     FAIL (max 1 retry)
                     │          └──▶ WORKER
                     ▼
               ┌────────────────┐
               │   EVALUATOR    │  (skipped if < 500 chars)
               │   Final polish │
               └──────┬─────────┘
                      ▼
                 User Response
```

## LLM Calls Per Request Type

| Request Type | Total LLM Calls | Path |
|---|---|---|
| Simple greeting ("hello") | **1** | Security (keyword only) → Quick Response |
| Complex query (all pass) | **5** | Security + Planner + Router + Worker + Validator |
| Complex + evaluator | **5–6** | Above + Evaluator (long responses only) |
| Complex + 1 retry | **7** | Above + Worker retry + Validator |
