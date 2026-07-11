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
