# 02 — Architecture & Multi-Agent Workflow

## How the Pipeline Works

Every user message flows through a **LangGraph StateGraph** — a directed graph where each node is a specialized agent. Routing is dynamic based on the content of the request.

## Architecture Diagram

<img width="1024" height="1536" alt="arch" src="https://github.com/user-attachments/assets/fce89528-ece8-4ff0-aed0-661073dd334b" />

## LLM Calls Per Request Type

| Request Type | Total LLM Calls | Path |
|---|---|---|
| Simple greeting ("hello") | **1** | Security (keyword only) → Quick Response |
| Complex query (all pass) | **5** | Security + Planner + Router + Worker + Validator |
| Complex + evaluator | **5–6** | Above + Evaluator (long responses only) |
| Complex + 1 retry | **7** | Above + Worker retry + Validator |
