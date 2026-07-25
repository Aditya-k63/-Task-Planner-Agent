# Task Planner Agent

> **AI agent that autonomously decomposes goals into dependency DAGs, executes tasks with tools, and adapts via interactive clarification — all through a real-time web UI.**

## What It Does

Given a high-level goal, the agent:
1. **Plans** — breaks the goal into ordered, prioritized subtasks with dependency tracking
2. **Executes** — uses tools (file ops, shell, web search, URL fetch) to complete each task
3. **Clarifies** — detects ambiguity mid-execution and presents 3 contextual options via interactive UI
4. **Reviews** — LLM-based review verifies task completion before proceeding
5. **Replans** — dynamically adjusts the plan when tasks fail or dependencies change

## Architecture

```
User Goal
    │
    ▼
┌─────────┐     ┌─────────┐     ┌───────────┐
│ Planner │────▶│ Router  │────▶│ Executor  │
│ (LLM)   │◀────│         │◀────│ (LLM+    │
└─────────┘     └────┬────┘     │  Tools)   │
                     │          └─────┬─────┘
                     ▼                │
                   DONE          ┌────┴────┐
                                 ▼         ▼
                           ┌──────────┐ ┌──────────┐
                           │ Reviewer │ │Clarifier │
                           │ (LLM)   │ │(3-option │
                           └──────────┘ │  UI)     │
                                        └──────────┘
```

## Tech Stack

- **LangGraph** — StateGraph-based agent orchestration with conditional edges
- **FastAPI** — async API endpoints for session management
- **LangChain + Groq** — LLM inference via llama-3.1-8b-instant
- **Pydantic** — typed state schema with validation
- **Custom Tool System** — registry pattern with lazy loading, concurrent-safe execution

## Key Features

| Feature | Implementation |
|---------|---------------|
| **Task DAG** | Dependency graph with priority levels, auto-scheduling of unblocked tasks |
| **Adaptive Clarification** | Executor detects ambiguity → presents 3 concrete options → pauses → resumes on user answer |
| **Tool Registry** | Plugin architecture — add new tools by registering a function |
| **LLM Review** | Automated quality check after each task, retry on failure |
| **Real-time UI** | SSE-like polling, dark theme, task progress visualization |

## Resume Highlights

- Built a multi-agent task planner using LangGraph StateGraph with autonomous task decomposition and dependency DAGs
- Implemented adaptive clarification system — agent detects ambiguity and presents contextual 3-option choices via interactive UI
- Designed modular tool registry with lazy-loaded, concurrent-safe tools (file ops, shell, search, web)
- Deployed on Render with Docker, PostgreSQL-ready persistence, and CI/CD pipeline

## Quick Start

```bash
# Local
pip install -r requirements.txt
cp .env.example .env  # add your GROQ_API_KEY
uvicorn app.main:app --reload

# Docker
docker build -t task-planner-agent .
docker run -p 8000:8000 --env-file .env task-planner-agent
```

Open http://localhost:8000

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/start` | POST | `{goal, session_id?}` — start planning |
| `/status/{id}` | GET | Get session state |
| `/clarify/{id}` | GET | Get pending clarification |
| `/clarify` | POST | `{session_id, answer}` — answer clarification |
| `/sessions` | List all sessions |
| `/health` | Health check |

## Project Structure

```
Task-Planner-Agent/
├── app/
│   ├── main.py              # FastAPI app + endpoints
│   ├── config.py            # Settings (pydantic-settings)
│   ├── state.py             # TypedDict schemas (AgentState, Task, Clarification)
│   ├── graph.py             # LangGraph StateGraph wiring
│   ├── llm.py               # LLM factory (Groq/OpenAI)
│   ├── nodes/
│   │   ├── planner.py       # Goal → task DAG
│   │   ├── executor.py      # Tool-use loop + clarify detection
│   │   ├── reviewer.py      # LLM-based task verification
│   │   ├── router.py        # Next-action decision
│   │   └── clarifier.py     # 3-option clarification handler
│   ├── tools/
│   │   ├── registry.py      # Tool registry + dispatch
│   │   ├── file_ops.py      # read/write/list/search files
│   │   ├── shell.py         # Sandboxed shell execution
│   │   ├── search.py        # Web search (Tavily/Wikipedia)
│   │   └── web.py           # URL fetch
│   └── memory/
│       └── sessions.py      # In-memory session store
├── static/
│   └── index.html           # Dark-theme interactive UI
├── tests/
├── Dockerfile
├── render.yaml
├── requirements.txt
└── .github/workflows/ci.yml
```

## License

MIT
