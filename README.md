# Task Planner Agent

An AI agent that takes a high-level goal, breaks it into smaller tasks, figures out the order to do them in, and executes them one by one — using real tools like file operations, shell commands, and web search.

But here's the interesting part: if the agent gets stuck or needs your input mid-execution, it pauses and shows you **3 clear options** to choose from in a clean UI. You pick one, and it keeps going.

## How It Works

You give it a goal like *"Build a REST API with authentication and tests"*. Here's what happens behind the scenes:

1. **Planning** — An LLM breaks your goal into concrete subtasks with dependencies. It figures out that you need to create the database schema before you can add authentication, and so on.

2. **Execution** — The agent picks up each task and uses tools to complete it. It can read and write files, run shell commands, search the web, and fetch URLs.

3. **Clarification** — When the agent hits something ambiguous (like "which database should I use?"), it doesn't guess. Instead, it shows you a card with 3 options:

   ```
   ┌─────────────────────────────────────────────┐
   │  CLARIFICATION NEEDED                       │
   │                                             │
   │  Which database should I use for auth?      │
   │                                             │
   │  ┌─────────────────────────────────────┐    │
   │  │ a │ SQLite + bcrypt                 │    │
   │  │   │ Simple, file-based              │    │
   │  ├───┼─────────────────────────────────┤    │
   │  │ b │ PostgreSQL + JWT                │    │
   │  │   │ Production-ready, scalable      │    │
   │  ├───┼─────────────────────────────────┤    │
   │  │ c │ Firebase Auth                   │    │
   │  │   │ Fully managed, no backend       │    │
   │  └─────────────────────────────────────┘    │
   └─────────────────────────────────────────────┘
   ```

   You click an option (or type your own answer), and the agent continues with that context.

4. **Review** — After each task, an LLM reviews whether it was actually completed correctly. If not, it retries or replans.

5. **Completion** — Once all tasks are done, you get a summary of everything that was accomplished.

## The Tech

- **LangGraph** — Orchestrates the agent loop using a state machine (StateGraph). Each node (planner, executor, reviewer, router) is a step in the graph, with conditional edges that determine what happens next.

- **FastAPI** — Powers the API endpoints and serves the web UI. Everything runs as a single service.

- **LangChain + Groq** — The LLM backbone. Uses Groq's free tier for fast inference (currently llama-3.3-70b via OpenRouter). Supports OpenRouter free models too.

- **Custom Tool System** — Tools are registered in a plugin-style registry. Each tool is just a function that takes a string and returns a string. New tools can be added by dropping a file in the tools folder.

## Getting Started

```bash
# Clone the repo
git clone https://github.com/Aditya-k63/-Task-Planner-Agent.git
cd -Task-Planner-Agent

# Install dependencies
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY or OPENROUTER_API_KEY

# Run it
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Using Docker

```bash
docker build -t task-planner-agent .
docker run -p 8000:8000 --env-file .env task-planner-agent
```

## API Endpoints

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/` | GET | The web UI |
| `/start` | POST | Send a goal, get back a session with tasks |
| `/status/{id}` | GET | Check what the agent is doing right now |
| `/clarify/{id}` | GET | Get the pending question (if any) |
| `/clarify` | POST | Answer a clarification question |
| `/sessions` | GET | List all active sessions |
| `/health` | GET | Health check |

## Project Structure

```
Task-Planner-Agent/
├── app/
│   ├── main.py              # FastAPI app and endpoints
│   ├── config.py            # Environment settings
│   ├── state.py             # Data schemas (AgentState, Task, Clarification)
│   ├── graph.py             # LangGraph wiring — connects all nodes
│   ├── llm.py               # LLM provider setup (Groq/OpenRouter)
│   ├── nodes/
│   │   ├── planner.py       # Turns goals into task lists
│   │   ├── executor.py      # Runs tools and detects when to ask for help
│   │   ├── reviewer.py      # Checks if tasks were done correctly
│   │   ├── router.py        # Decides what to do next
│   │   └── clarifier.py     # Handles the 3-option question flow
│   ├── tools/
│   │   ├── registry.py      # Tool registration and dispatch
│   │   ├── file_ops.py      # Read, write, list, search files
│   │   ├── shell.py         # Run shell commands safely
│   │   ├── search.py        # Search the web
│   │   └── web.py           # Fetch URLs
│   └── memory/
│       └── sessions.py      # Store session data
├── static/
│   └── index.html           # The web UI
├── tests/                   # Unit tests
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml
```

## License

MIT
