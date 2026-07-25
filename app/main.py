"""
FastAPI application — Task Planner Agent API + Web UI.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import get_settings
from app.memory.sessions import store, Session
from app.state import AgentPhase
from app.nodes.clarifier import clarifier_resume, get_pending_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(f"Task Planner Agent starting — provider={settings.llm_provider}, model={settings.llm_model}")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Task Planner Agent",
    description="AI agent that decomposes goals into tasks, executes them with tools, and asks for clarification when ambiguous.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Request/Response models ---

class StartRequest(BaseModel):
    goal: str
    session_id: str | None = None


class ClarifyRequest(BaseModel):
    session_id: str
    answer: str


# --- API Key check ---

def check_api_key(request: Request) -> None:
    settings = get_settings()
    key = request.headers.get("X-API-Key", "")
    if settings.api_key and key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Task Planner Agent</h1><p>UI not found.</p>")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/start")
async def start(req: StartRequest, request: Request):
    check_api_key(request)
    session_id = req.session_id or str(uuid.uuid4())[:12]
    session = store.create(session_id, req.goal)

    # Run planner synchronously
    try:
        from app.graph import get_graph
        graph = get_graph()
        initial_state = session.state
        result = graph.invoke(initial_state)
        # Update session state with graph result
        for k, v in result.items():
            session.state[k] = v
    except Exception as e:
        logger.error(f"Planner failed: {e}")
        session.state["phase"] = AgentPhase.ERROR.value
        session.state["error"] = str(e)

    return session.to_dict()


@app.get("/status/{session_id}")
async def status(session_id: str, request: Request):
    check_api_key(request)
    session = store.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return session.to_dict()


@app.get("/clarify/{session_id}")
async def get_clarification(session_id: str, request: Request):
    check_api_key(request)
    session = store.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    q = get_pending_question(session.state)
    if q is None:
        return {"pending": False}
    return {"pending": True, **q}


@app.post("/clarify")
async def answer_clarification(req: ClarifyRequest, request: Request):
    check_api_key(request)
    session = store.get(req.session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    if session.state.get("phase") != AgentPhase.CLARIFYING.value:
        raise HTTPException(400, "No pending clarification")

    # Merge answer into state
    updates = clarifier_resume(session.state, req.answer)
    store.update_state(req.session_id, updates)

    # Resume executor → reviewer → router chain
    try:
        from app.graph import get_graph
        graph = get_graph()
        result = graph.invoke(session.state)
        for k, v in result.items():
            session.state[k] = v
    except Exception as e:
        logger.error(f"Resume failed: {e}")
        session.state["phase"] = AgentPhase.ERROR.value
        session.state["error"] = str(e)

    return session.to_dict()


@app.get("/sessions")
async def list_sessions(request: Request):
    check_api_key(request)
    return store.list_all()


# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
