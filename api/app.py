"""
api/app.py
----------
FastAPI wrapper that exposes the LangChain agent as a REST API.

Endpoints:
  POST /query     — submit a natural language question, get an answer
  GET  /health    — liveness check for Docker health checks
  GET  /lineage   — return the last N lineage records from the log

Run locally:
    uvicorn api.app:app --reload --port 8000
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent.agent import build_agent

# ---------------------------------------------------------------------------
# Lifespan — builds the agent once at startup using the modern FastAPI pattern
# ---------------------------------------------------------------------------

agent_executor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor
    agent_executor = build_agent(verbose=False)
    yield
    # shutdown logic here if needed

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Financial Signal Agent API",
    description = "LLM-powered agent querying stock prices, news sentiment, and SEC filings",
    version     = "1.0.0",
    lifespan    = lifespan
)

# Allow browser requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Serve the frontend at the root URL
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")

LINEAGE_PATH = Path("./data/lineage/lineage_log.jsonl")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question   : str
    session_id : Optional[str] = "default"


class QueryResponse(BaseModel):
    answer     : str
    tools_used : list[str]
    sources    : list[str]
    latency_ms : float
    session_id : str


class HealthResponse(BaseModel):
    status    : str
    timestamp : str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
def health_check():
    """Liveness check. Returns 200 if the service is running."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/query", response_model=QueryResponse, tags=["Agent"])
def query_agent(request: QueryRequest):
    """Submit a natural language question to the agent.

    The agent decides which tool(s) to call, queries the underlying data
    pipeline, and returns a grounded answer with source citations.

    Example request body:
        { "question": "Is there a divergence between NVDA news sentiment and price?" }
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start_time = datetime.utcnow()

    try:
        result = agent_executor.invoke({"input": request.question})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

    tools_used = [
        step[0].tool
        for step in result.get("intermediate_steps", [])
    ]

    sources = []
    for step in result.get("intermediate_steps", []):
        observation = step[1]
        try:
            obs_json = json.loads(observation)
            for item in obs_json.get("results", []):
                src = item.get("source", "")
                if src and src not in sources:
                    sources.append(src)
        except (json.JSONDecodeError, AttributeError):
            pass

    return QueryResponse(
        answer     = result["output"],
        tools_used = tools_used,
        sources    = sources,
        latency_ms = round(elapsed_ms, 1),
        session_id = request.session_id
    )


@app.get("/lineage", tags=["Operations"])
def get_lineage(limit: int = 20):
    """Return the most recent N records from the data lineage log."""
    if not LINEAGE_PATH.exists():
        return {"records": [], "message": "Lineage log not found — run the pipeline first"}

    records = []
    with open(LINEAGE_PATH) as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass

    return {"records": list(reversed(records))[:limit], "total": len(records)}