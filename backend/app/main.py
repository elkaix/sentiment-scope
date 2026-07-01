"""FastAPI application entrypoint.

Run locally (inside the `ai` conda env):
    uvicorn app.main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.model import SentimentModel
from app.model_registry import ModelTask, get_default_model_id
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the default model ONCE for the process lifetime. Every request
    # shares it — safe because inference is read-only (no weight mutation
    # after eval()).
    model = SentimentModel()
    app.state.model = model
    # Optional registry models (for /api/compare, Task 9) are loaded lazily on
    # first use and memoized here — loading all four transformers at startup
    # would blow up memory on a laptop. The per-model locks stop two concurrent
    # requests from loading the same weights twice. See app.model.get_or_load_model.
    app.state.model_cache = {}
    app.state.model_locks = {}
    # Tests set SKIP_MODEL_LOAD=1 so the suite runs in milliseconds without torch.
    if os.getenv("SKIP_MODEL_LOAD") != "1":
        model.load()
        # Seed the eagerly-loaded default into the lazy cache so /api/compare
        # shares this ONE copy of the ~500MB weights instead of loading a second.
        # Only after a real load — an unloaded model here would poison the cache.
        app.state.model_cache[get_default_model_id(ModelTask.SENTIMENT)] = model
    yield


app = FastAPI(title="SentimentScope API", lifespan=lifespan)

# CORS is only needed when the frontend is served from a different origin
# (npm dev server without the proxy, or direct API access). The Vite proxy
# and nginx make requests same-origin, but this keeps direct access working.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
