"""FastAPI application entrypoint.

Run locally (inside the `ai` conda env):
    uvicorn app.main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.model import SentimentModel
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE for the process lifetime. Every request shares it —
    # safe because inference is read-only (no weight mutation after eval()).
    model = SentimentModel()
    app.state.model = model
    # Tests set SKIP_MODEL_LOAD=1 so the suite runs in milliseconds without torch.
    if os.getenv("SKIP_MODEL_LOAD") != "1":
        model.load()
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
