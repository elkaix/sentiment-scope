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

if os.getenv("PUBLIC_DEPLOY") == "1":
    # Public-endpoint abuse guard (Hugging Face Spaces, Task 16A). Lazy import:
    # slowapi ships only in the deployment image (requirements-docker.txt) —
    # dev and CI never take this branch, so they never need it installed.
    from fastapi import Request
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    def client_ip(request: Request) -> str:
        # Spaces terminates TLS at its ingress, so request.client.host is an
        # ingress hop, not the user — and the ingress is a fleet, so per-IP
        # buckets would fragment across hops (verified live: 36 rapid
        # requests, zero 429s). The real client is in X-Forwarded-For. Use
        # the RIGHTMOST entry: it's appended by the trusted ingress itself,
        # while leftmost entries arrive client-controlled — trusting those
        # would let an attacker mint fresh buckets with a spoofed header.
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.rsplit(",", 1)[-1].strip()
        return get_remote_address(request)

    # One global per-IP budget, NOT per-route @limiter.limit decorators —
    # decorators would force the slowapi import whenever routes.py is
    # imported, which breaks the no-slowapi CI env. application_limits (scope
    # "global") shares ONE bucket across every /api path, so hammering seven
    # endpoints doesn't multiply the budget by seven. 30/min covers real
    # interactive usage; /api/explain (~50 forward passes per call) is what
    # this protects.
    limiter = Limiter(key_func=client_ip, application_limits=["30/minute"])
    app.state.limiter = limiter
    # No app.add_exception_handler(RateLimitExceeded, ...): the middleware below
    # catches RateLimitExceeded and returns the 429 itself, so the exception
    # never propagates to an app-level handler — registering one would be dead code.

    # NOT slowapi's SlowAPIMiddleware: it resolves the handler by scanning
    # app.routes for an `endpoint` attribute, but FastAPI 0.139 wraps included
    # routers in an _IncludedRouter object without one — so it silently
    # exempts every request (verified empirically). This thin middleware
    # drives the same Limiter by path prefix instead: only /api/* spends
    # budget, so the static SPA assets under "/" stay free.
    @app.middleware("http")
    async def api_rate_limit(request: Request, call_next):
        if request.url.path.startswith("/api"):
            try:
                limiter._check_request_limit(request, None, True)
            except RateLimitExceeded as exc:
                return _rate_limit_exceeded_handler(request, exc)
        return await call_next(request)

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

# Same app, three serving topologies — the frontend never changes because it
# only ever calls relative /api/... paths:
#   1. Dev:      the Vite dev server proxies /api -> uvicorn (vite.config.ts).
#   2. Compose:  nginx serves the built SPA and proxies /api -> this backend.
#   3. Spaces:   a Space is exactly ONE container, so FastAPI itself serves
#                the built SPA via the StaticFiles mount below.
# Mount order matters: /api routes are matched first because the router is
# registered before the static mount at "/". STATIC_DIR is only set in the
# Spaces image — dev and compose don't use this.
static_dir = os.getenv("STATIC_DIR")
if static_dir:
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="spa")
