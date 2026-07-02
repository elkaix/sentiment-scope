"""Tests for the Spaces single-image deployment wiring in app.main (Task 16A):
the STATIC_DIR conditional SPA mount and the PUBLIC_DEPLOY rate limiter.

Both switches are read at import time in app.main (the `app` object is built
at module level), so monkeypatching the env after conftest imported the app
does nothing. Each test instead reloads app.main under a patched environment
and reloads once more afterwards, so every other test module keeps seeing a
default app with neither switch active.
"""

import importlib
import os

import pytest
from fastapi.testclient import TestClient

import app.main


def _reload_with(env: dict[str, str]):
    for key, value in env.items():
        os.environ[key] = value
    return importlib.reload(app.main)


def _restore(env_keys: list[str]) -> None:
    for key in env_keys:
        os.environ.pop(key, None)
    importlib.reload(app.main)


def test_static_dir_serves_spa_and_api_still_wins(tmp_path):
    """With STATIC_DIR set, GET / returns index.html while /api/* keeps hitting
    the API — the router is registered before the "/" mount, so mount order
    guarantees API precedence."""
    (tmp_path / "index.html").write_text("<html><body>sentiment-scope-spa</body></html>")
    module = _reload_with({"STATIC_DIR": str(tmp_path)})
    try:
        with TestClient(module.app) as c:
            root = c.get("/")
            assert root.status_code == 200
            assert "sentiment-scope-spa" in root.text

            health = c.get("/api/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"
    finally:
        _restore(["STATIC_DIR"])


def test_public_deploy_rate_limits_api_with_429():
    """PUBLIC_DEPLOY=1 arms the global 30/minute per-IP budget. slowapi ships
    only in the deployment image (requirements-docker.txt), so CI — which
    installs requirements-dev.txt — skips this test instead of failing."""
    pytest.importorskip("slowapi")
    module = _reload_with({"PUBLIC_DEPLOY": "1"})
    try:
        with TestClient(module.app) as c:
            statuses = [c.get("/api/health").status_code for _ in range(31)]
            assert statuses[:30] == [200] * 30
            assert statuses[30] == 429
    finally:
        _restore(["PUBLIC_DEPLOY"])


def test_rate_limit_buckets_key_on_rightmost_forwarded_for():
    """Behind the Spaces ingress the client IP arrives in X-Forwarded-For.
    Buckets must key on the RIGHTMOST entry (appended by the trusted ingress);
    leftmost entries are client-controlled, so keying on them would let a
    spoofed header mint unlimited fresh buckets."""
    pytest.importorskip("slowapi")
    module = _reload_with({"PUBLIC_DEPLOY": "1"})
    try:
        with TestClient(module.app) as c:
            exhaust = {"X-Forwarded-For": "6.6.6.6, 9.9.9.9"}
            statuses = [
                c.get("/api/health", headers=exhaust).status_code for _ in range(31)
            ]
            assert statuses[30] == 429

            # Same trusted (rightmost) hop, different spoofed leftmost entry:
            # still the SAME bucket — the spoof buys no fresh budget.
            spoof = {"X-Forwarded-For": "1.2.3.4, 9.9.9.9"}
            assert c.get("/api/health", headers=spoof).status_code == 429

            # A genuinely different client (different rightmost entry) is
            # unaffected by the exhausted bucket.
            other = {"X-Forwarded-For": "6.6.6.6, 7.7.7.7"}
            assert c.get("/api/health", headers=other).status_code == 200
    finally:
        _restore(["PUBLIC_DEPLOY"])


def test_default_app_has_no_limiter_or_static_mount():
    """Neither switch set (dev/CI default): no 429s ever, and GET / is a 404 —
    the SPA is served by Vite (dev) or nginx (compose), not FastAPI."""
    for key in ("PUBLIC_DEPLOY", "STATIC_DIR"):
        assert os.getenv(key) is None
    with TestClient(app.main.app) as c:
        statuses = {c.get("/api/health").status_code for _ in range(35)}
        assert statuses == {200}
        assert c.get("/").status_code == 404
