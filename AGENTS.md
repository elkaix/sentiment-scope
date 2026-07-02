# Repository Guidelines

## Project Structure & Module Organization

SentimentScope is split into a FastAPI backend and a Vite React frontend. Backend source lives in `backend/app/`, with API routes in `routes.py`, Pydantic schemas in `schemas.py`, and model-loading logic in `model.py` and `model_registry.py`. Backend tests are in `backend/tests/`. Frontend source lives in `frontend/src/`; component tests sit beside components as `*.test.tsx`. Evaluation scripts and fixtures are in `evals/`, screenshots and project notes are in `docs/`, and CSV sample input is in `sample-data/`.

## Build, Test, and Development Commands

- `docker compose up --build`: build and run the full app at `http://localhost:8080`.
- `cd backend && pip install -r requirements.txt`: install runtime backend dependencies.
- `cd backend && uvicorn app.main:app --reload --port 8000`: run the API locally.
- `cd frontend && npm install && npm run dev`: run the Vite dev server at `http://localhost:5173`.
- `cd frontend && npm run build`: type-check and build the frontend.
- `cd frontend && npm run lint`: run Oxlint.

## Coding Style & Naming Conventions

Python uses Ruff settings from `backend/pyproject.toml`: 100-column lines and `E`, `F`, `I`, `W` lint rules. Keep backend modules small and route validation at the API boundary. Frontend code uses TypeScript, React 19, Tailwind, and colocated component files such as `AnalyzeForm.tsx` plus `AnalyzeForm.test.tsx`. Use PascalCase for React components and snake_case for Python functions and test files.

## Testing Guidelines

Run `cd backend && pytest` for unit tests; integration tests are opt-in with `pytest -m integration` and may download real model weights. Run `cd frontend && npm test -- --run` for Vitest and Testing Library tests. Add tests for new endpoints, model-registry behavior, validation branches, and user-visible frontend behavior. Keep frontend tests named `*.test.ts` or `*.test.tsx`.

## Commit & Pull Request Guidelines

Recent history uses short imperative or conventional-style subjects, for example `fix: run CSV batch inference off the event loop` and `chore: add React 19.2 best-practices skill`. Keep subjects concise and explain user-facing behavior or risk in the body when needed. PRs should include a clear summary, linked issue if available, test commands run, and screenshots for UI changes.

## Security & Configuration Tips

Do not hardcode model tokens, API keys, or deployment secrets. The frontend should keep using relative `/api/*` paths so dev, Docker, and Spaces deployment stay aligned. Avoid committing generated caches, model weights, `node_modules/`, or local agent/tooling directories.
