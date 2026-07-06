# SentimentScope — frontend

React + TypeScript + Vite SPA for SentimentScope. See the [root README](../README.md)
for the full project overview, architecture, and deployment topologies.

## Dev commands

```bash
npm install       # install dependencies
npm run dev       # Vite dev server (proxies /api -> backend on :8002)
npm test -- --run # run the vitest suite once
npm run lint      # oxlint
npm run build     # type-check (tsc -b) + production build
```

The UI only ever calls relative `/api/...` paths; the Vite dev proxy, nginx
(compose), and FastAPI static mount (Spaces) each wire those to the backend.
