# Current Task

## SaaS AI Detection UI Polish

**Goal:** Make the React UI open as a professional AI detection validation service while preserving the existing sentiment, batch, compare, and explainer workflows.

**Acceptance criteria**
- [x] The default workspace is AI validation, not sentiment analysis.
- [x] The shell reads as a SaaS product workbench on desktop and mobile.
- [x] AI detector copy avoids proof-of-authorship framing and generic AI-marketing tells.
- [x] Heavy batch charts are not loaded before a batch result exists.
- [x] Frontend tests, lint, build, screenshots, and designer gate pass.

## Review

**Changed**
- `frontend/src/App.tsx`: widened the app shell, made AI validation the default tab, added compact service metadata, and fixed mobile tabs to avoid clipped labels.
- `frontend/src/components/AiTextDetector.tsx`: tightened detector sample/copy and ensured the submit control meets the 44px touch-target floor.
- `frontend/src/components/BatchUpload.tsx`: lazy-loads `AggregateCharts` after CSV results so Recharts is split out of the initial bundle.
- `frontend/vite.config.ts`: removed a stale Vitest triple-slash directive.
- Updated focused Vitest coverage for the default AI-validation workspace and lazy chart loading.

**Verification**
- `cd frontend && npm test -- --run` -> 10 files, 26 tests passed.
- `cd frontend && npm run lint` -> passed with no warnings.
- `cd frontend && npm run build` -> passed; initial JS chunk is now 216.30 kB gzip 67.44 kB, with charts split separately.
- Playwright CLI screenshots captured at 1440x1000 and 390x900 using Chrome.
- Designer gate on `frontend/src` -> PASS, score 100/100, 0 findings.

**Deferred**
- No backend/API changes were needed for this UI pass.

## AI Detection Highlight

**Goal:** When a detector labels submitted text as AI, show the reviewed passage with a clear color highlight.

**Acceptance criteria**
- [x] The highlight appears only after detector results include an `ai` label.
- [x] The highlighted passage preserves the submitted text snapshot.
- [x] The UI states confidence and avoids claiming word-level evidence.
- [x] Verification commands rerun after implementation.
