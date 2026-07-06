import { Activity, useRef, useState } from "react";
import type { ReactElement } from "react";
import AiTextDetector from "./components/AiTextDetector";
import AnalyzeForm from "./components/AnalyzeForm";
import BatchUpload from "./components/BatchUpload";
import CompareModels from "./components/CompareModels";
import HowItWorks from "./components/HowItWorks";

const TABS = [
  "Validate AI",
  "Analyze Sentiment",
  "Batch",
  "Compare Sentiment",
  "How it works",
] as const;
type Tab = (typeof TABS)[number];

// Panels are static JSX, hoisted so the elements are created once.
const PANELS: Record<Tab, ReactElement> = {
  "Validate AI": <AiTextDetector />,
  "Analyze Sentiment": <AnalyzeForm />,
  Batch: <BatchUpload />,
  "Compare Sentiment": <CompareModels />,
  "How it works": <HowItWorks />,
};

const slug = (t: Tab) => t.toLowerCase().replace(/\W+/g, "-");

export default function App() {
  const [tab, setTab] = useState<Tab>("Validate AI");
  const tabRefs = useRef(new Map<Tab, HTMLButtonElement | null>());

  // Roving tabindex per the WAI-ARIA tabs pattern: arrows move selection and
  // focus together; only the active tab is in the page tab order.
  const move = (dir: 1 | -1) => {
    const next = TABS[(TABS.indexOf(tab) + dir + TABS.length) % TABS.length];
    setTab(next);
    tabRefs.current.get(next)?.focus();
  };

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 border-b border-black/5 bg-white/85 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center gap-3 px-6 py-4">
          {/* Wordmark device: the three sentiment classes as a tiny bar
              readout. This is the only place chroma appears outside data. */}
          <span aria-hidden="true" className="flex items-end gap-[3px]">
            <span className="h-2 w-1.5 rounded-[2px] bg-red-500" />
            <span className="h-3 w-1.5 rounded-[2px] bg-amber-400" />
            <span className="h-4 w-1.5 rounded-[2px] bg-emerald-500" />
          </span>
          <div className="flex flex-wrap items-baseline gap-x-3">
            <h1 className="text-lg font-semibold tracking-tight text-slate-900">SentimentScope</h1>
            <p className="text-xs text-slate-500">AI detection validation service</p>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-3xl px-6 pb-16 pt-8">
        <div
          role="tablist"
          aria-label="SentimentScope views"
          className="mb-6 inline-flex flex-wrap gap-1 rounded-lg border border-black/5 bg-white p-1 shadow-[0_1px_2px_rgba(15,23,42,0.04)]"
        >
          {TABS.map((t) => (
            <button
              key={t}
              role="tab"
              aria-label={t}
              id={`tab-${slug(t)}`}
              aria-selected={tab === t}
              aria-controls={`panel-${slug(t)}`}
              tabIndex={tab === t ? 0 : -1}
              ref={(el) => {
                tabRefs.current.set(t, el);
              }}
              onClick={() => setTab(t)}
              onKeyDown={(e) => {
                if (e.key === "ArrowRight") move(1);
                if (e.key === "ArrowLeft") move(-1);
              }}
              className={`rounded-md px-3.5 py-2 text-sm font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900 ${
                tab === t
                  ? "bg-slate-900 text-white"
                  : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        <main>
          {/* Activity keeps every panel mounted: a half-typed sentence on
              Analyze or an uploaded batch survives a trip to How it works. */}
          {TABS.map((t) => (
            <Activity key={t} mode={tab === t ? "visible" : "hidden"}>
              <section
                role="tabpanel"
                id={`panel-${slug(t)}`}
                aria-labelledby={`tab-${slug(t)}`}
                className="rounded-2xl border border-black/5 bg-white p-6 shadow-[0_1px_2px_rgba(15,23,42,0.04)] sm:p-8"
              >
                {PANELS[t]}
              </section>
            </Activity>
          ))}
        </main>
      </div>
    </div>
  );
}
