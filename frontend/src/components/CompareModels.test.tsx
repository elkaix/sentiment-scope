import { expect, it, vi } from "vitest";
import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CompareModels from "./CompareModels";
import { compareModels, getModels } from "../api";

// CompareModels suspends on first render while the registry promise settles,
// so React 19 requires the initial render inside an awaited act().
const renderCompare = () => act(async () => render(<CompareModels />));

vi.mock("../api", () => ({ compareModels: vi.fn(), getModels: vi.fn() }));

const registry = {
  models: [
    {
      id: "twitter-roberta",
      name: "cardiffnlp/twitter-roberta-base-sentiment-latest",
      labels: ["negative", "neutral", "positive"],
      domain: "social / short English text",
      note: "",
      default: true,
      loaded: true,
    },
    {
      id: "distilbert-sst2",
      name: "distilbert-base-uncased-finetuned-sst-2-english",
      labels: ["negative", "positive"],
      domain: "general binary sentiment",
      note: "",
      default: true,
      loaded: true,
    },
    {
      id: "finbert",
      name: "ProsusAI/finbert",
      labels: ["negative", "neutral", "positive"],
      domain: "financial text",
      note: "Useful for finance/news sentences; misleading outside that domain.",
      default: false,
      loaded: false,
    },
  ],
};

it("checks the two default models and leaves lazy optional models off with a load note", async () => {
  vi.mocked(getModels).mockResolvedValue(registry);
  await renderCompare();

  expect(await screen.findByRole("checkbox", { name: /twitter-roberta/ })).toBeChecked();
  expect(screen.getByRole("checkbox", { name: /distilbert-sst2/ })).toBeChecked();
  expect(screen.getByRole("checkbox", { name: /finbert/ })).not.toBeChecked();
  // Optional models are lazy-loaded server-side; the UI must warn about the
  // one-time load cost before the user opts in.
  expect(screen.getByText(/loads on first use/i)).toBeInTheDocument();
  expect(getModels).toHaveBeenCalledWith("sentiment");
  expect(compareModels).not.toHaveBeenCalled();
});

it("compares the selected models and renders one readout per model with its own score keys", async () => {
  vi.mocked(getModels).mockResolvedValue(registry);
  vi.mocked(compareModels).mockResolvedValue({
    results: [
      {
        model_id: "twitter-roberta",
        name: "cardiffnlp/twitter-roberta-base-sentiment-latest",
        domain: "social / short English text",
        label: "positive",
        scores: { negative: 0.03, neutral: 0.17, positive: 0.8 },
        confidence: 0.8,
        latency_ms: 42.6,
        note: "",
      },
      {
        model_id: "distilbert-sst2",
        name: "distilbert-base-uncased-finetuned-sst-2-english",
        domain: "general binary sentiment",
        label: "negative",
        scores: { negative: 0.61, positive: 0.39 },
        confidence: 0.61,
        latency_ms: 17.2,
        note: "Binary model: it has no neutral class to fall back on.",
      },
    ],
  });

  const user = userEvent.setup();
  await renderCompare();
  await user.click(await screen.findByRole("button", { name: "Compare models" }));

  const roberta = await screen.findByRole("group", { name: "twitter-roberta" });
  const distilbert = screen.getByRole("group", { name: "distilbert-sst2" });

  // Each readout shows its prediction (badge + score bar), confidence
  // (readout + winning bar), and latency.
  expect(within(roberta).getAllByText("positive")).toHaveLength(2);
  expect(within(roberta).getAllByText("80.0%")).toHaveLength(2);
  expect(within(roberta).getByText("43 ms")).toBeInTheDocument();
  expect(within(distilbert).getAllByText("negative")).toHaveLength(2);
  expect(within(distilbert).getByText("17 ms")).toBeInTheDocument();
  expect(within(distilbert).getByText(/no neutral class/)).toBeInTheDocument();

  // Dynamic scores: the 2-class model renders only its own keys.
  expect(within(distilbert).queryByText("neutral")).not.toBeInTheDocument();
  expect(within(roberta).getByText("neutral")).toBeInTheDocument();

  // The default textarea text and the default selection go to the API as-is.
  expect(compareModels).toHaveBeenCalledWith("Our quarterly revenue outlook improved", [
    "twitter-roberta",
    "distilbert-sst2",
  ]);
});

it("includes an opted-in lazy model in the request", async () => {
  vi.mocked(getModels).mockResolvedValue(registry);
  vi.mocked(compareModels).mockResolvedValue({ results: [] });

  const user = userEvent.setup();
  await renderCompare();
  await user.click(await screen.findByRole("checkbox", { name: /finbert/ }));
  await user.click(screen.getByRole("button", { name: "Compare models" }));

  expect(compareModels).toHaveBeenCalledWith("Our quarterly revenue outlook improved", [
    "twitter-roberta",
    "distilbert-sst2",
    "finbert",
  ]);
});

it("surfaces the API error detail when the comparison fails", async () => {
  vi.mocked(getModels).mockResolvedValue(registry);
  vi.mocked(compareModels).mockRejectedValue(new Error("Unknown model id: finbert"));

  const user = userEvent.setup();
  await renderCompare();
  await user.click(await screen.findByRole("button", { name: "Compare models" }));

  expect(await screen.findByText("Unknown model id: finbert")).toBeInTheDocument();
});
