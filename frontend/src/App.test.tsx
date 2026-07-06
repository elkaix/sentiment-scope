import { beforeEach, expect, it, vi } from "vitest";
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import { getModelInfo, getModels } from "./api";

// Every tab's data dependency is mocked: hidden tabs are pre-rendered by
// <Activity>, so their fetches fire even before the user visits them.
vi.mock("./api", () => ({
  analyze: vi.fn(),
  explainText: vi.fn(),
  analyzeCsv: vi.fn(),
  getModelInfo: vi.fn(),
  getModels: vi.fn(),
  compareModels: vi.fn(),
  compareAiDetectors: vi.fn(),
}));

beforeEach(() => {
  vi.mocked(getModels).mockResolvedValue({
    models: [
      {
        id: "twitter-roberta",
        name: "cardiffnlp/twitter-roberta-base-sentiment-latest",
        task: "sentiment",
        labels: ["negative", "neutral", "positive"],
        domain: "social / short English text",
        note: "",
        default: true,
        loaded: true,
      },
    ],
  });
  vi.mocked(getModelInfo).mockResolvedValue({
    name: "cardiffnlp/twitter-roberta-base-sentiment-latest",
    labels: ["negative", "neutral", "positive"],
    max_tokens: 512,
    device: "mps",
    description: "RoBERTa-base fine-tuned on tweets.",
  });
});

// Hidden tabs suspend while their promises settle, so React 19 requires the
// initial render inside an awaited act().
const renderApp = () => act(async () => render(<App />));

it("shows the wordmark and opens on the AI validation workspace by default", async () => {
  await renderApp();

  expect(screen.getByText("SentimentScope")).toBeInTheDocument();
  expect(screen.getByText("AI detection validation service")).toBeInTheDocument();
  const tabs = screen.getAllByRole("tab");
  expect(tabs.map((t) => t.getAttribute("aria-label"))).toEqual([
    "Validate AI",
    "Analyze Sentiment",
    "Batch",
    "Compare Sentiment",
    "How it works",
  ]);
  expect(screen.getByRole("tab", { name: "Validate AI" })).toHaveAttribute("aria-selected", "true");
  expect(screen.getByRole("textbox", { name: /text to check for ai authorship/i })).toBeVisible();
});

it("switches the visible panel when a tab is clicked", async () => {
  const user = userEvent.setup();
  await renderApp();

  await user.click(screen.getByRole("tab", { name: "Compare Sentiment" }));
  expect(await screen.findByRole("button", { name: "Compare models" })).toBeVisible();
  expect(screen.getByRole("tab", { name: "Compare Sentiment" })).toHaveAttribute("aria-selected", "true");

  await user.click(screen.getByRole("tab", { name: "How it works" }));
  expect(await screen.findByRole("heading", { name: /tokenization/i })).toBeVisible();
});

it("preserves in-progress tab state when switching away and back", async () => {
  const user = userEvent.setup();
  await renderApp();

  await user.click(screen.getByRole("tab", { name: "Analyze Sentiment" }));
  await user.type(screen.getByPlaceholderText(/type or paste text/i), "battery life is incredible");
  await user.click(screen.getByRole("tab", { name: "How it works" }));
  await user.click(screen.getByRole("tab", { name: "Analyze Sentiment" }));

  expect(screen.getByPlaceholderText(/type or paste text/i)).toHaveValue("battery life is incredible");
});
