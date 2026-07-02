import { expect, it, vi } from "vitest";
import { act, render, screen } from "@testing-library/react";
import HowItWorks from "./HowItWorks";
import { getModelInfo } from "../api";

vi.mock("../api", () => ({ getModelInfo: vi.fn() }));

// The model-spec footer suspends while getModelInfo settles, so React 19
// requires the initial render inside an awaited act().
const renderPage = () => act(async () => render(<HowItWorks />));

it("walks through the pipeline and shows the live model spec footer", async () => {
  vi.mocked(getModelInfo).mockResolvedValue({
    name: "cardiffnlp/twitter-roberta-base-sentiment-latest",
    labels: ["negative", "neutral", "positive"],
    max_tokens: 512,
    device: "mps",
    description: "RoBERTa-base fine-tuned on ~124M tweets for sentiment analysis.",
  });

  await renderPage();

  // The educational spine: tokenization -> encoder -> softmax -> IG ->
  // honest limitations -> why models disagree.
  expect(screen.getByRole("heading", { name: /tokenization/i })).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: /transformer encoder/i })).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: /softmax/i })).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: /integrated gradients/i })).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: /honest limitations/i })).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: /models disagree/i })).toBeInTheDocument();

  // Live spec footer comes from /api/model, not hardcoded copy.
  expect(await screen.findByText("cardiffnlp/twitter-roberta-base-sentiment-latest")).toBeInTheDocument();
  expect(screen.getByText("negative / neutral / positive")).toBeInTheDocument();
  expect(screen.getByText("512")).toBeInTheDocument();
  expect(screen.getByText("mps")).toBeInTheDocument();
});

it("still renders the article when the backend is down, just without the spec footer", async () => {
  vi.mocked(getModelInfo).mockRejectedValue(new Error("Request failed (503)"));

  await renderPage();

  expect(screen.getByRole("heading", { name: /tokenization/i })).toBeInTheDocument();
  expect(screen.queryByText(/max tokens/i)).not.toBeInTheDocument();
});
