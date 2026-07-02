import { expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import AiTextDetector from "./AiTextDetector";
import { compareAiDetectors } from "../api";
import type { AiDetectItem } from "../api";

// The detector tab talks to exactly one endpoint: compare-all-detectors.
vi.mock("../api", () => ({ compareAiDetectors: vi.fn() }));

// The verbatim uncertainty string the backend attaches to every response.
// The tab must render THIS string, not a hardcoded copy of it.
const WARNING =
  "AI detectors are probabilistic and can be wrong, especially on short, edited, " +
  "non-native, highly formal, or mixed-authorship text. Do not use this as proof of authorship.";

const item = (over: Partial<AiDetectItem> = {}): AiDetectItem => ({
  model_id: "desklib-ai-detector",
  name: "desklib-ai-text-detector-v1.01",
  domain: "general AI-written text detection",
  label: "ai",
  scores: { human: 0.1, ai: 0.9 },
  confidence: 0.9,
  latency_ms: 120.4,
  note: "Default detector.",
  ...over,
});

it("shows the probabilistic-signal disclaimer before any detection runs", () => {
  render(<AiTextDetector />);

  expect(screen.getByText(/not proof of authorship/i)).toBeInTheDocument();
  expect(compareAiDetectors).not.toHaveBeenCalled();
});

it("detects across all detectors by default and renders the API's own warning", async () => {
  vi.mocked(compareAiDetectors).mockResolvedValue({
    results: [item()],
    disagreement: false,
    warning: WARNING,
  });

  const user = userEvent.setup();
  render(<AiTextDetector />);
  await user.clear(screen.getByRole("textbox"));
  await user.type(screen.getByRole("textbox"), "some text to check");
  await user.click(screen.getByRole("button", { name: /detect ai text/i }));

  // Default = compare across ALL detectors: no model_ids argument.
  expect(compareAiDetectors).toHaveBeenCalledWith("some text to check");
  // The prominent callout renders the string the backend actually returned.
  expect(await screen.findByText(WARNING)).toBeInTheDocument();
});

it("renders a readout per detector with its verdict, scores, latency, and note", async () => {
  vi.mocked(compareAiDetectors).mockResolvedValue({
    results: [
      item({
        model_id: "desklib-ai-detector",
        name: "desklib-ai-text-detector-v1.01",
        label: "ai",
        scores: { human: 0.02, ai: 0.98 },
        confidence: 0.98,
        latency_ms: 130,
        note: "Default detector.",
      }),
      item({
        model_id: "fakespot-ai-detector",
        name: "fakespot-roberta-base-ai-text-detection-v1",
        label: "human",
        scores: { human: 0.7, ai: 0.3 },
        confidence: 0.7,
        latency_ms: 45,
        note: "RoBERTa-based detector.",
      }),
    ],
    disagreement: true,
    warning: WARNING,
  });

  const user = userEvent.setup();
  render(<AiTextDetector />);
  await user.click(screen.getByRole("button", { name: /detect ai text/i }));

  const desklib = await screen.findByRole("group", { name: "desklib-ai-detector" });
  const fakespot = screen.getByRole("group", { name: "fakespot-ai-detector" });

  expect(within(desklib).getByText("desklib-ai-text-detector-v1.01")).toBeInTheDocument();
  expect(within(desklib).getByText("general AI-written text detection")).toBeInTheDocument();
  expect(within(desklib).getByText("2.0%")).toBeInTheDocument(); // P(human)
  expect(within(desklib).getAllByText("98.0%").length).toBeGreaterThan(0); // P(ai) + confidence
  expect(within(desklib).getByText("130 ms")).toBeInTheDocument();
  expect(within(desklib).getByText("Default detector.")).toBeInTheDocument();

  expect(within(fakespot).getByText("45 ms")).toBeInTheDocument();
  expect(within(fakespot).getByText(/RoBERTa-based/)).toBeInTheDocument();
});

it("shows a disagreement indicator when detectors reach different verdicts", async () => {
  vi.mocked(compareAiDetectors).mockResolvedValue({
    results: [
      item({ model_id: "a", label: "ai" }),
      item({ model_id: "b", label: "human", scores: { human: 0.8, ai: 0.2 } }),
    ],
    disagreement: true,
    warning: WARNING,
  });

  const user = userEvent.setup();
  render(<AiTextDetector />);
  await user.click(screen.getByRole("button", { name: /detect ai text/i }));

  expect(await screen.findByText(/detectors disagree/i)).toBeInTheDocument();
});

it("omits the disagreement indicator when the detectors agree", async () => {
  vi.mocked(compareAiDetectors).mockResolvedValue({
    results: [item({ model_id: "a", label: "ai" }), item({ model_id: "b", label: "ai" })],
    disagreement: false,
    warning: WARNING,
  });

  const user = userEvent.setup();
  render(<AiTextDetector />);
  await user.click(screen.getByRole("button", { name: /detect ai text/i }));

  await screen.findByRole("group", { name: "a" });
  expect(screen.queryByText(/detectors disagree/i)).not.toBeInTheDocument();
});

it("surfaces the API error detail when a detector is disabled on the public deploy", async () => {
  vi.mocked(compareAiDetectors).mockRejectedValue(
    new Error("Model 'fakespot-ai-detector' is disabled on the public deployment."),
  );

  const user = userEvent.setup();
  render(<AiTextDetector />);
  await user.click(screen.getByRole("button", { name: /detect ai text/i }));

  expect(await screen.findByText(/disabled on the public deployment/i)).toBeInTheDocument();
});
