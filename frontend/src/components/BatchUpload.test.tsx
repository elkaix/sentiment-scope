import { expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import BatchUpload from "./BatchUpload";
import { analyzeCsv } from "../api";

vi.mock("../api", () => ({ analyzeCsv: vi.fn() }));

// AggregateCharts renders real recharts SVG, which is irrelevant to what
// BatchUpload itself is responsible for (upload -> loading -> results table
// / error text). Stub it so these tests exercise BatchUpload's own state
// machine, not chart rendering fidelity — the latter is covered separately
// in AggregateCharts.test.tsx.
vi.mock("./AggregateCharts", () => ({
  default: ({ aggregates }: { aggregates: { counts: Record<string, number> } }) => (
    <div data-testid="aggregate-charts">{JSON.stringify(aggregates.counts)}</div>
  ),
}));

const csvFile = () => new File(["text\nI love this\nThis is bad"], "sample.csv", { type: "text/csv" });

it("uploads a CSV and renders the results table and aggregate charts", async () => {
  vi.mocked(analyzeCsv).mockResolvedValue({
    results: [
      { text: "I love this", label: "positive", scores: { negative: 0.05, neutral: 0.15, positive: 0.8 } },
      { text: "This is bad", label: "negative", scores: { negative: 0.7, neutral: 0.2, positive: 0.1 } },
    ],
    aggregates: {
      counts: { negative: 1, positive: 1 },
      mean_scores: { negative: 0.375, neutral: 0.175, positive: 0.45 },
    },
  });

  const user = userEvent.setup();
  const { container } = render(<BatchUpload />);

  // The file input has no accessible label/placeholder/title in the brief's
  // markup, so a container query is the only reliable way to reach it.
  const input = container.querySelector('input[type="file"]') as HTMLInputElement;
  await user.upload(input, csvFile());

  expect(await screen.findByText("I love this")).toBeInTheDocument();
  expect(screen.getByText("This is bad")).toBeInTheDocument();
  expect(screen.getByText("positive")).toBeInTheDocument();
  expect(screen.getByText("negative")).toBeInTheDocument();
  expect(screen.getByText("80.0%")).toBeInTheDocument();
  expect(screen.getByText("70.0%")).toBeInTheDocument();
  expect(screen.getByTestId("aggregate-charts")).toBeInTheDocument();
  expect(analyzeCsv).toHaveBeenCalledWith(csvFile());
});

it("shows the server's error detail and clears it on a subsequent successful upload", async () => {
  vi.mocked(analyzeCsv).mockRejectedValueOnce(new Error("CSV has more than 500 non-empty rows"));

  const user = userEvent.setup();
  const { container } = render(<BatchUpload />);
  const input = container.querySelector('input[type="file"]') as HTMLInputElement;
  await user.upload(input, csvFile());

  expect(await screen.findByText("CSV has more than 500 non-empty rows")).toBeInTheDocument();

  vi.mocked(analyzeCsv).mockResolvedValueOnce({
    results: [{ text: "ok now", label: "neutral", scores: { negative: 0.2, neutral: 0.6, positive: 0.2 } }],
    aggregates: { counts: { neutral: 1 }, mean_scores: { negative: 0.2, neutral: 0.6, positive: 0.2 } },
  });
  await user.upload(input, csvFile());

  expect(await screen.findByText("ok now")).toBeInTheDocument();
  expect(screen.queryByText("CSV has more than 500 non-empty rows")).not.toBeInTheDocument();
});
