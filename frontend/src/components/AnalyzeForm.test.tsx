import { expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import AnalyzeForm from "./AnalyzeForm";
import { analyze } from "../api";

vi.mock("../api", () => ({
  analyze: vi.fn(),
  explainText: vi.fn(),
}));

it("disables both buttons until text is entered, then analyzes on click", async () => {
  vi.mocked(analyze).mockResolvedValue({
    label: "positive",
    scores: { negative: 0.05, neutral: 0.15, positive: 0.8 },
  });

  const user = userEvent.setup();
  render(<AnalyzeForm />);

  const analyzeButton = screen.getByRole("button", { name: "Analyze" });
  const explainButton = screen.getByRole("button", { name: "Analyze + Explain" });
  expect(analyzeButton).toBeDisabled();
  expect(explainButton).toBeDisabled();

  await user.type(screen.getByPlaceholderText(/type or paste text/i), "I love this");
  expect(analyzeButton).toBeEnabled();

  await user.click(analyzeButton);

  // "positive" appears twice once results land: the sentiment badge and the
  // ConfidenceBars row label for the positive class.
  expect(await screen.findAllByText("positive")).toHaveLength(2);
  expect(screen.getByText("80.0%")).toBeInTheDocument();
  expect(analyze).toHaveBeenCalledWith("I love this");
});
