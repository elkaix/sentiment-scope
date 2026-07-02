import { expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import ConfidenceBars from "./ConfidenceBars";

it("renders one bar per class with percentages", () => {
  render(<ConfidenceBars scores={{ negative: 0.05, neutral: 0.15, positive: 0.8 }} />);
  expect(screen.getByText("negative")).toBeInTheDocument();
  expect(screen.getByText("neutral")).toBeInTheDocument();
  expect(screen.getByText("positive")).toBeInTheDocument();
  expect(screen.getByText("80.0%")).toBeInTheDocument();
});
