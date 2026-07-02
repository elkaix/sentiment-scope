import type { ReactNode } from "react";
import { expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import AggregateCharts from "./AggregateCharts";

// Recharts renders raw SVG using canvas text measurement and ResizeObserver
// for layout, neither of which jsdom implements — real charts either render
// empty (0×0 container) or, once forced to a fixed size, still collapse axis
// tick labels down to a single one because jsdom can't measure text width.
// Stubbing the recharts primitives lets us test what AggregateCharts is
// actually responsible for — turning `aggregates` into per-label chart
// rows — without depending on jsdom's incomplete SVG support.
vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: ReactNode }) => <>{children}</>,
  BarChart: ({ data }: { data: Array<Record<string, unknown>> }) => (
    <ul>
      {data.map((row) => (
        <li key={String(row.label)}>{`${row.label}:${row.count ?? row.mean}`}</li>
      ))}
    </ul>
  ),
  Bar: () => null,
  Cell: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
}));

it("maps aggregates.counts and aggregates.mean_scores into per-label chart rows", () => {
  render(
    <AggregateCharts
      aggregates={{
        counts: { negative: 2, neutral: 1, positive: 3 },
        mean_scores: { negative: 0.4, neutral: 0.5, positive: 0.9 },
      }}
    />,
  );

  expect(screen.getByText("Sentiment counts")).toBeInTheDocument();
  expect(screen.getByText("Mean confidence per class")).toBeInTheDocument();

  expect(screen.getByText("negative:2")).toBeInTheDocument();
  expect(screen.getByText("neutral:1")).toBeInTheDocument();
  expect(screen.getByText("positive:3")).toBeInTheDocument();
  expect(screen.getByText("negative:0.4")).toBeInTheDocument();
  expect(screen.getByText("neutral:0.5")).toBeInTheDocument();
  expect(screen.getByText("positive:0.9")).toBeInTheDocument();
});
