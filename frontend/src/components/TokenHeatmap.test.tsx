import { expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import TokenHeatmap from "./TokenHeatmap";

it("colors positive attributions green and negative red, and keeps tooltips", () => {
  render(
    <TokenHeatmap
      tokens={[
        { token: " love", attribution: 0.9 },
        { token: " not", attribution: -0.5 },
      ]}
    />,
  );
  const love = screen.getByText("love");
  const not = screen.getByText("not");
  expect(love.style.backgroundColor).toContain("16, 185, 129"); // emerald
  expect(not.style.backgroundColor).toContain("239, 68, 68"); // red
  expect(love.title).toBe("attribution: 0.900");
  expect(not.title).toBe("attribution: -0.500");
});

it("wraps at word boundaries but keeps subword tokens joined in one atomic group", () => {
  const { container } = render(
    <TokenHeatmap
      tokens={[
        { token: "I", attribution: 0.1 },
        { token: " love", attribution: 0.4 },
        { token: " Fa", attribution: 0.2 },
        { token: "ble", attribution: 0.6 },
      ]}
    />,
  );

  // Outer container must be a wrapping flex row so long inputs break onto
  // new lines instead of overflowing as one unbreakable line.
  const outer = container.querySelector("p");
  expect(outer?.className).toContain("flex");
  expect(outer?.className).toContain("flex-wrap");
  expect(outer?.className).toContain("gap-x"); // horizontal spacing between words

  // "I", "love", "Fable" — three words, "Fable" made of two BPE subtokens.
  const groups = container.querySelectorAll<HTMLElement>('[data-testid="word-group"]');
  expect(groups).toHaveLength(3);

  const fableGroup = groups[2];
  expect(within(fableGroup).getByText("Fa")).toBeInTheDocument();
  expect(within(fableGroup).getByText("ble")).toBeInTheDocument();
});

it("renders newline tokens as a break instead of literal Ċ text", () => {
  const { container } = render(
    <TokenHeatmap
      tokens={[
        { token: "it", attribution: 0.1 },
        { token: "Ċ", attribution: 0.0 },
        { token: "Ċ", attribution: 0.0 },
        { token: " 3", attribution: 0.2 },
      ]}
    />,
  );

  expect(screen.queryByText("Ċ")).not.toBeInTheDocument();
  expect(screen.queryByText(/Ċ/)).not.toBeInTheDocument();
  expect(container.querySelectorAll('[data-testid="newline-break"]')).toHaveLength(2);
  expect(screen.getByText("it")).toBeInTheDocument();
  expect(screen.getByText("3")).toBeInTheDocument();
});
