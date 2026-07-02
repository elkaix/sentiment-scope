import { expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import TokenHeatmap from "./TokenHeatmap";

it("colors positive attributions green and negative red", () => {
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
});
