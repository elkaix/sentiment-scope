import { afterEach, describe, expect, it, vi } from "vitest";
import { analyze } from "./api";

describe("api client", () => {
  afterEach(() => vi.restoreAllMocks());

  it("returns parsed JSON on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ label: "positive", scores: { negative: 0.1, neutral: 0.1, positive: 0.8 } })),
    ));
    const result = await analyze("great stuff");
    expect(result.label).toBe("positive");
  });

  it("throws the server's detail message on error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "Model is not loaded" }), { status: 503 }),
    ));
    await expect(analyze("hi")).rejects.toThrow("Model is not loaded");
  });
});
