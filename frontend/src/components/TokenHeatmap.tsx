import type { TokenAttribution } from "../api";

/**
 * Renders the input text token-by-token, tinted by Integrated Gradients
 * attribution: green = pushed the model toward its prediction, red = pushed
 * against it, intensity = relative magnitude. Magnitudes are scaled against
 * the largest attribution in THIS sentence — colors show relative influence
 * within one input and aren't comparable across inputs.
 */
export default function TokenHeatmap({ tokens }: { tokens: TokenAttribution[] }) {
  const maxAbs = Math.max(...tokens.map((t) => Math.abs(t.attribution)), 1e-6);

  return (
    <p className="leading-8">
      {tokens.map((t, i) => {
        const strength = Math.abs(t.attribution) / maxAbs;
        const color =
          t.attribution >= 0
            ? `rgba(16, 185, 129, ${(0.15 + 0.7 * strength).toFixed(2)})`
            : `rgba(239, 68, 68, ${(0.15 + 0.7 * strength).toFixed(2)})`;
        // Tokens keep their leading-space marker from the backend; trim for
        // display but re-add spacing via margin so words don't run together.
        const display = t.token.trimStart();
        const leadingSpace = t.token.startsWith(" ");
        return (
          <span
            key={i}
            className={`rounded px-0.5 ${leadingSpace ? "ml-1" : ""}`}
            style={{ backgroundColor: color }}
            title={`attribution: ${t.attribution.toFixed(3)}`}
          >
            {display}
          </span>
        );
      })}
    </p>
  );
}
