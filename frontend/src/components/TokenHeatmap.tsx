import type { TokenAttribution } from "../api";

/**
 * Renders the input text token-by-token, tinted by Integrated Gradients
 * attribution: green = pushed the model toward its prediction, red = pushed
 * against it, intensity = relative magnitude. Magnitudes are scaled against
 * the largest attribution in THIS sentence — colors show relative influence
 * within one input and aren't comparable across inputs.
 *
 * Tokens keep their BPE markers from the backend: a leading space (mapped
 * from "Ġ") starts a new word, so a run of tokens with no leading space are
 * subword pieces of the SAME word (e.g. " Fa" + "ble" → "Fable"). Those are
 * grouped into one atomic, non-wrapping unit so readers can see BPE splits
 * inside a word, while the line only ever breaks BETWEEN words — CSS margin
 * spacing alone can't do this because a margin isn't a soft-wrap opportunity,
 * so a long input would render as one unbreakable line. A newline marker
 * ("Ċ") renders as an explicit line break instead of a literal "Ċ" glyph.
 */

const NEWLINE_MARKER = "Ċ";
const TAB_MARKER = "ĉ";

/** True if `str` is one or more repetitions of `marker` and nothing else. */
function isOnlyMarker(str: string, marker: string): boolean {
  return str.length > 0 && [...str].every((ch) => ch === marker);
}

function tint(attribution: number, maxAbs: number): string {
  const strength = Math.abs(attribution) / maxAbs;
  return attribution >= 0
    ? `rgba(16, 185, 129, ${(0.15 + 0.7 * strength).toFixed(2)})`
    : `rgba(239, 68, 68, ${(0.15 + 0.7 * strength).toFixed(2)})`;
}

/** Strip the leading BPE space marker; map a lone tab marker to a space. */
function displayText(token: string): string {
  const stripped = token.trimStart();
  return isOnlyMarker(stripped, TAB_MARKER) ? " " : stripped;
}

type TokenWithIndex = { token: TokenAttribution; index: number };
type Group =
  | { kind: "word"; tokens: TokenWithIndex[] }
  | { kind: "break"; token: TokenAttribution; index: number };

/** Groups tokens into words (joined subtokens) and newline breaks. */
function groupTokens(tokens: TokenAttribution[]): Group[] {
  const groups: Group[] = [];
  tokens.forEach((token, index) => {
    const leadingSpace = token.token.startsWith(" ");
    const rest = leadingSpace ? token.token.slice(1) : token.token;
    if (isOnlyMarker(rest, NEWLINE_MARKER)) {
      groups.push({ kind: "break", token, index });
      return;
    }
    const last = groups[groups.length - 1];
    if (!leadingSpace && last?.kind === "word") {
      last.tokens.push({ token, index });
    } else {
      groups.push({ kind: "word", tokens: [{ token, index }] });
    }
  });
  return groups;
}

export default function TokenHeatmap({ tokens }: { tokens: TokenAttribution[] }) {
  const maxAbs = Math.max(...tokens.map((t) => Math.abs(t.attribution)), 1e-6);
  const groups = groupTokens(tokens);

  return (
    <p className="flex flex-wrap items-baseline gap-x-1 gap-y-2 leading-8">
      {groups.flatMap((group) => {
        if (group.kind === "break") {
          const { token, index } = group;
          return [
            <span key={`${index}-line`} data-testid="newline-break" className="basis-full h-0" aria-hidden="true" />,
            <span key={index} className="text-xs text-slate-400" title={`attribution: ${token.attribution.toFixed(3)}`}>
              ⏎
            </span>,
          ];
        }
        return [
          <span key={group.tokens[0].index} data-testid="word-group" className="inline-flex">
            {group.tokens.map(({ token, index }) => (
              <span
                key={index}
                className="rounded px-0.5"
                style={{ backgroundColor: tint(token.attribution, maxAbs) }}
                title={`attribution: ${token.attribution.toFixed(3)}`}
              >
                {displayText(token.token)}
              </span>
            ))}
          </span>,
        ];
      })}
    </p>
  );
}
