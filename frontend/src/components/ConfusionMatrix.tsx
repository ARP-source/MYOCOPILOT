import { POSE_SET } from "../types";
import { useStore } from "../store/useStore";

const SHORT: Record<string, string> = {
  neutral: "neu",
  fist: "fist",
  index_point: "idx",
  pinch: "pin",
  tripod: "tri",
  wrist_flex: "wF",
  wrist_extend: "wE",
  pronation: "pro",
  supination: "sup",
};

function cellColor(value: number, max: number): string {
  if (max === 0 || value === 0) return "transparent";
  const t = value / max;
  const alpha = 0.12 + t * 0.75;
  return `rgba(53, 219, 232, ${alpha.toFixed(2)})`;
}

export function ConfusionMatrix() {
  const matrix = useStore((s) => s.confusionMatrix);
  const max = Math.max(1, ...matrix.flat());

  return (
    <div className="panel">
      <div className="panel__title">
        Confusion matrix <span className="panel__subtitle">actual (rows) vs. predicted (cols)</span>
      </div>
      <div className="cm-scroll">
        <table className="cm">
          <thead>
            <tr>
              <th className="cm__corner" />
              {POSE_SET.map((g) => (
                <th key={g} title={g} className="cm__head">
                  {SHORT[g]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {POSE_SET.map((rowGesture, r) => (
              <tr key={rowGesture}>
                <th title={rowGesture} className="cm__rowhead">
                  {SHORT[rowGesture]}
                </th>
                {POSE_SET.map((_, c) => {
                  const v = matrix[r]?.[c] ?? 0;
                  return (
                    <td
                      key={c}
                      className={r === c ? "cm__cell cm__cell--diag" : "cm__cell"}
                      style={{ background: cellColor(v, max) }}
                    >
                      {v || ""}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
