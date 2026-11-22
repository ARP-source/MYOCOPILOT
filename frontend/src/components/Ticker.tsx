import { useStore } from "../store/useStore";

export function Ticker() {
  const ticker = useStore((s) => s.ticker);
  const items = [...ticker].reverse();

  return (
    <div className="panel">
      <div className="panel__title">Latest classifications</div>
      {items.length === 0 ? (
        <div className="panel__empty">No classifications yet.</div>
      ) : (
        <ul className="ticker">
          {items.map((entry) => (
            <li key={entry.key} className={entry.correct ? "ticker__row ticker__row--ok" : "ticker__row ticker__row--bad"}>
              <span className="ticker__indicator">{entry.correct ? "✓" : "✗"}</span>
              <span className="ticker__true">{entry.true_gesture ?? ", "}</span>
              <span className="ticker__arrow">→</span>
              <span className="ticker__pred">{entry.pred_gesture}</span>
              <span className="ticker__conf">{(entry.confidence * 100).toFixed(0)}%</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
