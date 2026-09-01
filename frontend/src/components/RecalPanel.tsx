import { useStore } from "../store/useStore";

export function RecalPanel() {
  const recalRequest = useStore((s) => s.recalRequest);
  const recalResult = useStore((s) => s.recalResult);
  const demoMode = useStore((s) => s.demoMode);
  const ackRecal = useStore((s) => s.ackRecal);

  if (!recalRequest && !recalResult) {
    return (
      <div className="panel recal recal--idle">
        <div className="panel__title">Recalibration</div>
        <div className="panel__empty">No drift detected. Recognition is stable.</div>
      </div>
    );
  }

  return (
    <div className="panel recal">
      <div className="panel__title">Recalibration</div>

      {recalRequest && (
        <div className="recal__chat">
          <div className="recal__badge">DRIFT DETECTED</div>
          <p className="recal__question">{recalRequest.question}</p>
          <div className="recal__meta">
            <span>
              target: <strong>{recalRequest.target_gesture}</strong>
            </span>
            <span>
              confused with: <strong>{recalRequest.confused_with}</strong>
            </span>
          </div>
          <button
            className="btn btn--primary"
            disabled={demoMode}
            onClick={() => ackRecal(recalRequest.target_gesture)}
          >
            {demoMode ? "Auto-acknowledging…" : "Send fresh samples"}
          </button>
        </div>
      )}

      {recalResult && (
        <div className="recal__result">
          <div className="recal__result-row">
            <span>{recalResult.gesture}</span>
            <span className="recal__acc-before">{(recalResult.acc_before * 100).toFixed(0)}%</span>
            <span className="recal__arrow">→</span>
            <span className="recal__acc-after">{(recalResult.acc_after * 100).toFixed(0)}%</span>
          </div>
        </div>
      )}
    </div>
  );
}
