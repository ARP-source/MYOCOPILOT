import { useState } from "react";
import { useStore } from "../store/useStore";
import type { DriftMode } from "../types";

const DRIFT_BUTTONS: { mode: DriftMode; label: string }[] = [
  { mode: "electrode_shift", label: "Electrode shift" },
  { mode: "sweat", label: "Sweat" },
  { mode: "channel_swap", label: "Channel swap" },
];

export function ControlBar() {
  const subject = useStore((s) => s.subject);
  const playing = useStore((s) => s.playing);
  const connected = useStore((s) => s.connected);
  const demoMode = useStore((s) => s.demoMode);
  const demoPhase = useStore((s) => s.demoPhase);
  const cumulativeCost = useStore((s) => s.cumulativeCost);
  const setSubject = useStore((s) => s.setSubject);
  const sendControl = useStore((s) => s.sendControl);
  const injectDrift = useStore((s) => s.injectDrift);
  const setDemoMode = useStore((s) => s.setDemoMode);

  const [subjectInput, setSubjectInput] = useState(subject);

  function applySubject() {
    if (subjectInput.trim() && subjectInput !== subject) {
      setSubject(subjectInput.trim());
    }
  }

  return (
    <div className="control-bar">
      <div className="control-bar__group">
        <span className={connected ? "status-dot status-dot--on" : "status-dot"} title={connected ? "connected" : "disconnected"} />
        <input
          className="subject-input"
          value={subjectInput}
          onChange={(e) => setSubjectInput(e.target.value)}
          onBlur={applySubject}
          onKeyDown={(e) => e.key === "Enter" && applySubject()}
          spellCheck={false}
        />
      </div>

      <div className="control-bar__group">
        <button className="btn" onClick={() => sendControl(playing ? "pause" : "play")}>
          {playing ? "Pause" : "Play"}
        </button>
      </div>

      <div className="control-bar__group control-bar__group--drift">
        {DRIFT_BUTTONS.map((d) => (
          <button key={d.mode} className="btn btn--ghost" onClick={() => injectDrift(d.mode)}>
            {d.label}
          </button>
        ))}
      </div>

      <div className="control-bar__group control-bar__group--right">
        {demoMode && demoPhase && <span className="demo-phase">{demoPhase}</span>}
        <label className="demo-toggle">
          <input type="checkbox" checked={demoMode} onChange={(e) => setDemoMode(e.target.checked)} />
          DEMO MODE
        </label>
        <span className="cost">${cumulativeCost.toFixed(4)}</span>
      </div>
    </div>
  );
}
