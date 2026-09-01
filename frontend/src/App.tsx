import { useEffect } from "react";
import { HandScene } from "./components/HandScene";
import { AccuracyChart } from "./components/AccuracyChart";
import { ConfusionMatrix } from "./components/ConfusionMatrix";
import { Ticker } from "./components/Ticker";
import { ControlBar } from "./components/ControlBar";
import { RecalPanel } from "./components/RecalPanel";
import { useStore } from "./store/useStore";

export default function App() {
  const connect = useStore((s) => s.connect);
  const disconnect = useStore((s) => s.disconnect);
  const latestPred = useStore((s) => s.latestPred);

  useEffect(() => {
    connect();
    return () => disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const gesture = latestPred?.gesture ?? "neutral";
  const confidence = latestPred?.confidence ?? 0;
  const activation = latestPred ? Math.min(1, latestPred.confidence + 0.15) : 0;

  return (
    <div className="app">
      <header className="app__header">
        <div className="app__brand">
          MYO<span>COPILOT</span>
        </div>
        <div className="app__tagline">Conversational myoelectric control</div>
      </header>

      <main className="app__body">
        <section className="app__left">
          <HandScene gesture={gesture} confidence={confidence} activation={activation} />
          <div className="hand-readout">
            <div className="hand-readout__gesture">{gesture.replace(/_/g, " ")}</div>
            <div className="hand-readout__bar-row">
              <span className="hand-readout__label">confidence</span>
              <div className="hand-readout__bar">
                <div className="hand-readout__bar-fill" style={{ width: `${confidence * 100}%` }} />
              </div>
              <span className="hand-readout__value">{confidence.toFixed(2)}</span>
            </div>
            <div className="hand-readout__bar-row">
              <span className="hand-readout__label">activation</span>
              <div className="hand-readout__bar hand-readout__bar--activation">
                <div className="hand-readout__bar-fill" style={{ width: `${activation * 100}%` }} />
              </div>
              <span className="hand-readout__value">{activation.toFixed(2)}</span>
            </div>
          </div>
          <RecalPanel />
        </section>

        <section className="app__right">
          <AccuracyChart />
          <ConfusionMatrix />
          <Ticker />
        </section>
      </main>

      <footer className="app__footer">
        <ControlBar />
      </footer>
    </div>
  );
}
