import { Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useStore } from "../store/useStore";

const THRESHOLD = 0.6;

function formatSecondsAgo(t_ms: number, latest: number): string {
  const s = Math.max(0, Math.round((latest - t_ms) / 1000));
  return `-${s}s`;
}

export function AccuracyChart() {
  const series = useStore((s) => s.accuracySeries);

  if (series.length === 0) {
    return (
      <div className="panel">
        <div className="panel__title">Rolling accuracy (60s)</div>
        <div className="panel__empty">Waiting for classifications…</div>
      </div>
    );
  }

  const latest = series[series.length - 1].t_ms;

  return (
    <div className="panel">
      <div className="panel__title">Rolling accuracy (60s)</div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={series} margin={{ top: 8, right: 16, bottom: 0, left: -16 }}>
          <XAxis
            dataKey="t_ms"
            tickFormatter={(v) => formatSecondsAgo(v, latest)}
            stroke="#54575e"
            fontSize={10}
            tickLine={false}
          />
          <YAxis domain={[0, 1]} stroke="#54575e" fontSize={10} tickLine={false} width={32} />
          <Tooltip
            contentStyle={{ background: "#15161a", border: "1px solid #23252b", fontSize: 12 }}
            labelFormatter={(v) => formatSecondsAgo(Number(v), latest)}
            formatter={(v: number) => v.toFixed(2)}
          />
          <ReferenceLine y={THRESHOLD} stroke="#e8515a" strokeDasharray="6 4" strokeWidth={1.5} />
          <Line type="monotone" dataKey="acc" stroke="#35dbe8" strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
