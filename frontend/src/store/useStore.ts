import { create } from "zustand";
import {
  POSE_SET,
  type ClassificationMsg,
  type DriftMode,
  type HistoryEntry,
  type RecalibrationRequestMsg,
  type RecalibrationResultMsg,
  type ServerMsg,
  type StatsResponse,
  type TickerEntry,
} from "../types";

const RECONNECT_BASE_MS = 500;
const RECONNECT_MAX_MS = 8000;
const ACCURACY_WINDOW_MS = 60_000;
const TICKER_SIZE = 12;

function emptyConfusionMatrix(): number[][] {
  return POSE_SET.map(() => POSE_SET.map(() => 0));
}

interface AccuracyPoint {
  t_ms: number;
  acc: number;
}

interface StoreState {
  subject: string;
  connected: boolean;
  playing: boolean;
  demoMode: boolean;
  demoPhase: string | null;

  latestPred: ClassificationMsg["pred"] | null;
  latestTrueGesture: string | null;
  rollingAccuracy: number;
  cumulativeCost: number;
  totalTokens: number;

  accuracySeries: AccuracyPoint[];
  confusionMatrix: number[][];
  ticker: TickerEntry[];

  recalRequest: RecalibrationRequestMsg | null;
  recalResult: RecalibrationResultMsg | null;

  connect: () => void;
  disconnect: () => void;
  setSubject: (subject: string) => void;
  sendControl: (action: "play" | "pause") => void;
  injectDrift: (drift: DriftMode) => void;
  ackRecal: (gesture: string) => void;
  setDemoMode: (on: boolean) => void;
  hydrate: () => Promise<void>;
}

let socket: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let reconnectDelay = RECONNECT_BASE_MS;
let demoTimers: ReturnType<typeof setTimeout>[] = [];

function wsUrl(subject: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/stream?subject=${encodeURIComponent(subject)}`;
}

function clearDemoTimers() {
  demoTimers.forEach(clearTimeout);
  demoTimers = [];
}

export const useStore = create<StoreState>((set, get) => ({
  subject: "DB2_s1",
  connected: false,
  playing: true,
  demoMode: false,
  demoPhase: null,

  latestPred: null,
  latestTrueGesture: null,
  rollingAccuracy: 0,
  cumulativeCost: 0,
  totalTokens: 0,

  accuracySeries: [],
  confusionMatrix: emptyConfusionMatrix(),
  ticker: [],

  recalRequest: null,
  recalResult: null,

  connect: () => {
    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
      return;
    }
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }

    const { subject } = get();
    const ws = new WebSocket(wsUrl(subject));
    socket = ws;

    ws.onopen = () => {
      reconnectDelay = RECONNECT_BASE_MS;
      set({ connected: true });
      void get().hydrate();
    };

    ws.onmessage = (evt) => {
      let msg: ServerMsg;
      try {
        msg = JSON.parse(evt.data);
      } catch {
        return;
      }
      handleServerMsg(msg, set, get);
    };

    ws.onclose = () => {
      set({ connected: false });
      socket = null;
      reconnectTimer = setTimeout(() => {
        reconnectDelay = Math.min(reconnectDelay * 2, RECONNECT_MAX_MS);
        get().connect();
      }, reconnectDelay);
    };

    ws.onerror = () => {
      ws.close();
    };
  },

  disconnect: () => {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    clearDemoTimers();
    socket?.close();
    socket = null;
    set({ connected: false });
  },

  setSubject: (subject: string) => {
    set({
      subject,
      accuracySeries: [],
      confusionMatrix: emptyConfusionMatrix(),
      ticker: [],
      latestPred: null,
      latestTrueGesture: null,
      recalRequest: null,
      recalResult: null,
    });
    get().disconnect();
    get().connect();
  },

  sendControl: (action) => {
    set({ playing: action === "play" });
    socket?.send(JSON.stringify({ type: "control", action }));
  },

  injectDrift: (drift: DriftMode) => {
    socket?.send(JSON.stringify({ type: "control", action: "inject", drift }));
  },

  ackRecal: (gesture: string) => {
    socket?.send(JSON.stringify({ type: "recal_ack", gesture }));
  },

  setDemoMode: (on: boolean) => {
    clearDemoTimers();
    set({ demoMode: on, demoPhase: on ? "Streaming clean data…" : null });
    if (!on) return;

    const { connected, connect, sendControl, injectDrift } = get();
    if (!connected) connect();
    sendControl("play");

    demoTimers.push(
      setTimeout(() => {
        set({ demoPhase: "Injecting electrode shift…" });
        injectDrift("electrode_shift");
      }, 15_000),
    );
  },

  hydrate: async () => {
    try {
      const [historyRes, statsRes] = await Promise.all([fetch("/history"), fetch("/stats")]);
      const history: HistoryEntry[] = await historyRes.json();
      const stats: StatsResponse = await statsRes.json();

      const ticker = history.slice(-TICKER_SIZE).map(toTickerEntry);
      const last = history[history.length - 1];

      set({
        confusionMatrix: stats.confusion_matrix,
        accuracySeries: stats.accuracy_series,
        totalTokens: stats.total_tokens,
        cumulativeCost: stats.cost_usd,
        ticker,
        latestPred: last?.pred ?? null,
        latestTrueGesture: last?.true_gesture ?? null,
        rollingAccuracy: last?.rolling_accuracy_20 ?? 0,
      });
    } catch {
      // Backend may not be up yet; the live WS stream will fill state in.
    }
  },
}));

function toTickerEntry(h: HistoryEntry): TickerEntry {
  return {
    key: `${h.t_ms}`,
    t_ms: h.t_ms,
    true_gesture: h.true_gesture,
    pred_gesture: h.pred.gesture,
    confidence: h.pred.confidence,
    correct: h.true_gesture !== null && h.true_gesture === h.pred.gesture,
  };
}

function handleServerMsg(
  msg: ServerMsg,
  set: (partial: Partial<StoreState> | ((s: StoreState) => Partial<StoreState>)) => void,
  get: () => StoreState,
) {
  if (msg.type === "classification") {
    set((s) => {
      const nextSeries = [...s.accuracySeries, { t_ms: msg.t_ms, acc: msg.rolling_accuracy_20 }].filter(
        (p) => p.t_ms >= msg.t_ms - ACCURACY_WINDOW_MS,
      );

      let confusionMatrix = s.confusionMatrix;
      const trueIdx = msg.true_gesture ? POSE_SET.indexOf(msg.true_gesture as (typeof POSE_SET)[number]) : -1;
      const predIdx = POSE_SET.indexOf(msg.pred.gesture as (typeof POSE_SET)[number]);
      if (trueIdx >= 0 && predIdx >= 0) {
        confusionMatrix = s.confusionMatrix.map((row) => [...row]);
        confusionMatrix[trueIdx][predIdx] += 1;
      }

      const entry = toTickerEntry({
        t_ms: msg.t_ms,
        true_gesture: msg.true_gesture,
        pred: msg.pred,
        rolling_accuracy_20: msg.rolling_accuracy_20,
        cumulative_cost_usd: msg.cumulative_cost_usd,
      });
      const ticker = [...s.ticker, entry].slice(-TICKER_SIZE);

      return {
        latestPred: msg.pred,
        latestTrueGesture: msg.true_gesture,
        rollingAccuracy: msg.rolling_accuracy_20,
        cumulativeCost: msg.cumulative_cost_usd,
        accuracySeries: nextSeries,
        confusionMatrix,
        ticker,
      };
    });
  } else if (msg.type === "recalibration_request") {
    set({
      recalRequest: msg,
      recalResult: null,
      demoPhase: get().demoMode ? "Accuracy dropped, recalibration requested" : null,
    });
    if (get().demoMode) {
      const t = setTimeout(() => {
        get().ackRecal(msg.target_gesture);
        set({ demoPhase: "Auto-acknowledging fresh samples…" });
      }, 1800);
      demoTimers.push(t);
    }
  } else if (msg.type === "recalibration_result") {
    set({
      recalResult: msg,
      recalRequest: null,
      demoPhase: get().demoMode ? "Accuracy recovered, stream resuming" : null,
    });
  }
}
