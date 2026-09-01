// Mirrors the contracts declared VERBATIM in CLAUDE.md ("Contracts" section).
// This file must stay in lockstep with that document. Any contract change
// must go through CONTRACT_CHANGES.md, not a silent edit here.

export const POSE_SET = [
  "neutral",
  "fist",
  "index_point",
  "pinch",
  "tripod",
  "wrist_flex",
  "wrist_extend",
  "pronation",
  "supination",
] as const;

export type Gesture = (typeof POSE_SET)[number];

export interface ClassifierResponse {
  gesture: string;
  confidence: number;
  drift_suspected: boolean;
  notes: string;
}

export type DriftMode = "electrode_shift" | "sweat" | "channel_swap";

export interface ClassificationMsg {
  type: "classification";
  t_ms: number;
  true_gesture: string | null;
  pred: ClassifierResponse;
  rolling_accuracy_20: number;
  cumulative_cost_usd: number;
}

export interface RecalibrationRequestMsg {
  type: "recalibration_request";
  target_gesture: string;
  confused_with: string;
  question: string;
}

export interface RecalibrationResultMsg {
  type: "recalibration_result";
  gesture: string;
  acc_before: number;
  acc_after: number;
}

export type ServerMsg =
  | ClassificationMsg
  | RecalibrationRequestMsg
  | RecalibrationResultMsg;

export interface ControlMsg {
  type: "control";
  action: "play" | "pause" | "inject";
  drift?: DriftMode;
}

export interface RecalAckMsg {
  type: "recal_ack";
  gesture: string;
}

export type ClientMsg = ControlMsg | RecalAckMsg;

export interface HistoryEntry {
  t_ms: number;
  true_gesture: string | null;
  pred: ClassifierResponse;
  rolling_accuracy_20: number;
  cumulative_cost_usd: number;
}

export interface StatsResponse {
  confusion_matrix: number[][];
  accuracy_series: { t_ms: number; acc: number }[];
  total_tokens: number;
  cost_usd: number;
}

export interface TickerEntry {
  key: string;
  t_ms: number;
  true_gesture: string | null;
  pred_gesture: string;
  confidence: number;
  correct: boolean;
}
