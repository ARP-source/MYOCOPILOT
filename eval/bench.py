"""
eval.bench, offline accuracy and latency benchmark.

Replays a full subject recording through the classifier without real-time
throttling and reports per-gesture accuracy, mean latency, and total API cost.

Usage
-----
    python -m eval.bench --subject DB2_s1 [--drift electrode_shift]
"""

from __future__ import annotations
import argparse
from myosignal import stream, WindowEnvelope
from agent.types import ClassifierResponse


async def run_benchmark(
    subject: str,
    drift_at_s: float | None = None,
    drift_mode: str = "electrode_shift",
) -> dict:
    """
    Run the full offline benchmark for *subject*.

    Parameters
    ----------
    subject:
        Dataset subject identifier.
    drift_at_s:
        Optional time (s) at which to inject drift.
    drift_mode:
        Type of drift to inject (see myosignal.drift.DriftMode).

    Returns
    -------
    {
        "accuracy":        float,
        "per_gesture_acc": dict[str, float],
        "mean_latency_ms": float,
        "total_tokens":    int,
        "cost_usd":        float,
    }
    """
    raise NotImplementedError


def main() -> None:
    """CLI entry point: parse args and run the benchmark."""
    parser = argparse.ArgumentParser(description="MyoCopilot offline benchmark")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--drift", choices=["electrode_shift", "sweat", "channel_swap"])
    parser.add_argument("--drift-at-s", type=float, default=None)
    raise NotImplementedError


if __name__ == "__main__":
    main()
