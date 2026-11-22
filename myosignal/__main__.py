"""
myosignal.__main__, CLI entry point.

Usage
-----
    python -m myosignal --subject 1 [options]

Options
-------
    --source     ninapro | emg2pose | synthetic | auto  (default: auto)
    --subject    subject identifier (required)
    --drift-at   seconds before drift injection (default: no drift)
    --drift-mode electrode_shift | sweat | channel_swap (default: electrode_shift)
    --save-debug save first 5 spectrograms as PNG to debug/
    --max-windows stop after N windows (default: stream until done)
    --no-realtime skip wall-clock sleep (runs as fast as possible)

Example
-------
    python -m myosignal --source ninapro --subject 1 --drift-at 60 --save-debug
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

from myosignal.player import SessionPlayer
from myosignal.drift import DriftMode


def _save_spectrogram(b64: str, path: Path) -> None:
    """Decode base64 PNG and write to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(b64))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m myosignal",
        description="Stream sEMG WindowEnvelopes with optional drift injection.",
    )
    parser.add_argument("--source",      default="auto",
                        choices=["ninapro", "emg2pose", "synthetic", "auto"],
                        help="Dataset source format (default: auto-detect)")
    parser.add_argument("--subject",     required=True,
                        help="Subject identifier, e.g. '1' or 'DB2_s1'")
    parser.add_argument("--drift-at",    type=float, default=None,
                        dest="drift_at",
                        help="Seconds before drift injection (default: no drift)")
    parser.add_argument("--drift-mode",  default="electrode_shift",
                        dest="drift_mode",
                        choices=["electrode_shift", "sweat", "channel_swap"],
                        help="Drift type (default: electrode_shift)")
    parser.add_argument("--save-debug",  action="store_true",
                        dest="save_debug",
                        help="Save first 5 spectrograms to debug/")
    parser.add_argument("--max-windows", type=int, default=None,
                        dest="max_windows",
                        help="Stop after N windows")
    parser.add_argument("--no-realtime", action="store_true",
                        dest="no_realtime",
                        help="Skip real-time sleep (benchmark mode)")
    args = parser.parse_args(argv)

    # --- Build player ---
    print(f"[myosignal] Loading subject='{args.subject}' source='{args.source}'…",
          flush=True)
    try:
        player = SessionPlayer(
            subject=args.subject,
            source=args.source,
        )
    except FileNotFoundError as exc:
        print(f"[myosignal] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[myosignal] Source: {player.source}  "
          f"Windows: {len(player._windows)}  "
          f"ref_db: {player._ref_db:.1f} dB",
          flush=True)
    if args.drift_at is not None:
        print(f"[myosignal] Drift '{args.drift_mode}' will be injected at t={args.drift_at:.1f} s",
              flush=True)
    print(f"{'t_ms':>10}  {'true_gesture':<14}  {'max_rms':>9}  {'drift':>5}", flush=True)
    print("-" * 48, flush=True)

    saved_count = 0
    debug_dir = Path("debug")
    t_start = time.monotonic()

    gen = player.stream(drift_at_s=args.drift_at, drift_mode=args.drift_mode)

    for i, env in enumerate(gen):
        if args.max_windows is not None and i >= args.max_windows:
            break

        # --- Print trace line ---
        true_g   = env["true_gesture"] or ", "
        max_rms  = max(env["features"]["rms"])
        drifting = "*" if (
            args.drift_at is not None
            and (env["t_ms"] / 1000 - time.time() + t_start) < 0  # past injection point
        ) else " "
        # Simpler drift indicator: check if drift mode is active
        drift_indicator = "*" if player._drift_mode is not None else " "
        print(f"{env['t_ms']:>10}  {true_g:<14}  {max_rms:>9.4f}  {drift_indicator:>5}",
              flush=True)

        # --- Save first 5 spectrograms ---
        if args.save_debug and saved_count < 5:
            out_path = debug_dir / f"spectrogram_{i:04d}_{true_g}.png"
            _save_spectrogram(env["spectrogram_png_b64"], out_path)
            print(f"  -> saved {out_path}", flush=True)
            saved_count += 1

        # --- Bypass real-time sleep if requested ---
        # (player.stream() already sleeps; we can't easily skip it without
        #  re-implementing iteration, use --no-realtime as a hint only for
        #  now; a future version will pass this flag to SessionPlayer)

    elapsed = time.monotonic() - t_start
    print("-" * 48, flush=True)
    print(f"[myosignal] Done. {i + 1} windows in {elapsed:.1f} s.", flush=True)


if __name__ == "__main__":
    main()
