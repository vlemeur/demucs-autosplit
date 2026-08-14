from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd

RUNNING = True
StatePayload = dict[str, Any]


def _handle_signal(_signum, _frame) -> None:
    global RUNNING
    RUNNING = False


def _write_state(state_file: Path, payload: StatePayload) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = state_file.with_suffix(".tmp")
    tmp_file.write_text(json.dumps(payload), encoding="utf-8")
    tmp_file.replace(state_file)


def _resolve_samplerate(
    input_device_index: int,
    input_channels: int,
    preferred_sample_rate: float,
) -> float:
    candidates = [48000.0, 44100.0, preferred_sample_rate]
    checked: set[int] = set()
    for candidate in candidates:
        rounded = int(round(candidate))
        if rounded in checked:
            continue
        checked.add(rounded)
        try:
            sd.check_input_settings(
                device=input_device_index,
                channels=input_channels,
                samplerate=rounded,
            )
            return float(rounded)
        except Exception:
            continue
    return float(int(round(preferred_sample_rate)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Dedicated audio monitor daemon.")
    parser.add_argument("--input-device-index", type=int, required=True)
    parser.add_argument("--input-channels", required=True)
    parser.add_argument("--input-total-channels", type=int, required=True)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--preferred-sample-rate", type=float, required=True)
    parser.add_argument("--blocksize", type=int, default=256)
    args = parser.parse_args()

    state_file = Path(args.state_file)
    pid_file = Path(args.pid_file)
    selected_input_channels = tuple(
        int(value) for value in args.input_channels.split(",") if value.strip()
    )
    sample_rate = _resolve_samplerate(
        input_device_index=args.input_device_index,
        input_channels=args.input_total_channels,
        preferred_sample_rate=args.preferred_sample_rate,
    )

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()), encoding="utf-8")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    latest_state: StatePayload = {
        "running": True,
        "error": None,
        "rms": 0.0,
        "peak": 0.0,
        "per_channel_peaks": [],
        "sample_rate": sample_rate,
        "updated_at": time.time(),
        "input_channels": [channel + 1 for channel in selected_input_channels],
        "blocksize": int(args.blocksize),
        "input_device_index": int(args.input_device_index),
        "pid": os.getpid(),
    }

    _write_state(state_file, latest_state)

    def callback(indata, frames, _time_info, status) -> None:
        nonlocal latest_state
        if status:
            latest_state["error"] = str(status)
        if frames <= 0:
            return

        samples = np.asarray(indata, dtype=np.float32)
        if samples.size == 0:
            return

        per_channel_peaks = [
            float(np.max(np.abs(samples[:, channel_index])))
            for channel_index in range(samples.shape[1])
        ]
        selected = samples[:, list(selected_input_channels)]
        rms = float(np.sqrt(np.mean(np.square(selected))))
        peak = float(np.max(np.abs(selected)))

        latest_state = {
            "running": True,
            "error": latest_state.get("error"),
            "rms": rms,
            "peak": peak,
            "per_channel_peaks": per_channel_peaks,
            "sample_rate": sample_rate,
            "updated_at": time.time(),
            "input_channels": [channel + 1 for channel in selected_input_channels],
            "blocksize": int(args.blocksize),
            "input_device_index": int(args.input_device_index),
            "pid": os.getpid(),
        }

    try:
        with sd.InputStream(
            device=args.input_device_index,
            channels=args.input_total_channels,
            samplerate=sample_rate,
            dtype="float32",
            blocksize=args.blocksize,
            latency="low",
            callback=callback,
        ):
            while RUNNING:
                latest_state["updated_at"] = time.time()
                _write_state(state_file, latest_state)
                time.sleep(0.1)
    except Exception as exc:
        _write_state(
            state_file,
            {
                "running": False,
                "error": str(exc),
                "rms": 0.0,
                "peak": 0.0,
                "per_channel_peaks": [],
                "sample_rate": sample_rate,
                "updated_at": time.time(),
                "input_channels": [channel + 1 for channel in selected_input_channels],
                "blocksize": int(args.blocksize),
                "input_device_index": int(args.input_device_index),
                "pid": os.getpid(),
            },
        )
        return 1
    finally:
        _write_state(
            state_file,
            {
                "running": False,
                "error": latest_state.get("error"),
                "rms": latest_state.get("rms", 0.0),
                "peak": latest_state.get("peak", 0.0),
                "per_channel_peaks": latest_state.get("per_channel_peaks", []),
                "sample_rate": sample_rate,
                "updated_at": time.time(),
                "input_channels": [channel + 1 for channel in selected_input_channels],
                "blocksize": int(args.blocksize),
                "input_device_index": int(args.input_device_index),
                "pid": os.getpid(),
            },
        )
        pid_file.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
