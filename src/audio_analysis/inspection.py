from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from audio_analysis.chord_detection import (
    ChordBenchmarkResult,
    predict_chords_from_wave,
    run_chord_benchmark,
)


def extract_wav_clip_bytes(
    wav_path: Path, start_s: float, duration_s: float
) -> tuple[bytes, float]:
    """Extract a lightweight WAV preview clip for playback."""
    if not wav_path.exists():
        raise FileNotFoundError(f"File not found: {wav_path}")
    if start_s < 0.0:
        raise ValueError("start_s must be >= 0")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be > 0")

    try:
        audio, sample_rate = sf.read(wav_path, always_2d=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to read audio: {wav_path}") from exc

    n_samples = int(audio.shape[0])
    sr = float(sample_rate)
    start_idx = int(round(start_s * sr))
    if start_idx >= n_samples:
        return b"", 0.0

    end_idx = min(n_samples, start_idx + int(round(duration_s * sr)))
    clip = audio[start_idx:end_idx]
    if clip.size == 0:
        return b"", 0.0

    clip = clip.mean(axis=1, keepdims=True).astype(np.float32, copy=False)
    target_sample_rate = 22_050
    if int(sample_rate) != target_sample_rate:
        clip = resample_poly(clip, up=target_sample_rate, down=int(sample_rate), axis=0)
        sample_rate = target_sample_rate
        sr = float(sample_rate)

    buffer = io.BytesIO()
    try:
        sf.write(buffer, clip, int(sr), format="WAV")
    except Exception as exc:
        raise RuntimeError("Failed to write WAV clip") from exc

    clip_duration = float(clip.shape[0]) / sr
    return buffer.getvalue(), clip_duration


@dataclass(frozen=True)
class ChordSegment:
    """One chord span parsed from a lab file."""

    start_s: float
    end_s: float
    label: str


def read_chords_lab(lab_path: Path) -> list[ChordSegment]:
    """Read a chord lab file with `<start> <end> <label>` rows."""
    if not lab_path.exists():
        raise FileNotFoundError(f"File not found: {lab_path}")

    segments: list[ChordSegment] = []
    for raw_line in lab_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Invalid .lab line: {raw_line}")

        start_s = float(parts[0])
        end_s = float(parts[1])
        label = parts[2].strip()
        if end_s < start_s:
            raise ValueError(f"Invalid segment (end < start): {raw_line}")

        segments.append(ChordSegment(start_s=start_s, end_s=end_s, label=label))

    return segments


def load_waveform_for_plot(
    wav_path: Path,
    max_points: int = 200_000,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load a mono waveform preview for plotting."""
    if not wav_path.exists():
        raise FileNotFoundError(f"File not found: {wav_path}")
    if max_points <= 0:
        raise ValueError("max_points must be a positive integer")

    try:
        audio, sample_rate = sf.read(wav_path, always_2d=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to read audio: {wav_path}") from exc

    if audio.size == 0:
        raise RuntimeError(f"Empty audio file: {wav_path}")

    mono = audio.mean(axis=1).astype(np.float32)
    n_samples = int(mono.shape[0])
    sr = float(sample_rate)
    duration_s = float(n_samples) / sr

    if n_samples > max_points:
        stride = int(np.ceil(n_samples / float(max_points)))
        mono = mono[::stride]
        times_s = (np.arange(mono.shape[0], dtype=np.float32) * float(stride)) / sr
    else:
        times_s = np.arange(n_samples, dtype=np.float32) / sr

    return times_s, mono, duration_s


def predict_chords_for_stem(
    input_wav: Path,
    output_lab: Path,
    backend_id: str = "madmom",
) -> Path:
    """Run one chord-detection backend on a stem file."""
    if not input_wav.exists():
        raise FileNotFoundError(f"Stem wav not found: {input_wav}")

    output_lab.parent.mkdir(parents=True, exist_ok=True)
    bass_wav = input_wav.parent / "bass.wav"
    if bass_wav.resolve() == input_wav.resolve() or not bass_wav.exists():
        bass_wav = None
    predict_chords_from_wave(input_wav, output_lab, method=backend_id, bass_wav=bass_wav)
    return output_lab


def benchmark_chord_detection(
    input_wav: Path,
    output_dir: Path,
    backend_ids: list[str],
) -> list[ChordBenchmarkResult]:
    """Benchmark several chord backends on one input file."""
    if not input_wav.exists():
        raise FileNotFoundError(f"Stem wav not found: {input_wav}")
    return run_chord_benchmark(
        input_wav=input_wav,
        output_dir=output_dir,
        backend_ids=backend_ids,
    )


def compute_chord_label_agreement(
    first_segments: list[ChordSegment],
    second_segments: list[ChordSegment],
    *,
    step_s: float = 0.1,
) -> float | None:
    """Compute a simple exact-label agreement over time."""
    if not first_segments or not second_segments:
        return None
    if step_s <= 0:
        raise ValueError("step_s must be > 0")

    max_time = min(first_segments[-1].end_s, second_segments[-1].end_s)
    if max_time <= 0:
        return None

    matches = 0
    total = 0
    time_s = 0.0
    first_index = 0
    second_index = 0

    while time_s < max_time:
        while first_index + 1 < len(first_segments) and first_segments[first_index].end_s <= time_s:
            first_index += 1
        while (
            second_index + 1 < len(second_segments)
            and second_segments[second_index].end_s <= time_s
        ):
            second_index += 1

        if first_segments[first_index].label == second_segments[second_index].label:
            matches += 1
        total += 1
        time_s += step_s

    if total == 0:
        return None
    return matches / total
