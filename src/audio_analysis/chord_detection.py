from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import librosa
import numpy as np
import soundfile as sf
from madmom.audio.chroma import DeepChromaProcessor
from madmom.audio.signal import SignalProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

DEFAULT_CHORD_BACKEND = "madmom"
CHORDMINI_DIR_ENV = "CHORDMINI_DIR"
CHORDMINI_PYTHON_ENV = "CHORDMINI_PYTHON"
CHORDMINI_CONFIG_ENV = "CHORDMINI_CONFIG"
CHORDMINI_BTC_CHECKPOINT_ENV = "CHORDMINI_BTC_CHECKPOINT"
CHORDMINI_CHORDNET_CHECKPOINT_ENV = "CHORDMINI_CHORDNET_CHECKPOINT"
CHORDMINI_AUTO_DOWNLOAD_ENV = "CHORDMINI_AUTO_DOWNLOAD"
CHORDMINI_CACHE_DIR = Path(".cache") / "chordmini"
CHORDMINI_REPO_ZIP_URL = "https://codeload.github.com/ptnghia-j/ChordMini/zip/refs/heads/main"
PITCH_CLASS_TO_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH_CLASS_TO_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
NOTE_NAME_TO_PITCH_CLASS = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}


@dataclass(frozen=True)
class ChordDetectionBackend:
    """Metadata describing an available chord-detection backend."""

    backend_id: str
    label: str
    description: str
    supports_local_inference: bool


@dataclass(frozen=True)
class ChordBenchmarkResult:
    """Summary of one backend run during a benchmark."""

    backend_id: str
    label: str
    output_lab: Path | None
    runtime_s: float
    segment_count: int
    unique_label_count: int
    success: bool
    error: str | None = None


BACKENDS: tuple[ChordDetectionBackend, ...] = (
    ChordDetectionBackend(
        backend_id="madmom",
        label="Madmom DeepChroma",
        description="Legacy deep-chroma + CRF baseline with a mostly maj/min vocabulary.",
        supports_local_inference=True,
    ),
    ChordDetectionBackend(
        backend_id="chordmini_btc",
        label="ChordMini BTC",
        description="Transformer BTC backend from ChordMini, configured through a local checkout.",
        supports_local_inference=False,
    ),
    ChordDetectionBackend(
        backend_id="chordmini_chordnet",
        label="ChordMini ChordNet",
        description="ChordMini ChordNet / 2E1D backend, configured through a local checkout.",
        supports_local_inference=False,
    ),
)


def list_chord_detection_backends() -> list[ChordDetectionBackend]:
    """Return the supported chord-detection backends."""
    return list(BACKENDS)


def get_chord_detection_backend(backend_id: str) -> ChordDetectionBackend:
    """Look up a backend definition by id."""
    for backend in BACKENDS:
        if backend.backend_id == backend_id:
            return backend
    raise ValueError(f"Unsupported chord backend: {backend_id}")


def _default_chordmini_dir() -> Path:
    """Return the default local cache path for the auto-downloaded ChordMini checkout."""
    return CHORDMINI_CACHE_DIR / "ChordMini-main"


def _should_auto_download_chordmini() -> bool:
    """Return whether auto-download is enabled for ChordMini."""
    raw_value = os.environ.get(CHORDMINI_AUTO_DOWNLOAD_ENV, "1").strip().lower()
    return raw_value not in {"0", "false", "no", "off"}


def _download_chordmini_repo(target_root: Path) -> Path:
    """Download and extract the ChordMini repository into the local cache."""
    target_root.mkdir(parents=True, exist_ok=True)
    archive_path = target_root / "ChordMini-main.zip"
    extract_dir = target_root / "ChordMini-main"

    logger.info("⬇️ Downloading ChordMini into %s", extract_dir)
    with urlopen(CHORDMINI_REPO_ZIP_URL, timeout=120) as response:
        archive_path.write_bytes(response.read())

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target_root)

    archive_path.unlink(missing_ok=True)
    return extract_dir


def _ensure_chordmini_checkout() -> Path:
    """Resolve or bootstrap a local ChordMini checkout."""
    chordmini_dir_raw = os.environ.get(CHORDMINI_DIR_ENV)
    if chordmini_dir_raw:
        chordmini_dir = Path(chordmini_dir_raw).expanduser().resolve()
        if not chordmini_dir.exists():
            raise FileNotFoundError(f"ChordMini directory not found: {chordmini_dir}")
        return chordmini_dir

    chordmini_dir = _default_chordmini_dir().resolve()
    if chordmini_dir.exists():
        return chordmini_dir

    if not _should_auto_download_chordmini():
        raise RuntimeError(
            "ChordMini is not installed locally. Either set CHORDMINI_DIR or enable "
            "auto-download with CHORDMINI_AUTO_DOWNLOAD=1."
        )

    return _download_chordmini_repo(chordmini_dir.parent)


def _prepare_audio(input_path: Path, output_path: Path) -> None:
    """
    Prepare audio file for chord detection.

    Converts stereo to mono and ensures float32 format at 44100 Hz.
    """
    data, sr = sf.read(str(input_path))
    if data.ndim > 1:
        logger.info("🔄 Converting stereo to mono for: %s", input_path.name)
        data = np.mean(data, axis=1)

    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val * 0.99

    data = data.astype(np.float32)

    if sr != 44100:
        from scipy.signal import resample

        data = resample(data, int(len(data) * 44100 / sr))
        sr = 44100

    sf.write(str(output_path), data, sr, format="WAV")


def _write_standard_lab(chords, output_lab: Path) -> None:
    """Write a standard 3-column lab file: start, end, chord label."""
    with output_lab.open("w", encoding="utf-8") as handle:
        for start, end, label in chords:
            handle.write(f"{start:.3f}\t{end:.3f}\t{label}\n")


def _normalize_lab_file(input_lab: Path, output_lab: Path) -> None:
    """Normalize a backend lab file to the standard 3-column format."""
    normalized_lines: list[str] = []
    for raw_line in input_lab.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Invalid .lab line: {raw_line}")
        start_s = float(parts[0])
        end_s = float(parts[1])
        label = parts[2]
        normalized_lines.append(f"{start_s:.3f}\t{end_s:.3f}\t{label}")

    output_lab.write_text("\n".join(normalized_lines) + ("\n" if normalized_lines else ""), "utf-8")


def _parse_chord_label(label: str) -> tuple[str | None, str, str | None]:
    """Parse a chord label into root, quality, and optional slash bass."""
    normalized = label.strip()
    if normalized in {"", "N", "N.C.", "NC", "X"}:
        return None, "", None

    bass = None
    if "/" in normalized:
        normalized, bass = normalized.split("/", 1)
        bass = bass.strip() or None

    if ":" in normalized:
        root, quality = normalized.split(":", 1)
    else:
        root, quality = normalized, "maj"

    root = root.strip()
    quality = quality.strip() or "maj"
    if root not in NOTE_NAME_TO_PITCH_CLASS:
        return None, "", bass
    return root, quality, bass


def _quality_to_intervals(quality: str) -> tuple[int, ...]:
    """Return chord-tone intervals from the root for common chord qualities."""
    normalized = quality.lower().replace(":", "")
    if normalized in {"maj", ""}:
        return (0, 4, 7)
    if normalized in {"min", "m"}:
        return (0, 3, 7)
    if normalized in {"maj7", "ma7", "m7+"}:
        return (0, 4, 7, 11)
    if normalized in {"min7", "m7"}:
        return (0, 3, 7, 10)
    if normalized == "7":
        return (0, 4, 7, 10)
    if normalized in {"dim", "o"}:
        return (0, 3, 6)
    if normalized in {"dim7", "o7"}:
        return (0, 3, 6, 9)
    if normalized in {"hdim7", "m7b5"}:
        return (0, 3, 6, 10)
    if normalized in {"aug", "+"}:
        return (0, 4, 8)
    if normalized in {"6"}:
        return (0, 4, 7, 9)
    if normalized in {"min6", "m6"}:
        return (0, 3, 7, 9)
    if normalized.startswith("sus2"):
        return (0, 2, 7)
    if normalized.startswith("sus"):
        return (0, 5, 7)
    if "9" in normalized or "11" in normalized or "13" in normalized:
        return (0, 4, 7, 10)
    return (0, 4, 7)


def _pitch_class_to_note_name(pitch_class: int, *, prefer_flats: bool) -> str:
    note_names = PITCH_CLASS_TO_FLAT if prefer_flats else PITCH_CLASS_TO_SHARP
    return note_names[pitch_class % 12]


def _estimate_segment_bass_pitch_class(
    mono_audio: np.ndarray,
    sample_rate: int,
    start_s: float,
    end_s: float,
) -> tuple[int | None, float]:
    """Estimate the dominant bass pitch class over one chord segment."""
    start_idx = max(0, int(round(start_s * sample_rate)))
    end_idx = min(len(mono_audio), int(round(end_s * sample_rate)))
    clip = mono_audio[start_idx:end_idx]
    if clip.size < max(2048, sample_rate // 10):
        return None, 0.0

    rms = float(np.sqrt(np.mean(np.square(clip.astype(np.float32)))))
    if rms < 0.01:
        return None, 0.0

    f0, voiced_flag, _voiced_prob = librosa.pyin(
        clip.astype(np.float32),
        fmin=float(librosa.note_to_hz("E1")),
        fmax=float(librosa.note_to_hz("C5")),
        sr=sample_rate,
        frame_length=4096,
        hop_length=512,
    )
    if f0 is None or voiced_flag is None:
        return None, 0.0

    voiced_f0 = f0[np.asarray(voiced_flag, dtype=bool)]
    if voiced_f0.size == 0:
        return None, 0.0

    midi_notes = np.rint(librosa.hz_to_midi(voiced_f0)).astype(int)
    pitch_classes = np.mod(midi_notes, 12)
    counts = np.bincount(pitch_classes, minlength=12)
    best_pitch_class = int(np.argmax(counts))
    confidence = float(counts[best_pitch_class]) / float(max(1, voiced_f0.size))
    return best_pitch_class, confidence


def _estimate_segment_lowest_pitch_class(
    mono_audio: np.ndarray,
    sample_rate: int,
    start_s: float,
    end_s: float,
) -> tuple[int | None, float]:
    """Estimate the lowest prominent pitch class inside a polyphonic chord segment."""
    start_idx = max(0, int(round(start_s * sample_rate)))
    end_idx = min(len(mono_audio), int(round(end_s * sample_rate)))
    clip = mono_audio[start_idx:end_idx]
    if clip.size < max(4096, sample_rate // 5):
        return None, 0.0

    rms = float(np.sqrt(np.mean(np.square(clip.astype(np.float32)))))
    if rms < 0.01:
        return None, 0.0

    spectrum = np.abs(
        librosa.stft(
            clip.astype(np.float32),
            n_fft=8192,
            hop_length=512,
            win_length=4096,
            center=False,
        )
    )
    average_spectrum = spectrum.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=8192)
    mask = (freqs >= 180.0) & (freqs <= 1400.0)
    if not np.any(mask):
        return None, 0.0

    freqs = freqs[mask]
    average_spectrum = average_spectrum[mask]
    if average_spectrum.size == 0 or float(np.max(average_spectrum)) <= 0.0:
        return None, 0.0

    peak_indices, _ = find_peaks(average_spectrum)
    if peak_indices.size == 0:
        return None, 0.0

    peak_magnitudes = average_spectrum[peak_indices]
    magnitude_threshold = float(np.max(peak_magnitudes)) * 0.3
    prominent_peaks = [
        (float(freqs[idx]), float(average_spectrum[idx]))
        for idx in peak_indices
        if float(average_spectrum[idx]) >= magnitude_threshold
    ]
    if not prominent_peaks:
        return None, 0.0

    lowest_freq, magnitude = min(prominent_peaks, key=lambda item: item[0])
    midi_note = int(round(float(librosa.hz_to_midi(lowest_freq))))
    pitch_class = midi_note % 12
    confidence = magnitude / float(np.max(average_spectrum))
    return pitch_class, confidence


def _apply_bass_slash_chords(
    output_lab: Path, analysis_wav: Path, bass_wav: Path | None = None
) -> None:
    """Rewrite chord labels with slash notes using the selected stem and optional bass stem."""
    if not output_lab.exists() or not analysis_wav.exists():
        return

    try:
        analysis_audio, sample_rate = sf.read(str(analysis_wav), always_2d=True)
    except Exception as exc:
        logger.warning("⚠️ Failed to read analysis stem for slash-chord post-processing: %s", exc)
        return

    mono_analysis = analysis_audio.mean(axis=1).astype(np.float32)
    mono_bass: np.ndarray | None = None
    if bass_wav is not None and bass_wav.exists():
        try:
            bass_audio, _bass_sr = sf.read(str(bass_wav), always_2d=True)
            mono_bass = bass_audio.mean(axis=1).astype(np.float32)
        except Exception as exc:
            logger.warning("⚠️ Failed to read bass stem for slash-chord post-processing: %s", exc)
            mono_bass = None

    rewritten_lines: list[str] = []
    applied_changes = 0

    for raw_line in output_lab.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 3:
            rewritten_lines.append(line)
            continue

        start_s = float(parts[0])
        end_s = float(parts[1])
        label = parts[2]
        root, quality, existing_bass = _parse_chord_label(label)
        if root is None or existing_bass:
            rewritten_lines.append(f"{start_s:.3f}\t{end_s:.3f}\t{label}")
            continue

        bass_pitch_class, confidence = _estimate_segment_lowest_pitch_class(
            mono_audio=mono_analysis,
            sample_rate=int(sample_rate),
            start_s=start_s,
            end_s=end_s,
        )
        if (bass_pitch_class is None or confidence < 0.45) and mono_bass is not None:
            bass_pitch_class, confidence = _estimate_segment_bass_pitch_class(
                mono_audio=mono_bass,
                sample_rate=int(sample_rate),
                start_s=start_s,
                end_s=end_s,
            )

        if bass_pitch_class is None or confidence < 0.45:
            rewritten_lines.append(f"{start_s:.3f}\t{end_s:.3f}\t{label}")
            continue

        root_pitch_class = NOTE_NAME_TO_PITCH_CLASS[root]
        chord_tones = {
            (root_pitch_class + interval) % 12 for interval in _quality_to_intervals(quality)
        }
        if bass_pitch_class == root_pitch_class or bass_pitch_class not in chord_tones:
            rewritten_lines.append(f"{start_s:.3f}\t{end_s:.3f}\t{label}")
            continue

        prefer_flats = "b" in root or "b" in quality.lower()
        bass_note = _pitch_class_to_note_name(bass_pitch_class, prefer_flats=prefer_flats)
        rewritten_label = f"{root}:{quality}/{bass_note}"
        rewritten_lines.append(f"{start_s:.3f}\t{end_s:.3f}\t{rewritten_label}")
        applied_changes += 1

    if applied_changes:
        logger.info("🎹 Applied slash-chord bass post-processing to %s segments.", applied_changes)
        output_lab.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")


def _predict_with_madmom(input_wav: Path, output_lab: Path) -> None:
    """Run the local Madmom baseline and save a standard .lab file."""
    logger.info("🎵 Running chord recognition on: %s (backend=madmom)", input_wav.name)
    tmp_path = input_wav.with_name(f"{input_wav.stem}_madmom_tmp.wav")
    _prepare_audio(input_wav, tmp_path)

    try:
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        chroma = DeepChromaProcessor()
        decoder = DeepChromaChordRecognitionProcessor()

        audio = sig(str(tmp_path))
        chroma_features = chroma(audio)
        chords = decoder(chroma_features)
        _write_standard_lab(chords, output_lab)
    finally:
        tmp_path.unlink(missing_ok=True)


def _resolve_chordmini_paths(backend_id: str) -> tuple[Path, str, Path, Path]:
    """Resolve the local ChordMini checkout, python executable, config, and checkpoint."""
    chordmini_dir = _ensure_chordmini_checkout()

    python_candidate = os.environ.get(CHORDMINI_PYTHON_ENV)
    if python_candidate is None:
        python_candidate = sys.executable

    config_candidate = os.environ.get(CHORDMINI_CONFIG_ENV)
    config_path = (
        Path(config_candidate).expanduser().resolve()
        if config_candidate
        else chordmini_dir / "config" / "ChordMini.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"ChordMini config not found: {config_path}")

    checkpoint_env = (
        CHORDMINI_BTC_CHECKPOINT_ENV
        if backend_id == "chordmini_btc"
        else CHORDMINI_CHORDNET_CHECKPOINT_ENV
    )
    checkpoint_default = (
        chordmini_dir / "checkpoints" / "btc_model_best.pth"
        if backend_id == "chordmini_btc"
        else chordmini_dir / "checkpoints" / "2e1d_model_best.pth"
    )
    checkpoint_candidate = os.environ.get(checkpoint_env)
    checkpoint_path = (
        Path(checkpoint_candidate).expanduser().resolve()
        if checkpoint_candidate
        else checkpoint_default
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"ChordMini checkpoint not found: {checkpoint_path}")

    return chordmini_dir, python_candidate, config_path, checkpoint_path


def _predict_with_chordmini(backend_id: str, input_wav: Path, output_lab: Path) -> None:
    """Run a ChordMini backend through its official evaluation script."""
    chordmini_dir, python_cmd, config_path, checkpoint_path = _resolve_chordmini_paths(backend_id)
    input_wav = input_wav.expanduser().resolve()
    output_lab = output_lab.expanduser().resolve()
    chordmini_dir = chordmini_dir.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    script_path = (chordmini_dir / "src" / "evaluation" / "test.py").resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"ChordMini inference script not found: {script_path}")

    model_type = "BTC" if backend_id == "chordmini_btc" else "ChordNet"
    raw_save_dir = (output_lab.parent / f"{input_wav.stem}_{backend_id}_raw").resolve()
    raw_output_lab = (raw_save_dir / f"{input_wav.stem}.lab").resolve()
    raw_save_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = (chordmini_dir / ".mplconfig").resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_cmd,
        str(script_path),
        "--model_type",
        model_type,
        "--checkpoint",
        str(checkpoint_path),
        "--config",
        str(config_path),
        "--audio_dir",
        str(input_wav),
        "--save_dir",
        str(raw_save_dir),
        "--use_overlap",
        "--use_gaussian",
        "--kernel_size",
        "9",
        "--vote_aggregation",
        "logit",
        "--min_segment_duration",
        "0.5",
        "--smooth_predictions",
    ]

    if backend_id == "chordmini_btc":
        cmd.append("--smooth_logits")

    child_env = os.environ.copy()
    child_env["MPLCONFIGDIR"] = str(mpl_config_dir)

    completed = subprocess.run(
        cmd,
        cwd=str(chordmini_dir),
        check=True,
        capture_output=True,
        env=child_env,
        text=True,
        timeout=3600,
    )
    if completed.stdout.strip():
        logger.info("%s", completed.stdout.strip())
    if completed.stderr.strip():
        logger.info("%s", completed.stderr.strip())

    if not raw_output_lab.exists():
        raise RuntimeError(f"ChordMini did not produce the expected lab file: {raw_output_lab}")

    _normalize_lab_file(raw_output_lab, output_lab)


def predict_chords_from_wave(
    input_wav: Path,
    output_lab: Path,
    method: str = DEFAULT_CHORD_BACKEND,
    bass_wav: Path | None = None,
) -> None:
    """
    Predict chords from a WAV file and save to a standard .lab file.

    Parameters
    ----------
    input_wav : Path
        Path to the input WAV file (mono or stereo).
    output_lab : Path
        Path to the output .lab file (will be created/overwritten).
    method : str, default="madmom"
        Chord detection backend id.
    bass_wav : Path or None, default=None
        Optional bass stem WAV used for slash-chord post-processing.
    """
    get_chord_detection_backend(method)

    if not input_wav.exists():
        raise FileNotFoundError(f"Input file not found: {input_wav}")

    output_lab.parent.mkdir(parents=True, exist_ok=True)

    try:
        if method == "madmom":
            _predict_with_madmom(input_wav, output_lab)
        elif method in {"chordmini_btc", "chordmini_chordnet"}:
            _predict_with_chordmini(method, input_wav, output_lab)
        else:
            raise ValueError(f"Unsupported chord backend: {method}")

        _apply_bass_slash_chords(output_lab=output_lab, analysis_wav=input_wav, bass_wav=bass_wav)
        logger.info("✅ Chord predictions saved to: %s", output_lab)
    except subprocess.CalledProcessError as exc:
        logger.error(
            "❌ Chord backend %s failed for %s (returncode=%s).\nstdout: %s\nstderr: %s",
            method,
            input_wav.name,
            exc.returncode,
            (exc.stdout or "").strip(),
            (exc.stderr or "").strip(),
        )
        raise RuntimeError(f"Chord prediction failed with backend {method}: {exc}") from exc
    except Exception as exc:
        logger.error("❌ Failed to predict chords for %s with %s: %s", input_wav.name, method, exc)
        raise RuntimeError(f"Chord prediction failed with backend {method}: {exc}") from exc


def run_chord_benchmark(
    input_wav: Path,
    output_dir: Path,
    backend_ids: list[str],
) -> list[ChordBenchmarkResult]:
    """Run several backends on the same file and return lightweight benchmark summaries."""
    results: list[ChordBenchmarkResult] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for backend_id in backend_ids:
        backend = get_chord_detection_backend(backend_id)
        output_lab = output_dir / f"chords_{backend_id}.lab"
        start = time.perf_counter()
        try:
            predict_chords_from_wave(input_wav=input_wav, output_lab=output_lab, method=backend_id)
            runtime_s = time.perf_counter() - start
            labels = set()
            segment_count = 0
            for raw_line in output_lab.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                segment_count += 1
                labels.add(parts[2])
            results.append(
                ChordBenchmarkResult(
                    backend_id=backend_id,
                    label=backend.label,
                    output_lab=output_lab,
                    runtime_s=runtime_s,
                    segment_count=segment_count,
                    unique_label_count=len(labels),
                    success=True,
                )
            )
        except Exception as exc:
            runtime_s = time.perf_counter() - start
            results.append(
                ChordBenchmarkResult(
                    backend_id=backend_id,
                    label=backend.label,
                    output_lab=output_lab if output_lab.exists() else None,
                    runtime_s=runtime_s,
                    segment_count=0,
                    unique_label_count=0,
                    success=False,
                    error=str(exc),
                )
            )

    return results
