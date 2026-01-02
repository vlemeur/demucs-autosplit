"""Main service."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
from demucs_audiosplit.audiosplit import run_demucs
from demucs_audiosplit.chords_predict import predict_chords_from_wave


@dataclass(frozen=True)
class ChordSegment:
    """
    A chord segment parsed from a .lab file.

    Attributes
    ----------
    start_s : float
        Segment start time in seconds.
    end_s : float
        Segment end time in seconds.
    label : str
        Chord label (e.g. "C:maj", "A:min", "N").
    """

    start_s: float
    end_s: float
    label: str


def read_chords_lab(lab_path: Path) -> List[ChordSegment]:
    """
    Read a chords .lab file with lines formatted as: <start> <end> <label>.

    Parameters
    ----------
    lab_path : Path
        Path to the .lab file.

    Returns
    -------
    list of ChordSegment
        Parsed chord segments in file order.

    Raises
    ------
    FileNotFoundError
        If the lab file does not exist.
    ValueError
        If a line cannot be parsed.
    """
    if not lab_path.exists():
        raise FileNotFoundError(f"File not found: {lab_path}")

    segments: List[ChordSegment] = []
    for raw_line in lab_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split(maxsplit=2)
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
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load a wav file and return a downsampled mono waveform for plotting.

    The waveform is converted to mono by averaging channels. If the file is
    long, the returned arrays are downsampled by striding to keep plotting fast.

    Parameters
    ----------
    wav_path : Path
        Path to the wav file.
    max_points : int, default=200_000
        Maximum number of points returned for plotting performance.

    Returns
    -------
    times_s : numpy.ndarray
        Time axis in seconds.
    mono : numpy.ndarray
        Mono waveform samples (float32).
    duration_s : float
        Total duration in seconds.

    Raises
    ------
    FileNotFoundError
        If the wav file does not exist.
    RuntimeError
        If the audio cannot be read.
    ValueError
        If max_points is not positive.
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"File not found: {wav_path}")

    if max_points <= 0:
        raise ValueError("max_points must be a positive integer")

    try:
        # always_2d ensures shape (n_samples, n_channels)
        audio, sample_rate = sf.read(wav_path, always_2d=True)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise RuntimeError(f"Failed to read audio: {wav_path}") from exc

    if audio.size == 0:
        raise RuntimeError(f"Empty audio file: {wav_path}")

    # Convert to mono by averaging channels
    mono = audio.mean(axis=1).astype(np.float32)
    n_samples = int(mono.shape[0])
    sr = float(sample_rate)
    duration_s = float(n_samples) / sr

    # Downsample by stride for performance
    if n_samples > max_points:
        stride = int(np.ceil(n_samples / float(max_points)))
        mono = mono[::stride]
        times_s = (np.arange(mono.shape[0], dtype=np.float32) * float(stride)) / sr
    else:
        times_s = np.arange(n_samples, dtype=np.float32) / sr

    return times_s, mono, duration_s


def safe_filename(name: str) -> str:
    """
    Sanitize a filename to reduce filesystem issues.

    Parameters
    ----------
    name : str
        Original filename.

    Returns
    -------
    str
        Sanitized filename.
    """
    return "".join(c for c in name if c.isalnum() or c in ("-", "_", ".", " ")).strip()


def save_bytes_to_file(data: bytes, dest_path: Path) -> Path:
    """
    Save raw bytes to disk.

    Parameters
    ----------
    data : bytes
        File content.
    dest_path : Path
        Destination path.

    Returns
    -------
    Path
        Path to the saved file.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(data)
    return dest_path


def validate_extension(file_path: Path, supported_ext: Set[str]) -> bool:
    """
    Check whether a file extension is supported.

    Parameters
    ----------
    file_path : Path
        File to validate.
    supported_ext : set of str
        Supported extensions (e.g. {".wav", ".mp3"}).

    Returns
    -------
    bool
        True if supported, otherwise False.
    """
    return file_path.suffix.lower() in supported_ext


def run_split(audio_path: Path, output_dir: Path, try_filter_others: bool) -> None:
    """
    Run Demucs on a single audio file.

    Parameters
    ----------
    audio_path : Path
        Input audio file.
    output_dir : Path
        Output root directory.
    try_filter_others : bool
        Enable optional filtering behavior.

    Returns
    -------
    None
        Demucs writes outputs to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    run_demucs(
        file_path=audio_path,
        output_dir=output_dir,
        try_filter_others=try_filter_others,
    )


def find_stems_dir(output_root: Path, track_name: str, stems: List[str]) -> Optional[Path]:
    """
    Locate the directory containing all stem wav files for a given track.

    This is robust to output layouts like:
    outputs/<model>/<track_name>/<stem>.wav

    Parameters
    ----------
    output_root : Path
        Root Demucs output directory.
    track_name : str
        Track name (usually input file stem).
    stems : list of str
        Expected stem base names (e.g. ["drums", "bass", "other", "vocals"]).

    Returns
    -------
    Path or None
        Directory containing all stems, or None if not found.
    """
    candidates: List[Path] = []

    # Heuristic 1: directories matching track_name
    for p in output_root.rglob(track_name):
        if p.is_dir():
            candidates.append(p)

    # Heuristic 2: any directory under output_root
    for p in output_root.rglob("*"):
        if p.is_dir():
            candidates.append(p)

    def has_all(directory: Path) -> bool:
        return all((directory / f"{stem}.wav").exists() for stem in stems)

    def score(directory: Path) -> int:
        present = sum((directory / f"{stem}.wav").exists() for stem in stems)
        return present * 100 - len(directory.parts)

    best: Optional[Path] = None
    best_score = -(10**9)

    for directory in candidates:
        s = score(directory)
        if s > best_score:
            best_score = s
            best = directory

    if best is not None and has_all(best):
        return best
    return None


def read_stems(stems_dir: Path, stems: List[str]) -> Dict[str, bytes]:
    """
    Read stem wav files into memory.

    Parameters
    ----------
    stems_dir : Path
        Directory containing the stem files.
    stems : list of str
        Stem base names.

    Returns
    -------
    dict of str to bytes
        Mapping {stem_name: wav_bytes}.
    """
    data: Dict[str, bytes] = {}
    for stem in stems:
        data[stem] = (stems_dir / f"{stem}.wav").read_bytes()
    return data


def zip_stems(stems_dir: Path, stems: List[str]) -> bytes:
    """
    Create an in-memory ZIP archive containing all stems.

    Parameters
    ----------
    stems_dir : Path
        Directory containing the stem files.
    stems : list of str
        Stem base names.

    Returns
    -------
    bytes
        ZIP archive bytes with `<stem>.wav` entries.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for stem in stems:
            fp = stems_dir / f"{stem}.wav"
            zip_file.write(fp, arcname=fp.name)
    return buffer.getvalue()


def clear_workspace(work_dir: Path) -> None:
    """
    Best-effort cleanup of the workspace directory.

    Parameters
    ----------
    work_dir : Path
        Workspace directory.

    Returns
    -------
    None
    """
    if not work_dir.exists():
        return

    for p in sorted(work_dir.rglob("*"), reverse=True):
        try:
            if p.is_file():
                p.unlink()
            else:
                p.rmdir()
        except OSError:
            # Ignore errors (locked files, non-empty dirs, etc.)
            pass


def list_stems_wav(stems_dir: Path, stems: List[str]) -> Dict[str, Path]:
    """
    List existing stem wav files in a directory.

    Parameters
    ----------
    stems_dir : Path
        Directory containing stems.
    stems : list of str
        Expected stem base names.

    Returns
    -------
    dict of str to Path
        Mapping {stem_name: wav_path} for stems that exist on disk.
    """
    existing: Dict[str, Path] = {}
    for stem in stems:
        wav_path = stems_dir / f"{stem}.wav"
        if wav_path.exists():
            existing[stem] = wav_path
    return existing


def predict_chords_for_stem(input_wav: Path, output_lab: Path) -> Path:
    """
    Predict chords from a wav file and write a .lab output.

    Parameters
    ----------
    input_wav : Path
        Input wav path (one stem).
    output_lab : Path
        Output .lab file path (created/overwritten).

    Returns
    -------
    Path
        The output_lab path.

    Raises
    ------
    FileNotFoundError
        If the input wav does not exist.
    """
    if not input_wav.exists():
        raise FileNotFoundError(f"Stem wav not found: {input_wav}")

    output_lab.parent.mkdir(parents=True, exist_ok=True)
    predict_chords_from_wave(input_wav, output_lab)
    return output_lab


def read_text_file(path: Path) -> str:
    """
    Read a UTF-8 text file from disk.

    Parameters
    ----------
    path : Path
        Text file path.

    Returns
    -------
    str
        File content as a string.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return path.read_text(encoding="utf-8")
