from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set

from demucs_audiosplit.audiosplit import run_demucs


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
