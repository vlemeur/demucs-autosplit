from __future__ import annotations

import zipfile
from pathlib import Path

from audio_analysis.separation import run_demucs


def run_split(
    audio_path: Path,
    output_dir: Path,
    model: str = "htdemucs",
    segment: int = 7,
    device: str = "auto",
    two_stems: str | None = None,
    overlap: float = 0.25,
    shifts: int = 1,
    mp3: bool = False,
    mp3_bitrate: int = 320,
) -> Path | None:
    """Run stem separation on one file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_demucs(
        file_path=audio_path,
        output_dir=output_dir,
        model=model,
        segment=segment,
        device=device,
        two_stems=two_stems,
        overlap=overlap,
        shifts=shifts,
        mp3=mp3,
        mp3_bitrate=mp3_bitrate,
    )


def read_stems(stems_dir: Path, stems: list[str]) -> dict[str, bytes]:
    """Read stem wav files into memory."""
    return {stem: (stems_dir / f"{stem}.wav").read_bytes() for stem in stems}


def zip_stems(stems_dir: Path, stems: list[str]) -> bytes:
    """Build an in-memory zip archive for the selected stems."""
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for stem in stems:
            fp = stems_dir / f"{stem}.wav"
            zip_file.write(fp, arcname=fp.name)
    return buffer.getvalue()


def list_stems_wav(stems_dir: Path, stems: list[str]) -> dict[str, Path]:
    """Return only stem wav files that exist on disk."""
    existing: dict[str, Path] = {}
    for stem in stems:
        wav_path = stems_dir / f"{stem}.wav"
        if wav_path.exists():
            existing[stem] = wav_path
    return existing
