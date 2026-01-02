import subprocess
from pathlib import Path
from typing import List

from demucs_audiosplit import logger
from demucs_audiosplit.filters import apply_simple_filters


def find_audio_files(directory: Path, extensions: List[str]) -> List[Path]:
    """
    Find all audio files in a given directory with specified extensions.

    Parameters
    ----------
    directory : Path
        The directory to search in.
    extensions : List[str]
        A list of file extensions to include.

    Returns
    -------
    List[Path]
        A list of matching audio file paths.
    """
    return [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in extensions]


def run_demucs(file_path: Path, output_dir: Path, try_filter_others: bool = False) -> None:
    """
    Run Demucs separation on a single audio file.

    Parameters
    ----------
    file_path : Path
        The path to the input audio file.
    output_dir : Path
        The directory where the separated stems will be saved.
    try_filter_others : bool, optional
        If True, try to apply extra filters on the 'other.wav' stem.

    Returns
    -------
    None
    """
    stem_dir = output_dir / "htdemucs" / file_path.stem

    expected_stems = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
    existing = [stem_dir / stem for stem in expected_stems]

    if all(path.exists() for path in existing):
        logger.info("⏭️  Skipping '%s': stems already exist.", file_path.name)
        return

    logger.info("🔍 Separating: %s", file_path.name)

    try:
        subprocess.run(
            ["demucs", "--out", str(output_dir), str(file_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        # Use error level for failures
        logger.error(
            "❌ Failed to process %s (returncode=%s). stderr=%s",
            file_path.name,
            exc.returncode,
            (exc.stderr or "").strip(),
        )
        return
    except FileNotFoundError:
        # demucs executable not found
        logger.error("❌ 'demucs' command not found. Is it installed and on PATH?")
        return

    if not try_filter_others:
        return

    extra_stems = ["other_lowband.wav", "other_highband.wav"]
    extra_existing = [stem_dir / stem for stem in extra_stems]

    if all(path.exists() for path in extra_existing):
        logger.info(
            "✅ Extra filters already produced low/high band stems for '%s'.", file_path.name
        )
        return

    other_path = stem_dir / "other.wav"
    try:
        apply_simple_filters(other_path)
    except (OSError, ValueError, RuntimeError) as exc:
        # Keep it specific: adjust exceptions based on what apply_simple_filters can raise
        logger.error("❌ Extra extraction failed for %s: %s", other_path.name, exc)
