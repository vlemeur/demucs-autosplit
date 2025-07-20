import subprocess
from pathlib import Path
from typing import List
from demucs_audiosplit import logger
from demucs_audiosplit.filters import apply_simple_filters


def find_audio_files(directory: Path, extensions: List[str] = None) -> List[Path]:
    """
    Find all audio files in a given directory with specified extensions.

    Parameters
    ----------
    directory : Path
        The directory to search in.
    extensions : List[str], optional
        A list of file extensions to include (default is [".wav", ".mp3"]).

    Returns
    -------
    List[Path]
        A list of matching audio file paths.
    """
    return [f for f in directory.iterdir() if f.suffix.lower() in extensions and f.is_file()]


def run_demucs(file_path: Path, output_dir: Path, try_filter_others: bool = False) -> None:
    """
    Run Demucs separation on a single audio file.

    Parameters
    ----------
    file_path : Path
        The path to the input audio file.
    output_dir : Path
        The directory where the separated stems will be saved.
    try_filter_others : bool
        try to apply extra filters on others.wav track
    """
    already_processed = False
    stem_dir = output_dir  / "htdemucs" / file_path.stem

    # Check if all stem files already exist
    expected_stems = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
    existing = [stem_dir / stem for stem in expected_stems]

    if all(path.exists() for path in existing):
        logger.info(f"⏭️  Skipping '{file_path.name}': stems already exist.")
        already_processed = True
    logger.info(f"🔍 Separating: {file_path.name}")
    if not already_processed:
        try:
            subprocess.run(
                ["demucs", "--out", str(output_dir), str(file_path)],
                check=True
            )
        except subprocess.CalledProcessError as e:
            logger.info(f"❌ Failed to process {file_path.name}: {e}")
    if try_filter_others:
        expected_stems = ["other_lowband.wav", "other_highband.wav"]
        existing = [stem_dir / stem for stem in expected_stems]
        if not all(path.exists() for path in existing):
            try:
                # Try to filter 'other.wav' from stem output
                stem_dir = output_dir /  "htdemucs" / file_path.stem
                other_path = stem_dir / "other.wav"
                apply_simple_filters(other_path)
            except:
                logger.info(f"❌ Failed to perform extra extraction for  {other_path.name}: {e}")
        else:
            logger.info("Already extracted with extra filters")

