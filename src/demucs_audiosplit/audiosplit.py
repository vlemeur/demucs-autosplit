import subprocess
import sys
from pathlib import Path

from demucs_audiosplit import logger
from demucs_audiosplit.filters import apply_simple_filters

# Available Demucs v4 models (Hybrid Transformer)
# These are the state-of-the-art models with the best quality
DEMUCS_MODELS: tuple[str, ...] = (
    "htdemucs",  # Default: Hybrid Transformer, 9.0 dB SDR
    "htdemucs_ft",  # Fine-tuned: best quality, 9.2 dB SDR (4x slower)
    "htdemucs_6s",  # 6 sources: adds guitar and piano stems
)

# Default model
DEFAULT_MODEL: str = "htdemucs"

# Maximum segment length for Hybrid Transformer models (in seconds)
# These models cannot handle segments longer than 7.8 seconds
MAX_SEGMENT_HYBRID_TRANSFORMER: float = 7.8

# Default segment for Hybrid Transformer models
DEFAULT_SEGMENT: int = 7


def find_audio_files(directory: Path, extensions: list[str]) -> list[Path]:
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


def _get_expected_stems(model: str) -> list[str]:
    """
    Get the expected stem files for a given Demucs model.

    Parameters
    ----------
    model : str
        Demucs model name.

    Returns
    -------
    list[str]
        List of expected stem file names (without .wav extension).
    """
    if model in ("htdemucs_6s",):
        return ["vocals", "drums", "bass", "other", "guitar", "piano"]
    return ["vocals", "drums", "bass", "other"]


def _get_model_dir(model: str) -> str:
    """
    Get the output directory name for a given Demucs model.

    Parameters
    ----------
    model : str
        Demucs model name.

    Returns
    -------
    str
        Directory name where stems will be saved.
    """
    # Demucs uses the model name as the directory
    return model


def run_demucs(
    file_path: Path,
    output_dir: Path,
    try_filter_others: bool = False,
    model: str = DEFAULT_MODEL,
    segment: int = DEFAULT_SEGMENT,
    device: str = "auto",
    two_stems: str | None = None,
    overlap: float = 0.25,
    shifts: int = 1,
    mp3: bool = False,
    mp3_bitrate: int = 320,
) -> Path | None:
    """
    Run Demucs separation on a single audio file.

    Parameters
    ----------
    file_path : Path
        The path to the input audio file.
    output_dir : Path
        The directory where the separated stems will be saved.
    try_filter_others : bool, optional
        If True, try to apply extra filters on the 'other.wav' stem. Default is False.
    model : str, optional
        Demucs model to use. Options: "htdemucs", "htdemucs_ft", "htdemucs_6s".
        Default is "htdemucs".
    segment : int, optional
        Segment length in seconds for GPU processing. Default is 7.
        **Note**: Hybrid Transformer models support max 7.8 seconds.
        Smaller values use less memory but may reduce quality.
    device : str, optional
        Device to use: "cpu", "cuda", or "auto". Default is "auto".
    two_stems : str or None, optional
        If provided, only separate this source from the rest (karaoke mode).
        Must be one of the expected stems for the model.
    overlap : float, optional
        Overlap between segments (0.0 to 1.0). Default is 0.25.
    shifts : int, optional
        Number of random shifts for prediction averaging. Default is 1.
        Higher values improve quality but increase processing time.
    mp3 : bool, optional
        If True, save output as MP3 instead of WAV. Default is False.
    mp3_bitrate : int, optional
        MP3 bitrate in kbps. Default is 320.

    Returns
    -------
    Path or None
        Path to the directory containing the separated stems, or None if failed.

    Raises
    ------
    ValueError
        If model is not supported or parameters are invalid.

    Notes
    -----
    The `htdemucs_ft` model is fine-tuned and produces better quality but takes
    4x longer to process. The `htdemucs_6s` model separates into 6 stems instead
    of 4, adding guitar and piano tracks.

    **Hybrid Transformer models cannot use segments longer than 7.8 seconds.**

    For GPU processing, ensure you have enough memory. Use smaller `segment`
    values if you encounter out-of-memory errors.
    """
    if model not in DEMUCS_MODELS:
        raise ValueError(
            f"Unsupported Demucs model: {model}. Available models: {', '.join(DEMUCS_MODELS)}"
        )

    # Validate segment length for Hybrid Transformer models
    if segment > MAX_SEGMENT_HYBRID_TRANSFORMER:
        raise ValueError(
            f"Segment length {segment}s is too long for Hybrid Transformer models. "
            f"Maximum supported is {MAX_SEGMENT_HYBRID_TRANSFORMER}s. "
            f"Please use --segment {int(MAX_SEGMENT_HYBRID_TRANSFORMER)} or lower."
        )

    expected_stems = _get_expected_stems(model)
    model_dir = _get_model_dir(model)
    stem_dir = output_dir / model_dir / file_path.stem

    existing = [stem_dir / f"{stem}.wav" for stem in expected_stems]

    if all(path.exists() for path in existing):
        logger.info("⏭️  Skipping '%s': stems already exist for model %s.", file_path.name, model)
        return stem_dir

    logger.info("🔍 Separating '%s' with model %s", file_path.name, model)

    # Build the command
    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--out",
        str(output_dir),
        "-n",
        model,
        "--segment",
        str(segment),
        "--overlap",
        str(overlap),
        "--shifts",
        str(shifts),
    ]

    # Only add device if it's not "auto" - Demucs handles auto-detection by default
    if device != "auto":
        cmd.extend(["-d", device])

    if two_stems:
        cmd.extend(["--two-stems", two_stems])

    if mp3:
        cmd.extend(["--mp3", "--mp3-bitrate", str(mp3_bitrate)])

    cmd.append(str(file_path))

    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
    except subprocess.TimeoutExpired:
        logger.error("❌ Demucs timed out after 1 hour for %s", file_path.name)
        return None
    except subprocess.CalledProcessError as exc:
        logger.error(
            "❌ Failed to process %s with model %s (returncode=%s).\nstdout: %s\nstderr: %s",
            file_path.name,
            model,
            exc.returncode,
            (exc.stdout or "").strip(),
            (exc.stderr or "").strip(),
        )
        return None
    except FileNotFoundError:
        logger.error("❌ Demucs is not installed in the current Python environment.")
        return None

    if completed.stdout.strip():
        logger.info("%s", completed.stdout.strip())

    # Verify stems were created
    if not all(path.exists() for path in existing):
        logger.error(
            "❌ Demucs completed but expected stems not found for model %s in %s",
            model,
            stem_dir,
        )
        return None

    logger.info("✅ Successfully separated '%s' with model %s", file_path.name, model)

    if not try_filter_others:
        return stem_dir

    # Apply extra filters to 'other' stem
    extra_stems = ["other_lowband.wav", "other_highband.wav"]
    extra_existing = [stem_dir / stem for stem in extra_stems]

    if all(path.exists() for path in extra_existing):
        logger.info(
            "✅ Extra filters already produced low/high band stems for '%s'.", file_path.name
        )
        return stem_dir

    other_path = stem_dir / "other.wav"
    try:
        apply_simple_filters(other_path)
        logger.info("✅ Applied extra filters to 'other' stem for '%s'", file_path.name)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("❌ Extra extraction failed for %s: %s", other_path.name, exc)

    return stem_dir
