import argparse
from pathlib import Path

from audio_analysis import logger
from audio_analysis.chord_detection import list_chord_detection_backends, predict_chords_from_wave
from audio_analysis.separation import DEFAULT_MODEL, DEMUCS_MODELS, _get_expected_stems


def find_stems_dir(output_root: Path, track_name: str, model: str) -> Path | None:
    """
    Find the stems directory for a given track and model.

    Parameters
    ----------
    output_root : Path
        Root output directory.
    track_name : str
        Track name (without extension).
    model : str
        Demucs model name.

    Returns
    -------
    Path or None
        Path to the stems directory, or None if not found.
    """
    stems = _get_expected_stems(model)
    model_dir = output_root / model / track_name

    # Check if all stems exist
    if all((model_dir / f"{stem}.wav").exists() for stem in stems):
        return model_dir

    return None


def main() -> None:
    """
    Detect chords from a stem WAV file.

    Usage:
        python scripts/chords_detect.py [--track TRACK] [--stem STEM] [--model MODEL]

    Arguments:
        --track TRACK     Track name (without extension)
        --stem STEM       Stem name (e.g., 'other', 'vocals', 'guitar')
        --model MODEL     Separation model used for separation (default: htdemucs)
        --method METHOD   Chord detection backend (default: madmom)
    """
    backend_ids = [backend.backend_id for backend in list_chord_detection_backends()]
    parser = argparse.ArgumentParser(
        description="Detect chords from a stem WAV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--track",
        "-t",
        type=str,
        default="looking_for_love",
        help="Track name (without extension)",
    )
    parser.add_argument(
        "--stem",
        "-s",
        type=str,
        default="other",
        help="Stem name to analyze (e.g., 'other', 'vocals', 'guitar')",
    )
    parser.add_argument(
        "--model",
        "-n",
        type=str,
        default=DEFAULT_MODEL,
        choices=DEMUCS_MODELS,
        help=(
            f"Demucs-family model used. Options: {', '.join(DEMUCS_MODELS)}. "
            "Must match the model used for stem separation."
        ),
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default="madmom",
        choices=backend_ids,
        help=f"Chord detection backend. Options: {', '.join(backend_ids)}",
    )

    args = parser.parse_args()

    # Find stems directory
    output_root = Path("outputs")
    stems_dir = find_stems_dir(output_root, args.track, args.model)

    if stems_dir is None:
        logger.error(
            "Stems directory not found for track '%s' with model '%s'",
            args.track,
            args.model,
        )
        return

    # Check if stem exists
    input_wav = stems_dir / f"{args.stem}.wav"
    if not input_wav.exists():
        stems = _get_expected_stems(args.model)
        logger.error(
            "Stem '%s' not found. Available stems: %s",
            args.stem,
            ", ".join(stems),
        )
        return

    output_lab = stems_dir / f"chords_{args.method}.lab"
    bass_wav = stems_dir / "bass.wav"
    if not bass_wav.exists() or bass_wav.resolve() == input_wav.resolve():
        bass_wav = None

    logger.info("🎵 Detecting chords from: %s", input_wav.name)
    predict_chords_from_wave(input_wav, output_lab, method=args.method, bass_wav=bass_wav)
    logger.info("✅ Chord detection complete. Output: %s", output_lab)


if __name__ == "__main__":
    main()
