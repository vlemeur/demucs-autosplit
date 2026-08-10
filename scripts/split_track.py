import argparse
from pathlib import Path

from demucs_audiosplit import logger
from demucs_audiosplit.audiosplit import DEFAULT_MODEL, DEMUCS_MODELS, find_audio_files, run_demucs

AUDIO_DIR = Path("audio")
OUTPUT_DIR = Path("outputs")
TRY_FILTERS_OTHERS = False


def main() -> None:
    """
    Batch process all audio files in AUDIO_DIR using Demucs.

    Usage:
        python scripts/split_track.py [--model MODEL] [--filter-others]

    Arguments:
        --model MODEL       Demucs model to use (default: htdemucs)
        --filter-others    Apply extra filters to the 'other' stem (default: False)
        --segment SEGMENT   Segment length in seconds for GPU processing (default: 10)
        --device DEVICE     Device to use: cpu, cuda, or auto (default: auto)
    """
    parser = argparse.ArgumentParser(
        description="Batch process audio files with Demucs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "-n",
        type=str,
        default=DEFAULT_MODEL,
        choices=DEMUCS_MODELS,
        help=f"Demucs model to use. Options: {', '.join(DEMUCS_MODELS)}",
    )
    parser.add_argument(
        "--filter-others",
        action="store_true",
        default=TRY_FILTERS_OTHERS,
        help="Apply extra filters to the 'other' stem",
    )
    parser.add_argument(
        "--segment",
        type=int,
        default=10,
        help="Segment length in seconds for GPU processing (default: 10)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device to use: cpu, cuda, or auto (default: auto)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Overlap between segments (0.0 to 1.0, default: 0.25)",
    )
    parser.add_argument(
        "--shifts",
        type=int,
        default=1,
        help="Number of random shifts for prediction averaging (default: 1)",
    )

    args = parser.parse_args()

    AUDIO_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    audio_files = find_audio_files(AUDIO_DIR, extensions=[".wav", ".mp3"])

    if not audio_files:
        logger.warning("⚠️  No .wav or .mp3 files found in 'audio/'")
        return

    logger.info("🎵 Processing %d audio file(s) with model %s", len(audio_files), args.model)

    for file in audio_files:
        run_demucs(
            file_path=file,
            output_dir=OUTPUT_DIR,
            try_filter_others=args.filter_others,
            model=args.model,
            segment=args.segment,
            device=args.device,
            overlap=args.overlap,
            shifts=args.shifts,
        )

    logger.info("✅ All files processed.")


if __name__ == "__main__":
    main()
