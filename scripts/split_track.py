from pathlib import Path

from demucs_audiosplit import logger
from demucs_audiosplit.audiosplit import find_audio_files, run_demucs


AUDIO_DIR = Path("audio")
OUTPUT_DIR = Path("outputs")
TRY_FILTERS_OTHERS = False


def main() -> None:
    """
    Batch process all audio files in AUDIO_DIR using Demucs.
    """
    AUDIO_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    audio_files = find_audio_files(AUDIO_DIR, extensions=[".wav", ".mp3"])

    if not audio_files:
        logger.warning("⚠️  No .wav or .mp3 files found in 'audio/'")
        return

    for file in audio_files:
        run_demucs(
            file_path=file, output_dir=OUTPUT_DIR, try_filter_others=TRY_FILTERS_OTHERS
        )

    logger.info("✅ All files processed.")


if __name__ == "__main__":
    main()
