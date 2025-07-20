from pathlib import Path
from demucs_audiosplit import logger
from demucs_audiosplit.audiosplit import find_audio_files, run_demucs



AUDIO_DIR = Path("audio")
OUTPUT_DIR = Path("outputs")


def main() -> None:
    """
    Batch process all audio files in AUDIO_DIR using Demucs.
    """
    AUDIO_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    audio_files = find_audio_files(AUDIO_DIR)

    if not audio_files:
        logger.warning("⚠️  No .wav or .mp3 files found in 'audio/'")
        return

    for file in audio_files:
        run_demucs(file, OUTPUT_DIR)

    logger.info("✅ All files processed.")


if __name__ == "__main__":
    main()
