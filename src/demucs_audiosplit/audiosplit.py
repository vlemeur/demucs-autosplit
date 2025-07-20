import subprocess
from pathlib import Path
from typing import List


AUDIO_DIR = Path("audio")
OUTPUT_DIR = Path("outputs")


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


def run_demucs(file_path: Path, output_dir: Path) -> None:
    """
    Run Demucs separation on a single audio file.

    Parameters
    ----------
    file_path : Path
        The path to the input audio file.
    output_dir : Path
        The directory where the separated stems will be saved.
    """
    print(f"🔍 Separating: {file_path.name}")
    try:
        subprocess.run(
            ["demucs", "--out", str(output_dir), str(file_path)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to process {file_path.name}: {e}")


def main() -> None:
    """
    Batch process all audio files in AUDIO_DIR using Demucs.
    """
    AUDIO_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    audio_files = find_audio_files(AUDIO_DIR)

    if not audio_files:
        print("⚠️  No .wav or .mp3 files found in 'audio/'")
        return

    for file in audio_files:
        run_demucs(file, OUTPUT_DIR)

    print("✅ Separation complete.")


if __name__ == "__main__":
    main()
