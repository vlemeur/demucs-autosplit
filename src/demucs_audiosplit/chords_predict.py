import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from madmom.audio.chroma import DeepChromaProcessor
from madmom.audio.signal import SignalProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor

from demucs_audiosplit.chords_library import get_chord_notes

logger = logging.getLogger(__name__)


def _prepare_audio(input_path: Path, output_path: Path) -> None:
    """
    Prepare audio file for chord detection.

    Converts stereo to mono and ensures float32 format at 44100 Hz.

    Parameters
    ----------
    input_path : Path
        Path to the input audio file.
    output_path : Path
        Path to save the prepared mono audio file.

    Raises
    ------
    FileNotFoundError
        If input_path does not exist.
    RuntimeError
        If audio cannot be read or written.
    """
    data, sr = sf.read(str(input_path))
    if data.ndim > 1:
        logger.info(f"🔄 Converting stereo to mono for: {input_path.name}")
        data = np.mean(data, axis=1)

    # Normalize to prevent clipping
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val * 0.99

    data = data.astype(np.float32)

    # Madmom expects 44100 Hz
    if sr != 44100:
        from scipy.signal import resample

        data = resample(data, int(len(data) * 44100 / sr))
        sr = 44100

    sf.write(str(output_path), data, sr, format="WAV")


def predict_chords_from_wave(
    input_wav: Path,
    output_lab: Path,
    method: str = "madmom",
) -> None:
    """
    Predict chords from a WAV file and save to a .lab file.

    Parameters
    ----------
    input_wav : Path
        Path to the input WAV file (mono or stereo).
    output_lab : Path
        Path to the output .lab file (will be created/overwritten).
    method : str, default="madmom"
        Chord detection method. Currently only "madmom" is implemented.
        Future: "chordino" for external Chordino tool.

    Raises
    ------
    FileNotFoundError
        If input_wav does not exist.
    ValueError
        If method is not supported.
    RuntimeError
        If chord prediction fails.

    Notes
    -----
    The output .lab file format is:
    START_TIME\tEND_TIME\tCHORD_LABEL\tNOTES

    Example:
    0.000\t1.500\tC:maj\tC3,E3,G3
    """
    if method != "madmom":
        raise ValueError(
            f"Unsupported chord detection method: {method}. Use 'madmom' or install chordino."
        )

    if not input_wav.exists():
        raise FileNotFoundError(f"Input file not found: {input_wav}")

    try:
        logger.info(f"🎵 Running chord recognition on: {input_wav.name} (method={method})")
        tmp_path = input_wav.with_name(f"{input_wav.stem}_madmom_tmp.wav")
        _prepare_audio(input_wav, tmp_path)

        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        chroma = DeepChromaProcessor()
        decoder = DeepChromaChordRecognitionProcessor()

        audio = sig(str(tmp_path))
        chroma_features = chroma(audio)
        chords = decoder(chroma_features)

        # Clean up temp file
        tmp_path.unlink(missing_ok=True)

        with output_lab.open("w", encoding="utf-8") as f:
            for start, end, label in chords:
                notes = get_chord_notes(label)
                notes_str = ",".join(notes) if notes else "-"
                f.write(f"{start:.3f}\t{end:.3f}\t{label}\t{notes_str}\n")

        logger.info(f"✅ Chord predictions saved to: {output_lab}")

    except Exception as e:
        logger.error(f"❌ Failed to predict chords for {input_wav.name}: {e}")
        raise RuntimeError(f"Chord prediction failed: {e}") from e
