import logging
from pathlib import Path
import numpy as np
import soundfile as sf

from madmom.audio.signal import SignalProcessor
from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor
from demucs_audiosplit import logger


def _prepare_audio(input_path: Path, output_path: Path) -> None:
    data, sr = sf.read(str(input_path))
    if data.ndim > 1:
        logger.info(f"🔄 Converting stereo to mono for: {input_path.name}")
        data = np.mean(data, axis=1)
    data = data.astype(np.float32)
    sf.write(str(output_path), data, sr)


def predict_chords_from_wave(input_wav: Path, output_lab: Path) -> None:
    if not input_wav.exists():
        logger.error(f"Input file not found: {input_wav}")
        return

    try:
        logger.info(f"🎵 Running chord recognition on: {input_wav.name}")
        tmp_path = input_wav.with_name(f"{input_wav.stem}_madmom_tmp.wav")
        _prepare_audio(input_wav, tmp_path)

        # Construct processor chain manually
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        chroma = DeepChromaProcessor()
        decoder = DeepChromaChordRecognitionProcessor()

        audio = sig(str(tmp_path))         # Signal object
        chroma_features = chroma(audio)    # (T, 12) chroma features
        chords = decoder(chroma_features) # → (N, 3) [start, end, label]

        with output_lab.open("w") as f:
            for start, end, label in chords:
                f.write(f"{start:.3f}\t{end:.3f}\t{label}\n")

        logger.info(f"✅ Chord predictions saved to: {output_lab}")

    except Exception as e:
        logger.error(f"❌ Failed to predict chords for {input_wav.name}: {e}")
