import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from madmom.audio.chroma import DeepChromaProcessor
from madmom.audio.signal import SignalProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor

logger = logging.getLogger(__name__)

# Mapping simplifié : à compléter si besoin
CHORD_MAPPING = {
    "C:maj": ["C4", "E4", "G4"],
    "C:min": ["C4", "Eb4", "G4"],
    "C#:maj": ["C#4", "F4", "G#4"],
    "C#:min": ["C#4", "E4", "G#4"],
    "D:maj": ["D4", "F#4", "A4"],
    "D:min": ["D4", "F4", "A4"],
    "D#:maj": ["D#4", "G4", "A#4"],
    "D#:min": ["D#4", "F#4", "A#4"],
    "E:maj": ["E4", "G#4", "B4"],
    "E:min": ["E4", "G4", "B4"],
    "F:maj": ["F4", "A4", "C5"],
    "F:min": ["F4", "Ab4", "C5"],
    "F#:maj": ["F#4", "A#4", "C#5"],
    "F#:min": ["F#4", "A4", "C#5"],
    "G:maj": ["G4", "B4", "D5"],
    "G:min": ["G4", "A#4", "D5"],
    "G#:maj": ["G#4", "C5", "D#5"],
    "G#:min": ["G#4", "B4", "D#5"],
    "A:maj": ["A4", "C#5", "E5"],
    "A:min": ["A4", "C5", "E5"],
    "A#:maj": ["A#4", "D5", "F5"],
    "A#:min": ["A#4", "C#5", "F5"],
    "B:maj": ["B4", "D#5", "F#5"],
    "B:min": ["B4", "D5", "F#5"],
}


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

        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        chroma = DeepChromaProcessor()
        decoder = DeepChromaChordRecognitionProcessor()

        audio = sig(str(tmp_path))
        chroma_features = chroma(audio)
        chords = decoder(chroma_features)

        with output_lab.open("w") as f:
            for start, end, label in chords:
                notes = CHORD_MAPPING.get(label, [])
                notes_str = ",".join(notes) if notes else "-"
                f.write(f"{start:.3f}\t{end:.3f}\t{label}\t{notes_str}\n")

        logger.info(f"✅ Chord predictions saved to: {output_lab}")

    except Exception as e:
        logger.error(f"❌ Failed to predict chords for {input_wav.name}: {e}")
