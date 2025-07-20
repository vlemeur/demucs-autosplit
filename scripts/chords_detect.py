from pathlib import Path
from demucs_audiosplit.chords_predict import predict_chords_from_wave


TRACK_NAME = "looking_for_love"
STEM_NAME = "other.wav"


INPUT_WAV = Path(f"outputs/htdemucs/{TRACK_NAME}/{STEM_NAME}")
OUTPUT_LAB = Path(f"outputs/htdemucs/{TRACK_NAME}/chords.lab")

if __name__ == "__main__":
    predict_chords_from_wave(INPUT_WAV, OUTPUT_LAB)
