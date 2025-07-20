from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch
import torchaudio
from scipy.signal import resample
from torchaudio.functional import bandpass_biquad, highpass_biquad

from demucs_audiosplit import logger


def apply_simple_filters(
    stem_path: Path, low_center: float = 400.0, high_cutoff: float = 700.0
) -> None:
    """
    Apply frequency-based filtering to split a stem into lowband and highband approximations.

    Parameters
    ----------
    stem_path : Path
        Path to the input stem (e.g., 'other.wav').
    low_center : float, optional
        Central frequency in Hz for the bandpass (lowband).
    high_cutoff : float, optional
        Cutoff frequency in Hz for the high-pass filter (highband).
    """
    if not stem_path.exists():
        logger.warning(f"{stem_path.name} not found at {stem_path}")
        return

    try:
        waveform, sr = torchaudio.load(str(stem_path))

        # Low-band: bandpass filter (e.g., 150–700 Hz)
        lowband = bandpass_biquad(waveform, sr, central_freq=low_center, Q=0.707)
        lowband_path = stem_path.with_name(stem_path.stem + "_lowband.wav")
        torchaudio.save(str(lowband_path), lowband, sr)
        logger.info(f"📉 Saved lowband stem: {lowband_path.name}")

        # High-band: high-pass filter (>700 Hz)
        highband = highpass_biquad(waveform, sr, cutoff_freq=high_cutoff)
        highband_path = stem_path.with_name(stem_path.stem + "_highband.wav")
        torchaudio.save(str(highband_path), highband, sr)
        logger.info(f"📈 Saved highband stem: {highband_path.name}")

    except Exception as e:
        logger.error(f"❌ Error filtering {stem_path.name}: {e}")


def apply_yamnet_classification(stem_path: Path, top_k: int = 3) -> None:
    """
    Classify audio segments in a stem using YAMNet and extract segments labeled
    as guitar or keyboard.

    Parameters
    ----------
    stem_path : Path
        Path to the input stem (e.g., 'other.wav').
    top_k : int
        Number of top labels per frame to consider when matching guitar/keyboard.
    """
    if not stem_path.exists():
        logger.warning(f"{stem_path.name} not found at {stem_path}")
        return

    try:
        output_dir = stem_path.parent
        guitar_path = output_dir / f"{stem_path.stem}_yamnet_guitar.wav"
        keyboard_path = output_dir / f"{stem_path.stem}_yamnet_keyboard.wav"

        if guitar_path.exists() and keyboard_path.exists():
            logger.info("🧠 YAMNet results already exist — skipping classification.")
            return

        # Load and normalize audio
        waveform, sr = torchaudio.load(str(stem_path))
        waveform = waveform.mean(dim=0).numpy()  # mono
        original_sr = sr

        # Resample to 16kHz
        if sr != 16000:
            waveform = resample(waveform, int(len(waveform) * 16000 / sr))
        waveform = waveform / np.max(np.abs(waveform))  # normalize to [-1, 1]

        # Load YAMNet
        model = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = model.class_map_path().numpy().decode("utf-8")
        class_names = [
            line.strip() for line in Path(class_map_path).read_text().splitlines()
        ]

        # Inference
        waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
        scores, _, _ = model(waveform_tensor)
        scores = scores.numpy()

        frame_hop = 0.96  # seconds per frame
        label_counter = Counter()

        guitar_labels = {
            i for i, name in enumerate(class_names) if "guitar" in name.lower()
        }
        keyboard_labels = {
            i
            for i, name in enumerate(class_names)
            if any(k in name.lower() for k in ["keyboard", "organ"])
        }

        guitar_segments = []
        keyboard_segments = []

        for i, frame_scores in enumerate(scores):
            start_sample = int(i * frame_hop * original_sr)
            end_sample = start_sample + int(frame_hop * original_sr)
            if end_sample > len(waveform):
                break

            segment = waveform[start_sample:end_sample]
            top_indices = frame_scores.argsort()[-top_k:]
            label_counter.update(top_indices)

            if any(idx in guitar_labels for idx in top_indices):
                guitar_segments.append(segment)
            elif any(idx in keyboard_labels for idx in top_indices):
                keyboard_segments.append(segment)

        def save_segments(segments: list[np.ndarray], path: Path):
            if segments:
                audio = np.concatenate(segments)
                audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                torchaudio.save(str(path), audio_tensor, original_sr)
                logger.info(f"✅ Saved {path.name} ({len(segments)} segments)")
            else:
                logger.warning(
                    f"⚠️  No segments found for {path.name}, file not created."
                )

        save_segments(guitar_segments, guitar_path)
        save_segments(keyboard_segments, keyboard_path)

        logger.info("📊 Top YAMNet labels detected:")
        for idx, count in label_counter.most_common(10):
            logger.info(f"  - {class_names[idx]}: {count} frames")

    except Exception as e:
        logger.error(f"❌ Error in YAMNet classification for {stem_path.name}: {e}")
