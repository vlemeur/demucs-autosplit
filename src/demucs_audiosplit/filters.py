import logging
from pathlib import Path
import torchaudio
from torchaudio.functional import highpass_biquad, bandpass_biquad


logger = logging.getLogger(__name__)



def apply_simple_filters(stem_path: Path, low_center: float = 400.0, high_cutoff: float = 700.0) -> None:
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

