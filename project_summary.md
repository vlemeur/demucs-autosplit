
# 🎛️ Project: `demucs-autosplit` – Automatic Audio Splitting and Chord Analysis

## 🧠 Goal

The main goal of this project is to:

1. **Automatically extract instrumental stems** (vocals, drums, bass, others) from a single audio track (MP3 or WAV).
2. **Isolate the harmonic stem** (typically guitar, keyboard, synth).
3. **Automatically detect the chord progression** throughout the track.
4. **Translate each chord into actual piano notes**, from bass to treble, for interpretation or performance.

---

## 🧭 Processing Pipeline

Here's the step-by-step pipeline that runs automatically:

### 1. 🎵 User Input

- The user drops a `.mp3` or `.wav` file into the `audio/` folder.

### 2. 🎚️ Stem Separation with Demucs

- Uses the pretrained `htdemucs` model to split audio into 4 stems:
  - `vocals.wav`, `bass.wav`, `drums.wav`, `other.wav`.
- The `other.wav` file contains most of the **harmonic instruments** (guitar, piano, synths...).

### 3. 🎼 Chord Detection with Madmom

- A deep learning pipeline runs:
  - **Deep Chroma Features** extracted using a CNN.
  - **CRF Decoder** (Conditional Random Field) to smooth chord transitions.
- Outputs timestamped chords as `[start, end, label]`.

### 4. 🎹 Chord-to-Notes Mapping

- Each chord label (e.g. `A:min`, `F#:maj`) is mapped to **specific piano notes** (C4-style).
- Example:

  ```
  12.5   14.8   A:min   A4,C5,E5
  ```

### 5. 🗂️ Output

- A `.lab` file contains:
  - Start and end time
  - Chord label
  - Detailed piano notes
- Ready for practice, analysis or teaching.

---

## 🧑‍🔬 Models Used

### 1. Demucs (`htdemucs`) – Facebook AI Research

- **Type**: Convolutional U-Net with LSTM layers.
- **Purpose**: Separate audio into four stems.
- **Input**: Stereo audio.
- **Output**: Clean stems.
- **Trained on**: MUSDB18HQ dataset.
- **Highlights**: Robust performance on real-world mixed tracks.

### 2. Madmom – Chord Recognition Pipeline

- **Key components**:
  - `SignalProcessor`: mono conversion and resampling.
  - `DeepChromaProcessor`: pretrained CNN for chroma feature extraction (12D).
  - `DeepChromaChordRecognitionProcessor`: CRF-based sequence model to detect the most likely chord series.
- **Trained on**: datasets with labeled chord progressions.
- **Chords**: major/minor only.

---
