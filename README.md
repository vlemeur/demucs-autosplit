# 🎵 Music Workbench

> A comprehensive music production toolkit powered by [Demucs](https://github.com/facebookresearch/demucs), written in Python with a Streamlit interface.

Repository name: `music-workbench`

Music Workbench provides tools for musicians, producers, and learners:
- **Stem Separation**: Split audio tracks into isolated stems (vocals, drums, bass, other)
- **Chord Detection**: Analyze audio stems to detect chord progressions
- **Live Harmony**: Real-time MIDI chord detection from your keyboard

---

## 📦 Features

### 🎚️ Stem Separation
* 🔹 Upload `.wav` or `.mp3` audio files
* 🔹 Automatic stem separation using Demucs (vocals, drums, bass, other, guitar, piano)
* 🔹 Download individual stems or all stems as a ZIP
* 🔹 Support for multiple Demucs models (htdemucs, htdemucs_ft, htdemucs_6s)

### 🎼 Chord Detection
* 🔹 Detect chords from audio stems using Madmom's deep learning pipeline
* 🔹 Visualize chord progressions with interactive waveform plots
* 🔹 Preview and play back specific sections of audio
* 🔹 Export chord annotations as .lab files

### 🎹 Live Harmony (NEW!)
* 🔹 Connect your MIDI keyboard and detect chords in real-time
* 🔹 See the notes you're playing and the detected chord
* 🔹 Create and save chord sequences for practice
* 🔹 Support for all standard chord types (maj, min, 7ths, suspended, diminished, augmented)

---

## 🚀 Getting started

### 1. Clone the repository

```bash
git clone git@github.com:vlemeur/music-workbench.git
cd music-workbench
```

### 2. Set up the environment

```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install -e .
```

This project pins `torch` / `torchaudio` to versions that avoid the newer
`torchcodec` runtime requirement on most platforms. If your virtualenv was
created before that change, refresh it with:

```bash
uv sync
```

### 3. (Optional) Enable prek hooks

```bash
uv run prek install
```

### 4. Quality checks

This project uses [Just](https://github.com/casey/just) as a modern alternative to Make.
Run the common quality commands with:

```bash
just format
just lint
just check
```

To install Just:
- **macOS**: `brew install just`
- **Linux**: `cargo install just` (requires Rust)
- **Other**: See [Just installation guide](https://github.com/casey/just#installation).

---

## 🎚️ Usage

### Run the Streamlit application

Launch the web interface locally:

```bash
uv run streamlit run src/ui.py
```

Then open your browser at:

```
http://localhost:8501
```

Music Workbench exposes three main workflows:

### 🎚️ Stem Separation Tab
* Upload an audio file (`.wav` or `.mp3`)
* Run Demucs separation with your choice of model
* Preview and download the resulting stems (vocals, drums, bass, other, etc.)

### 🎼 Chord Detection Tab
* Select a separated stem (typically "other" for harmonic content)
* Run chord detection to identify the chord progression
* Visualize chords on the waveform with zoom and playback controls
* Export chord annotations as .lab files

### 🎹 Live Harmony Tab (NEW!)
* Connect your MIDI keyboard
* Play chords and see them detected in real-time
* View the individual notes being played
* Create and manage chord sequences
* See alternative chord interpretations

---

## 🎹 Live Harmony Setup

To use the Live Harmony feature with your MIDI keyboard:

1. **Connect your keyboard** via USB or MIDI interface
2. **Verify it's detected**: On macOS, open "Audio MIDI Setup" app to check your device appears
3. **Ensure MIDI output is enabled** on your keyboard
4. **Install required packages**:
   ```bash
   uv pip install mido python-rtmidi
   ```
5. **Launch the app** and go to the "Live Harmony" tab
6. **Select your device** from the dropdown and click "Start"

For quick MIDI input debugging outside the UI:

```bash
python scripts/midi_monitor.py
```

If several MIDI inputs are available, the script selects the first one by default and prints the exact commands to open a different port.

## 🧠 Notes

* The application runs Demucs from the active Python environment.
* Output files are stored in a local workspace directory (`.streamlit_workdir/`).
* NumPy is pinned to `<2` for compatibility with PyTorch / torchaudio.
* For portability, the project pins `torchaudio` below `2.9` on non-Intel platforms.
* The product/repository name is `Music Workbench`; the internal package name `demucs_audiosplit` is still used in the codebase for compatibility.

---

## 📜 License

MIT
