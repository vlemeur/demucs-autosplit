# 🎧 demucs-autosplit

> Audio stem separation tool powered by [Demucs](https://github.com/facebookresearch/demucs), written in Python and packaged with a simple Streamlit interface.

This project provides an easy way to split an audio track into isolated stems (vocals, drums, bass, other) using Demucs.
It is designed for musicians, learners, and lightweight audio processing workflows.

---

## 📦 Features

* 🔹 Upload `.wav` or `.mp3` audio files
* 🔹 Automatic stem separation (vocals, drums, bass, other)
* 🔹 Download individual stems or all stems as a ZIP
* 🔹 Simple Streamlit web interface
* 🔹 Clean Python project structure with linters and pre-commit hooks

---

## 🚀 Getting started

### 1. Clone the repository

```bash
git clone git@github.com:vlemeur/demucs-autosplit.git
cd demucs-autosplit
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

### 3. (Optional) Enable pre-commit hooks

```bash
uv run pre-commit install
```

### 4. Quality checks

Run the common quality commands with:

```bash
make format
make lint
make check
```

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

From the interface, you can:

* upload an audio file (`.wav` or `.mp3`)
* run Demucs separation
* preview and download the resulting stems

---

## 🧠 Notes

* The application runs Demucs from the active Python environment.
* Output files are stored in a local workspace directory (`.streamlit_workdir/`).
* NumPy is pinned to `<2` for compatibility with PyTorch / torchaudio.
* For portability, the project pins `torchaudio` below `2.9` on non-Intel platforms.

---

## 📜 License

MIT
