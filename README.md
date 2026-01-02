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
uv venv --python 3.11
source .venv/bin/activate
uv sync
uv pip install -e .
```

### 3. (Optional) Enable pre-commit hooks

```bash
uv run pre-commit install
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

* The application uses Demucs via the system command (`demucs` must be available in your environment).
* Output files are stored in a local workspace directory (`.streamlit_workdir/`).
* NumPy is pinned to `<2` for compatibility with PyTorch / torchaudio.

---

## 📜 License

MIT
