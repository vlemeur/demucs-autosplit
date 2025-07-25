# 🎧 demucs-autosplit

> Batch stem separation tool powered by [Demucs](https://github.com/facebookresearch/demucs), written in Python and structured for clean automation.

This project automates the process of splitting audio files into isolated stems (vocals, drums, bass, etc.) using Demucs. It's designed for musicians, learners, or data pipeline use cases.

---

## 📦 Features

- 🔹 Batch process `.wav` and `.mp3` files from a folder

---

## 🚀 Getting started

### 1. Clone the repository

```bash
git clone git@github.com:vlemeur/demucs-autosplit.git
cd demucs-autosplit
```

### Set up the environment

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
uv pip install -e .
```


## 🎚️ Usage

### 1. Add your audio files

Place .wav or .mp3 files in the audio/ folder.

### 2. Run the batch separator

```bash
python scripts/batch_split.py
```

### 3. Run chord detection

```bash
python scripts/chords_detect.py
```
