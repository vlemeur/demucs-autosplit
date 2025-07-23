#!/bin/bash

set -e

echo "🔧 Installing Madmom and TensorFlow via uv..."

if [ ! -d ".venv" ]; then
  echo "❌ .venv not found. Run 'uv venv' first."
  exit 1
fi

source .venv/bin/activate

# --- Detect platform ---
PLATFORM=$(uname)
echo "🔍 Detected platform: $PLATFORM"

# --- TensorFlow ---
echo "🧠 Installing TensorFlow..."

if [[ "$PLATFORM" == "Darwin" ]]; then
  echo "🍏 macOS detected – installing tensorflow-macos"
  uv pip install --prerelease allow tensorflow-macos tensorflow-hub
else
  echo "🐧 Linux/WSL detected – installing regular tensorflow"
  uv pip install tensorflow tensorflow-hub
fi

echo "✅ All extras installed successfully."
