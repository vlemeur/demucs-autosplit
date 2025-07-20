#!/bin/bash

set -e

echo "🔧 Installing Madmom and TensorFlow via uv..."

if [ ! -d ".venv" ]; then
  echo "❌ .venv not found. Run 'uv venv' first."
  exit 1
fi

source .venv/bin/activate


# --- TensorFlow ---
echo "🧠 Installing TensorFlow (macOS)..."
uv pip install --prerelease allow tensorflow-macos tensorflow-hub

echo "✅ All extras installed successfully."
