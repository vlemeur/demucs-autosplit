#!/bin/bash

set -e

echo "🔧 Installing TensorFlow extras via uv-compatible method..."

if [ ! -d ".venv" ]; then
  echo "❌ .venv not found. Run 'uv venv' first."
  exit 1
fi

source .venv/bin/activate

# Install only the optional dependencies
uv pip install ".[tensorflow]"

echo "✅ TensorFlow installed via optional dependencies."
