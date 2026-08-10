# Justfile
# Modern alternative to Makefile for demucs-autosplit

# Variables for venv and tools
venv_python := ".venv/bin/python"
ruff := "{{venv_python}} -m ruff"
pre_commit := "{{venv_python}} -m pre_commit"

# Check code style and formatting (without applying fixes)
lint:
    #!/usr/bin/env bash
    set -euxo pipefail
    {{ruff}} check .
    {{ruff}} format --check .

# Apply code style and formatting fixes
format:
    #!/usr/bin/env bash
    set -euxo pipefail
    {{ruff}} check . --fix
    {{ruff}} format .

# Run pre-commit hooks on all files
check:
    #!/usr/bin/env bash
    set -euxo pipefail
    {{pre_commit}} run --all-files
