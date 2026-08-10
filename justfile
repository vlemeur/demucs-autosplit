# justfile
# Modern alternative to Makefile for demucs-autosplit

set default-list := true

# Variables for venv and tools
PYTHON := ".venv/bin/python"
PREK := ".venv/bin/prek"

# Check code style and formatting (without applying fixes)
lint:
    {{PYTHON}} -m ruff check .
    {{PYTHON}} -m ruff format --check .

# Apply code style and formatting fixes
format:
    {{PYTHON}} -m ruff check . --fix
    {{PYTHON}} -m ruff format .

# Run prek hooks on all files
check:
    {{PREK}} run --all-files
