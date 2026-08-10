# justfile
# Modern alternative to Makefile for demucs-autosplit

set default-list := true

# Variables for venv and tools
PYTHON := ".venv/bin/python"

# Check code style and formatting (without applying fixes)
lint:
    {{PYTHON}} -m ruff check .
    {{PYTHON}} -m ruff format --check .

# Apply code style and formatting fixes
format:
    {{PYTHON}} -m ruff check . --fix
    {{PYTHON}} -m ruff format .

# Run pre-commit hooks on all files
check:
    {{PYTHON}} -m pre_commit run --all-files
