# justfile
# Modern alternative to Makefile for demucs-autosplit

set default-list := true

# Variables for venv and tools
PYTHON := ".venv/bin/python"
PREK := ".venv/bin/prek"
TY := ".venv/bin/ty"

# Check code style and formatting (without applying fixes)
lint:
    {{PYTHON}} -m ruff check .
    {{PYTHON}} -m ruff format --check .

# Apply code style and formatting fixes
format:
    {{PYTHON}} -m ruff check . --fix
    {{PYTHON}} -m ruff format .

# Run type checking with ty
type-check:
    {{TY}} check .

# Run prek hooks on all files
check:
    {{PREK}} run --all-files

# Run all quality checks (lint + type-check + pre-commit)
all:
    just lint
    just type-check
    just check

# Run the Streamlit application locally
run:
    {{PYTHON}} -m streamlit run src/ui.py
