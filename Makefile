VENV_PYTHON := .venv/bin/python
RUFF := $(VENV_PYTHON) -m ruff
PRE_COMMIT := $(VENV_PYTHON) -m pre_commit

.PHONY: lint format check

lint:
	$(RUFF) check .
	$(RUFF) format --check .

format:
	$(RUFF) check . --fix
	$(RUFF) format .

check:
	$(PRE_COMMIT) run --all-files
