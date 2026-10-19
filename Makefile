PYTHON ?= python3
ARGS ?=

.DEFAULT_GOAL := test
.PHONY: help test install clean

test:
	@$(PYTHON) scripts/run_tests.py

help:
	@echo "make test     - run the pytest suite (skips cleanly if deps are absent)"
	@echo "make install  - install runtime + test dependencies"
	@echo "make clean    - remove generated artefacts and caches"

install:
	@$(PYTHON) -m pip install -r requirements.txt

clean:
	@rm -rf outputs .pytest_cache
	@find . -type d -name __pycache__ -prune -exec rm -rf {} +
