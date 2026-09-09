PYTHON ?= python

.PHONY: install install-dev doctor lint format test check build clean

install:
	$(PYTHON) -m pip install -r environment/requirements-cu121.txt
	$(PYTHON) -m pip install --no-deps -e .

install-dev: install
	$(PYTHON) -m pip install -e ".[dev]"

doctor:
	dera doctor

lint:
	$(PYTHON) -m ruff check dera third_party tests configs
	$(PYTHON) -m ruff format --check dera third_party tests configs

format:
	$(PYTHON) -m ruff check --fix dera third_party tests configs
	$(PYTHON) -m ruff format dera third_party tests configs

test:
	$(PYTHON) -m pytest -q

check: lint test

build:
	$(PYTHON) -m build

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('build', 'dist', 'dera_xray.egg-info', '.pytest_cache', '.ruff_cache')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
