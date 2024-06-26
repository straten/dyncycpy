
PYTHON_SRC = cycfista.py pycyc.py

PYTHON_RUNNER ?=

PYTHON_LINE_LENGTH ?= 110

PYTHON_SWITCHES_FOR_BLACK ?= --line-length $(PYTHON_LINE_LENGTH) --exclude .+\.ipynb
PYTHON_SWITCHES_FOR_ISORT ?= --profile black --line-length $(PYTHON_LINE_LENGTH)
PYTHON_SWITCHES_FOR_PYLINT ?= --max-line-length $(PYTHON_LINE_LENGTH)
PYTHON_SWITCHES_FOR_FLAKE8 ?= --show-source --statistics --max-line-length $(PYTHON_LINE_LENGTH)
PYTHON_SWITCHES_FOR_AUTOFLAKE ?= --in-place --remove-unused-variables --remove-all-unused-imports --recursive --ignore-init-module-imports

PYTHON_LINT_TARGET ?= $(PYTHON_SRC)

all: python-pre-lint python-lint

python-lint-fix:
	$(PYTHON_RUNNER) isort $(PYTHON_SWITCHES_FOR_ISORT) $(PYTHON_LINT_TARGET)
	$(PYTHON_RUNNER) black $(PYTHON_SWITCHES_FOR_BLACK) $(PYTHON_LINT_TARGET)

python-autoflake:
	$(PYTHON_RUNNER) autoflake $(PYTHON_SWITCHES_FOR_AUTOFLAKE) $(PYTHON_LINT_TARGET)

python-pre-lint: python-lint-fix python-autoflake

flake8:
	$(PYTHON_RUNNER) flake8 $(PYTHON_SWITCHES_FOR_FLAKE8) $(PYTHON_LINT_TARGET)

python-format:
	$(PYTHON_RUNNER) isort $(PYTHON_SWITCHES_FOR_ISORT) $(PYTHON_LINT_TARGET)
	$(PYTHON_RUNNER) black $(PYTHON_SWITCHES_FOR_BLACK) $(PYTHON_LINT_TARGET)

python-lint:
	$(PYTHON_RUNNER) isort --check-only $(PYTHON_SWITCHES_FOR_ISORT) $(PYTHON_LINT_TARGET)
	$(PYTHON_RUNNER) black --check $(PYTHON_SWITCHES_FOR_BLACK) $(PYTHON_LINT_TARGET)
	-$(PYTHON_RUNNER) flake8 $(PYTHON_SWITCHES_FOR_FLAKE8) $(PYTHON_LINT_TARGET)
	-$(PYTHON_RUNNER) pylint $(PYTHON_SWITCHES_FOR_PYLINT) $(PYTHON_LINT_TARGET)
