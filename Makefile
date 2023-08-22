
PYTHON_SRC = cycfista.py pycyc.py

PYTHON_RUNNER ?=

PYTHON_LINE_LENGTH ?= 110

PYTHON_TEST_FOLDER_NBMAKE ?= .## Option folder for notebook tests

PYTHON_SWITCHES_FOR_BLACK ?=## Custom switches added to black

PYTHON_SWITCHES_FOR_ISORT ?=## Custom switches added to isort

PYTHON_SWITCHES_FOR_PYLINT ?=## Custom switches added to pylint for all python code

NOTEBOOK_SWITCHES_FOR_PYLINT ?=## Custom switches added to pylint for notebooks

PYTHON_SWITCHES_FOR_FLAKE8 ?=## Custom switches added to flake8 for all python code

NOTEBOOK_SWITCHES_FOR_FLAKE8 ?=## Custom switches added to flake8 for notebooks

PYTHON_LINT_TARGET ?= $(PYTHON_SRC) ## Paths containing python to be formatted and linted

PYTHON_SWITCHES_FOR_AUTOFLAKE ?= --in-place --remove-unused-variables --remove-all-unused-imports --recursive --ignore-init-module-imports

python-lint-fix:
	$(PYTHON_RUNNER) isort --profile black $(PYTHON_SWITCHES_FOR_ISORT) $(PYTHON_LINT_TARGET)
	$(PYTHON_RUNNER) black $(PYTHON_SWITCHES_FOR_BLACK) $(PYTHON_LINT_TARGET)

python-autoflake:
	$(PYTHON_RUNNER) autoflake $(PYTHON_SWITCHES_FOR_AUTOFLAKE) $(PYTHON_LINT_TARGET)

python-pre-lint: python-lint-fix python-autoflake

flake8:
	$(PYTHON_RUNNER) flake8 --show-source --statistics $(PYTHON_SWITCHES_FOR_FLAKE8) $(PYTHON_LINT_TARGET)

python-format:
	$(PYTHON_RUNNER) isort --profile black --line-length $(PYTHON_LINE_LENGTH) $(PYTHON_SWITCHES_FOR_ISORT) $(PYTHON_LINT_TARGET)
	$(PYTHON_RUNNER) black --exclude .+\.ipynb --line-length $(PYTHON_LINE_LENGTH) $(PYTHON_SWITCHES_FOR_BLACK) $(PYTHON_LINT_TARGET)

python-lint:
	@mkdir -p build/reports;
	$(PYTHON_RUNNER) isort --check-only --profile black --line-length $(PYTHON_LINE_LENGTH) $(PYTHON_SWITCHES_FOR_ISORT) $(PYTHON_LINT_TARGET)
	$(PYTHON_RUNNER) black --exclude .+\.ipynb --check --line-length $(PYTHON_LINE_LENGTH) $(PYTHON_SWITCHES_FOR_BLACK) $(PYTHON_LINT_TARGET)
	$(PYTHON_RUNNER) flake8 --show-source --statistics --max-line-length $(PYTHON_LINE_LENGTH) $(PYTHON_SWITCHES_FOR_FLAKE8) $(PYTHON_LINT_TARGET)
	$(PYTHON_RUNNER) pylint --output-format=parseable,parseable:build/code_analysis.stdout,pylint_junit.JUnitReporter:build/reports/linting-python.xml --max-line-length $(PYTHON_LINE_LENGTH) $(PYTHON_SWITCHES_FOR_PYLINT) $(PYTHON_LINT_TARGET)
	@make --no-print-directory join-lint-reports

