# Makefile for freecaykes/screener
# Usage:
#   make build     - create a virtualenv and install dependencies into it
#   make run       - run the screener (python __main__.py) using the venv
#   make clean     - delete the virtualenv (fully recreatable)
#   make rebuild   - clean + build in one step
#   make shell     - drop into a shell with the venv activated
#   make freeze    - show installed packages in the venv

# ---- config -----------------------------------------------------------
PYTHON      ?= python3
VENV_DIR    := .venv
VENV_BIN    := $(VENV_DIR)/bin
VENV_PYTHON := $(VENV_BIN)/python
VENV_PIP    := $(VENV_BIN)/pip

# marker file used to know the venv/deps are up to date
STAMP := $(VENV_DIR)/.install.stamp

.PHONY: all build run clean rebuild shell freeze venv

all: build

# ---- create the virtualenv --------------------------------------------
$(VENV_DIR)/pyvenv.cfg:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip

venv: $(VENV_DIR)/pyvenv.cfg

# ---- install project + dependencies (editable, from pyproject.toml) ---
$(STAMP): $(VENV_DIR)/pyvenv.cfg pyproject.toml
	$(VENV_PIP) install -e .
	touch $(STAMP)

build: $(STAMP)
	@echo "Virtualenv ready at $(VENV_DIR)"

# ---- run the app --------------------------------------------------------
# Loads .env (if present) into the environment before running, without
# requiring a specific quoting style in the file.
run: build
	@if [ -f .env ]; then \
		set -a; . ./.env; set +a; \
		$(VENV_PYTHON) __main__.py; \
	else \
		$(VENV_PYTHON) __main__.py; \
	fi

# ---- convenience targets ------------------------------------------------
shell: build
	@echo "Run: source $(VENV_BIN)/activate"
	@$(SHELL) -c "source $(VENV_BIN)/activate && exec $$SHELL"

freeze: build
	$(VENV_PIP) freeze

# ---- teardown -------------------------------------------------------------
clean:
	rm -rf $(VENV_DIR)

rebuild: clean build