# Makefile for freecaykes/screener
# Usage:
#   make build     - create a virtualenv and install dependencies into it
#   make run       - run the screener (python __main__.py) using the venv
#   make clean     - delete the virtualenv (fully recreatable)
#   make rebuild   - clean + build in one step
#   make shell     - drop into a shell with the venv activated
#   make freeze    - show installed packages in the venv
#
# Docker (long-running service, state kept in the screener-data volume):
#   make docker-build   - build the image
#   make docker-run     - run in the foreground (Ctrl-C stops it)
#   make docker-up      - run detached with restart-on-failure
#   make docker-logs    - follow the detached container's logs
#   make docker-down    - stop and remove the container
#   make docker-shell   - debug shell inside the image
#   make docker-clean   - remove container + image (keeps the data volume)
#   make docker-purge   - also delete the data volume (model + logs)
#   make docker-window  - run now, auto-stop at STOP_AT (default 15:00)
#   make docker-cron    - show the weekday 05:00-15:00 crontab line
#   make docker-cron-install / docker-cron-remove - manage that crontab entry
#
# The container loads .env via --env-file and mounts config/config.toml
# read-only at /config/config.toml (SCREENER_CONFIG). Override with:
#   make docker-run ENV_FILE=other.env CONFIG_FILE=other.toml

# ---- config -----------------------------------------------------------
PYTHON      ?= python3
VENV_DIR    := .venv
VENV_BIN    := $(VENV_DIR)/bin
VENV_PYTHON := $(VENV_BIN)/python
VENV_PIP    := $(VENV_BIN)/pip

# marker file used to know the venv/deps are up to date
STAMP := $(VENV_DIR)/.install.stamp

# ---- docker config ----------------------------------------------------
DOCKER           ?= docker
DOCKER_IMAGE     ?= screener:latest
DOCKER_CONTAINER ?= screener
DOCKER_VOLUME    ?= screener-data
ENV_FILE         ?= .env
CONFIG_FILE      ?= config/config.toml
STOP_AT          ?= 15:00
CRON_LOG         ?= screener-cron.log

# .env is optional (same behaviour as `make run`); config.toml is required.
DOCKER_ENV_ARGS    = $(if $(wildcard $(ENV_FILE)),--env-file $(ENV_FILE),)
DOCKER_CONFIG_ARGS = -v "$(abspath $(CONFIG_FILE)):/config/config.toml:ro" \
                     -e SCREENER_CONFIG=/config/config.toml
DOCKER_STATE_ARGS  = -v $(DOCKER_VOLUME):/data
DOCKER_COMMON      = --init $(DOCKER_ENV_ARGS) $(DOCKER_CONFIG_ARGS) $(DOCKER_STATE_ARGS)

.PHONY: all build run clean rebuild shell freeze venv \
        docker-build docker-check-config docker-run docker-up docker-logs \
        docker-down docker-shell docker-clean docker-purge docker-window \
        docker-cron docker-cron-install docker-cron-remove

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

# ---- docker ----------------------------------------------------------------
# State (setup_bundle_v1.pkl, signal_log.jsonl, decision_params.json) is written
# to the container's cwd, /data, which is backed by the DOCKER_VOLUME volume.
docker-build:
	$(DOCKER) build -t $(DOCKER_IMAGE) .

docker-check-config:
	@test -f "$(CONFIG_FILE)" || { \
		echo "error: $(CONFIG_FILE) not found; set CONFIG_FILE=/path/to/config.toml" >&2; \
		exit 1; \
	}

docker-run: docker-build docker-check-config
	$(DOCKER) run --rm -it --name $(DOCKER_CONTAINER) $(DOCKER_COMMON) $(DOCKER_IMAGE)

docker-up: docker-build docker-check-config
	-$(DOCKER) rm -f $(DOCKER_CONTAINER)
	$(DOCKER) run -d --name $(DOCKER_CONTAINER) --restart unless-stopped \
		$(DOCKER_COMMON) $(DOCKER_IMAGE)
	@echo "Started $(DOCKER_CONTAINER); follow logs with: make docker-logs"

docker-logs:
	$(DOCKER) logs -f $(DOCKER_CONTAINER)

docker-down:
	-$(DOCKER) rm -f $(DOCKER_CONTAINER)

docker-shell: docker-build docker-check-config
	$(DOCKER) run --rm -it --entrypoint /bin/sh $(DOCKER_COMMON) $(DOCKER_IMAGE)

docker-clean:
	-$(DOCKER) rm -f $(DOCKER_CONTAINER)
	-$(DOCKER) rmi $(DOCKER_IMAGE)

# Destructive: removes the model cache, signal log and learned parameters too.
docker-purge: docker-clean
	-$(DOCKER) volume rm $(DOCKER_VOLUME)

# Run for a bounded window now (weekdays 05:00-15:00 when driven by cron).
# The script builds the image only if it is missing, then stops the container
# at STOP_AT.
docker-window: docker-check-config
	STOP_AT=$(STOP_AT) ./scripts/docker-window.sh

# Weekdays only (Mon-Fri). Times use the host's local timezone unless the
# crontab sets CRON_TZ/TZ. The script stops the container at STOP_AT.
docker-cron:
	@echo "Add this line with 'crontab -e' (weekdays, host local time):"
	@echo "0 5 * * 1-5 STOP_AT=$(STOP_AT) '$(abspath scripts/docker-window.sh)' >> '$(abspath $(CRON_LOG))' 2>&1"

docker-cron-install:
	@( crontab -l 2>/dev/null | grep -vF 'scripts/docker-window.sh' || true; \
	   echo "0 5 * * 1-5 STOP_AT=$(STOP_AT) '$(abspath scripts/docker-window.sh)' >> '$(abspath $(CRON_LOG))' 2>&1" ) | crontab -
	@echo "Installed: weekdays 05:00-$(STOP_AT), logging to $(abspath $(CRON_LOG))"

docker-cron-remove:
	@( crontab -l 2>/dev/null | grep -vF 'scripts/docker-window.sh' || true ) | crontab -
	@echo "Removed the screener window entry from crontab"

# ---- teardown -------------------------------------------------------------
clean:
	rm -rf $(VENV_DIR)

rebuild: clean build