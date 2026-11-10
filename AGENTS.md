# AGENTS.md

Swing-setup screener: a long-running asyncio service that scans Yahoo Finance after the US close (16:10 ET) and polls for new headlines, detects one of four setups, estimates target-before-stop odds with a cached XGBoost bundle, then asks an LLM to read the news in context. There is no README and no test suite; module docstrings are the design notes.

## Commands

- `make run` — canonical run: creates `.venv`, `pip install -e .`, sources `.env`, then runs `python __main__.py`. Do not use it as a smoke test; it blocks forever and calls Yahoo/LLM APIs.
- `.venv/bin/stock-agent` — installed console script (calls `screener.run`); same long-running service.
- `make build` / `make clean` / `make rebuild` — venv lifecycle. The editable install exposes `agent`, `train`, `config`, `notifications`, `backtest`, and `screener`, so imports work from any cwd. Runtime state files are cwd-relative, so run the service from the repo root.
- `make docker-build` / `docker-run` / `docker-up` / `docker-down` — container lifecycle. The image runs from `/data` (the `screener-data` volume, where the model cache and logs persist), loads `.env` via `--env-file`, and bind-mounts `config/config.toml` read-only at `/config/config.toml` (`SCREENER_CONFIG`). Override with `ENV_FILE=` / `CONFIG_FILE=`. `docker-shell` opens a debug shell; `docker-purge` also deletes the volume.
- `make docker-window` — run now and stop at `STOP_AT` (default `15:00`). `make docker-cron-install` installs a weekday (Mon–Fri) `05:00` start entry that runs `scripts/docker-window.sh`; `make docker-cron-remove` removes it. Cron uses the host's local timezone. Note the app's daily post-close scan is 16:10 ET, so a window ending at 15:00 local only covers it when local time is at least ~1h10m behind ET.
- Tests: none exist yet. `pytest` is configured for `tests/`; dev tools are not installed by `make build`:
  - `.venv/bin/pip install -e '.[dev]'`
  - `.venv/bin/python -m pytest tests/test_x.py::test_name -v`
- Lint/format only the files you touch: `.venv/bin/ruff check <path>`, `.venv/bin/black <path>`, `.venv/bin/isort <path>` (line length 100). There is no CI or pre-commit, and the tree is not lint-clean (100+ existing ruff findings), so avoid repo-wide autofixes.

## Structure

- `screener.py` — startup trains/loads the model and refits the news weight, then starts the news producer (first poll fetches `runtime.news_limit` headlines, later polls fetch one; it queues a ticker only when its newest headline changes), daily scan, and worker consumers. `__main__.py` is a thin shim so `python __main__.py` keeps working.
- `agent/agent.py` — LangGraph pipeline: `fetch_news -> extract_headline -> check_headline_relevance -> (compute_indicators -> setup_probability -> news_analysis -> generate_signal | skip_irrelevant)`. News-triggered runs are gated by an LLM price-relevance check; daily `scan` runs skip the gate so technical setups still run.
- `train/features.py` + `train/setups.py` — the one feature/setup implementation shared by training and live inference.
- `train/setup_model.py` — dataset building, walk-forward validation, calibration, `predict_final`.
- `agent/decision.py` — pure `decide()`, plus the shadow log and news-weight refit. HOLD is normal: BUY/SELL requires out-of-sample validation and EV above `MIN_EV_R`.
- `config/config.py` + `config/config.toml` — frozen dataclasses; unknown keys raise; override the file with `SCREENER_CONFIG`.

## Gotchas

- `get_config()` is `lru_cache`d and called at import time in several modules; a TOML change needs a fresh process (or `cache_clear()` in tests).
- First run downloads ~10y per ticker and trains. Later runs reuse `setup_bundle_v1.pkl`; it is retrained when the feature list, strategy constants, ticker set, or benchmark mapping changes (`force_retrain=True` forces it).
- Only completed daily bars are used; `features.drop_incomplete_last_bar()` removes today's partial bar before 16:05 ET, so `as_of` is intentionally the prior session.
- yfinance calls are blocking: keep them in `asyncio.to_thread`, and keep network failures non-fatal (empty news / NaN features are expected).
- `backtest/` is legacy and broken (`Backtester.run` does not exist; the suite expects a missing parquet), so it is not a verification path.
