"""Typed configuration loaded from a TOML file.

Usage (run from the repo root, like __main__.py does):
    from config import load_config

    cfg = load_config()                      # ./config.toml next to this file
    cfg = load_config("other.toml")          # explicit path
    # or: SCREENER_CONFIG=/path/to.toml python __main__.py

    agent = TickerAgent(cfg.llm.model, cfg.llm.temperature)
    for t in cfg.general.tickers: ...

Secrets are never read from the TOML. Put API keys in .env (the VS Code
launch config loads it) and use `cfg.llm.api_key` to read the right one.
"""
from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

if sys.version_info >= (3, 11):
    import tomllib
else:  # pyproject allows Python 3.10: `pip install tomli`
    import tomli as tomllib

try:  # optional: also honour .env when running outside the debugger
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"
ENV_VAR = "SCREENER_CONFIG"

T = TypeVar("T")

DEFAULT_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
}


@dataclass(frozen=True)
class GeneralConfig:
    tickers: list[str] = field(default_factory=lambda: ["NOW"])

    def __post_init__(self) -> None:
        if not self.tickers or not all(isinstance(t, str) and t for t in self.tickers):
            raise ValueError("general.tickers must be a non-empty list of strings")


@dataclass(frozen=True)
class RuntimeConfig:
    news_poll_interval_sec: int = 300
    news_fetch_timeout_sec: int = 30
    news_limit: int = 5
    max_concurrent: int = 5

    def __post_init__(self) -> None:
        for name in ("news_poll_interval_sec", "news_fetch_timeout_sec", "news_limit", "max_concurrent"):
            if getattr(self, name) < 1:
                raise ValueError(f"runtime.{name} must be >= 1")


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "google"
    model: str = "gemini-3.6-flash"
    temperature: float = 0.0
    api_key_env: str = ""

    def __post_init__(self) -> None:
        if self.provider not in DEFAULT_KEY_ENV:
            raise ValueError(f"llm.provider must be one of {sorted(DEFAULT_KEY_ENV)}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("llm.temperature must be between 0 and 2")

    @property
    def api_key_var(self) -> str:
        return self.api_key_env or DEFAULT_KEY_ENV[self.provider]

    @property
    def api_key(self) -> str:
        """Read the API key from the environment (.env), never from the TOML."""
        value = os.environ.get(self.api_key_var)
        if not value:
            raise RuntimeError(f"Environment variable {self.api_key_var} is not set")
        return value


@dataclass(frozen=True)
class IndicatorsConfig:
    history_period: str = "60d"
    ema_length: int = 20
    rsi_length: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bbands_length: int = 20
    bbands_std: float = 2.0

    def __post_init__(self) -> None:
        if self.macd_fast >= self.macd_slow:
            raise ValueError("indicators.macd_fast must be smaller than macd_slow")

    @property
    def macd_col(self) -> str:
        return f"MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"

    @property
    def macd_signal_col(self) -> str:
        return f"MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"

    @property
    def bb_mid_col(self) -> str:
        return f"BBM_{self.bbands_length}_{float(self.bbands_std)}"

    @property
    def bb_bandwidth_col(self) -> str:
        return f"BBB_{self.bbands_length}_{float(self.bbands_std)}"


@dataclass(frozen=True)
class XGBoostConfig:
    history_period: str = "2y"
    min_rows: int = 100
    model_path: str = "xgboost_stock_delta_model.pkl"
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if not 0 < self.subsample <= 1 or not 0 < self.colsample_bytree <= 1:
            raise ValueError("xgboost.subsample / colsample_bytree must be in (0, 1]")

    def model_params(self) -> dict[str, Any]:
        """Keyword args ready for `XGBRegressor(**cfg.xgboost.model_params())`."""
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }


@dataclass(frozen=True)
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    indicators: IndicatorsConfig = field(default_factory=IndicatorsConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)


def _build(cls: type[T], raw: dict[str, Any], path: str = "") -> T:
    """Recursively build a dataclass from a dict, rejecting unknown keys."""
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise KeyError(f"Unknown config key(s) in [{path or cls.__name__}]: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        ftype = hints[name]
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _build(ftype, value, f"{path}.{name}" if path else name)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Read a TOML file and return a validated, immutable Config."""
    config_path = Path(path or os.environ.get(ENV_VAR) or DEFAULT_CONFIG_PATH)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)
    
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(level=loglevel, format="%(asctime)s %(message)s")

    return _build(Config, raw)


if __name__ == "__main__":
    logging.info(load_config())