"""Compatibility entry point: `python __main__.py` runs the screener.

The implementation lives in `screener.py` so the installed `stock-agent` console
script can import it from any working directory.
"""
from screener import run

if __name__ == "__main__":
    run()
