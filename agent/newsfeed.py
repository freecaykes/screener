"""
agent/newsfeed.py -- yfinance news access that doesn't fail silently or crash on odd shapes.

yfinance news items are not uniform: newer versions nest the fields under a
"content" dict, others are flat, and individual fields (summary, description)
are often None. Everything here tolerates all of that and never raises on
missing fields.
"""
from __future__ import annotations

import yfinance as yf


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def normalize(item) -> dict:
    """{'title': str, 'summary': str} from either yfinance shape. Missing fields become ''."""
    if not isinstance(item, dict):
        return {"title": "", "summary": ""}
    body = item.get("content")
    if not isinstance(body, dict):  # flat shape (or content is None)
        body = item
    return {
        "title": _text(body.get("title")),
        "summary": _text(body.get("summary")) or _text(body.get("description")),
    }


def fetch_news(ticker: str, limit: int = 5) -> list:
    """Raw yfinance news items (possibly empty). Blocking: call via asyncio.to_thread."""
    print()
    return list(yf.Ticker(ticker).news or [])[:limit]