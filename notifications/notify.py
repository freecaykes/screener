from urllib.parse import urljoin
import requests
import logging

from config import get_config

cfg = get_config()

def notify(title: str, msg: str, **kw):
    try:
        _unsafe_notify(msg, title, **kw)
    except requests.RequestException as e:
        logging.error(f"Notification failed: {e}")

def _unsafe_notify(message, title="", priority="default"):
    url = urljoin(cfg.general.notification_host, cfg.general.notification_topic)
    requests.post(
        url,
        data=message.encode("utf-8"),
        headers={"Title": title, "Priority": priority},  # priority: min, low, default, high, urgent
        timeout=10,
    )
