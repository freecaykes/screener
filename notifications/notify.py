import requests

from config import get_config
import urllib.parse

cfg = get_config()

def notify(msg, **kw):
    try:
        _unsafe_notify(msg, **kw)
    except requests.RequestException as e:
        print(f"Notification failed: {e}")

def _unsafe_notify(message, title="Python app", priority="default"):
    url = urljoin(cfg.general.notification_host, cfg.general.notification_topic)
    requests.post(
        url,
        data=message.encode("utf-8"),
        headers={"Title": title, "Priority": priority},  # priority: min, low, default, high, urgent
        timeout=10,
    )
