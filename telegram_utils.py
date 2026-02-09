"""Telegram-specific helpers: URL normalization and config for public channel preview (t.me/s)."""
from __future__ import annotations

import re
from pathlib import Path

from scraper import load_config, scrape_once, scrape_with_browser_session
from scraper import ScrapedItem

TELEGRAM_PREVIEW_BASE = "https://t.me/s"
CONFIG_TELEGRAM = "config_telegram.yml"


def telegram_url_from_input(raw: str) -> str:
    """Turn user input into a Telegram public channel preview URL.

    - '@channel' or 'channel' -> https://t.me/s/channel
    - 't.me/channel' or 't.me/s/channel' -> https://t.me/s/channel
    - Full URL with t.me -> normalized to t.me/s/ form for preview
    - Preserves query parameters like ?before=123 for pagination
    """
    s = (raw or "").strip()
    if not s:
        return ""

    # Extract and preserve query string if present
    query = ""
    if "?" in s:
        s, query = s.split("?", 1)
        query = "?" + query

    s = re.sub(r"^@", "", s)
    s = s.strip()
    if re.match(r"^https?://", s, re.I):
        # Normalize to t.me/s/channelname
        m = re.search(r"t\.me/s/([a-zA-Z0-9_]+)", s, re.I)
        if m:
            return f"{TELEGRAM_PREVIEW_BASE}/{m.group(1)}{query}"
        m = re.search(r"t\.me/([a-zA-Z0-9_]+)", s, re.I)
        if m:
            return f"{TELEGRAM_PREVIEW_BASE}/{m.group(1)}{query}"
        if "t.me" not in s.lower():
            return s + query
        return s.rstrip("/") + query
    # Plain channel name
    if "/" in s or " " in s:
        s = s.split("/")[-1].split()[0]
    return f"{TELEGRAM_PREVIEW_BASE}/{s}{query}"


def load_telegram_config(config_path: str | None = None) -> dict:
    """Load Telegram config, defaulting to config_telegram.yml next to this file."""
    path = config_path or str(Path(__file__).parent / CONFIG_TELEGRAM)
    cfg = load_config(path)
    cfg.setdefault("source", {})
    cfg.setdefault("refresh", {})
    cfg.setdefault("display", {})
    return cfg


def scrape_telegram(
    url_or_channel: str, config_path: str | None = None
) -> tuple[list[ScrapedItem], dict[str, bytes]]:
    """Scrape a Telegram public channel preview (t.me/s/...). Uses Playwright.
    When config has browser_user_data_dir (persistent profile), fetches images in the same
    browser session so a logged-in client can load images that require auth.
    Returns (items, image_bytes_by_url). image_bytes is empty when not using a session."""
    url = telegram_url_from_input(url_or_channel)
    if not url:
        raise ValueError("No URL or Telegram channel provided")
    cfg = load_telegram_config(config_path)
    cfg["source"] = {**cfg.get("source", {}), "url": url}
    source = cfg.get("source", {}) or {}
    if source.get("browser_user_data_dir") and source.get("use_browser"):
        return scrape_with_browser_session(cfg)
    return scrape_once(cfg), {}
