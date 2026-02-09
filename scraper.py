from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import urljoin

import requests
import yaml
from bs4 import BeautifulSoup

# Limit in-session image fetches (same browser as page load)
_MAX_SESSION_IMAGES = 50
_SESSION_IMAGE_TIMEOUT_MS = 15000


@dataclass(frozen=True)
class ScrapedItem:
    key: str
    title: str
    text: str
    image_urls: tuple[str, ...]
    source_url: str
    timestamp: str = ""  # ISO 8601 datetime string


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _normalize_ws(s: str) -> str:
    return " ".join((s or "").split())


def _hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()[:24]


def _extract_text(soup: BeautifulSoup, selector: str) -> str:
    if not selector:
        return ""
    chunks: list[str] = []
    for el in soup.select(selector):
        t = el.get_text(" ", strip=True)
        t = _normalize_ws(t)
        if t:
            chunks.append(t)
    return _normalize_ws("\n".join(chunks))


def _extract_first_text(item: BeautifulSoup, selector: str) -> str:
    if not selector:
        return ""
    el = item.select_one(selector)
    if not el:
        return ""
    return _normalize_ws(el.get_text(" ", strip=True))


def _extract_background_image_url(style: str) -> str | None:
    """Extract URL from background-image CSS property."""
    if not style:
        return None
    match = re.search(r"background-image:\s*url\(['\"]?([^'\")\s]+)['\"]?\)", style, re.I)
    return match.group(1) if match else None


def _extract_images(item: BeautifulSoup, selector: str, base_url: str) -> tuple[str, ...]:
    if not selector:
        return ()
    urls: list[str] = []
    for el in item.select(selector):
        # Try src attribute first (for img tags)
        src = (el.get("src") or "").strip()
        if src:
            urls.append(urljoin(base_url, src))
            continue
        # Try background-image in style attribute (for Telegram photo wraps)
        style = el.get("style") or ""
        bg_url = _extract_background_image_url(style)
        if bg_url:
            urls.append(urljoin(base_url, bg_url))
    # de-dupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return tuple(out)


def _fetch_html_requests(url: str, headers: Optional[dict] = None, timeout_s: int = 30) -> str:
    r = requests.get(url, headers=headers or {}, timeout=timeout_s)
    r.raise_for_status()
    return r.text


def _fetch_html_playwright(url: str, wait_ms: int = 1500, user_data_dir: Optional[str] = None) -> str:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        if user_data_dir:
            context = p.chromium.launch_persistent_context(user_data_dir, headless=True)
            page = context.pages[0] if context.pages else context.new_page()
        else:
            browser = p.chromium.launch(headless=True)
            context = browser
            page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded")
        if wait_ms:
            page.wait_for_timeout(wait_ms)
        html = page.content()
        context.close()
        return html


def _parse_html_to_items(html: str, source: dict, url: str) -> list[ScrapedItem]:
    soup = BeautifulSoup(html, "lxml")
    item_selector = source.get("item_selector") or "article, .post, .entry"
    title_selector = source.get("title_selector") or "h1, h2, h3"
    body_selector = source.get("body_selector") or "p"
    image_selector = source.get("image_selector") or "img"

    items: list[ScrapedItem] = []
    for item in soup.select(item_selector):
        title = _extract_first_text(item, title_selector)
        text = _extract_text(item, body_selector)
        imgs = _extract_images(item, image_selector, base_url=url)
        if not title and not text and not imgs:
            continue

        # Extract message-specific URL from data-post attribute (Telegram)
        item_url = url
        data_post = item.get("data-post")
        if data_post:
            # data-post is like "channel/12345" -> build full URL
            item_url = f"https://t.me/{data_post}"

        # Extract timestamp from <time datetime="..."> element
        timestamp = ""
        time_el = item.select_one("time[datetime]")
        if time_el:
            timestamp = time_el.get("datetime", "")

        key = _hash_key(title, text[:200], imgs[0] if imgs else "", item_url)
        items.append(
            ScrapedItem(key=key, title=title, text=text, image_urls=imgs, source_url=item_url, timestamp=timestamp)
        )
    if not items:
        page_title = _extract_first_text(soup, "title") or _extract_first_text(soup, "h1, h2, h3")
        page_text = _extract_text(soup, body_selector) or _extract_text(soup, "body")
        page_imgs = _extract_images(soup, image_selector, base_url=url)
        if page_title or page_text or page_imgs:
            key = _hash_key(page_title, page_text[:200], page_imgs[0] if page_imgs else "", url)
            items.append(
                ScrapedItem(
                    key=key,
                    title=page_title,
                    text=page_text,
                    image_urls=page_imgs,
                    source_url=url,
                )
            )
    return items


def scrape_once(cfg: dict) -> list[ScrapedItem]:
    source = cfg.get("source", {}) or {}
    url = source.get("url", "")
    if not url:
        raise ValueError("config.yml: source.url is required")

    use_browser = bool(source.get("use_browser", False))
    user_data_dir = source.get("browser_user_data_dir") or None
    if use_browser and user_data_dir:
        items, _ = scrape_with_browser_session(cfg)
        return items
    html = (
        _fetch_html_playwright(
            url,
            wait_ms=int(source.get("browser_wait_ms", 1500)),
            user_data_dir=user_data_dir if use_browser else None,
        )
        if use_browser
        else _fetch_html_requests(url, headers=source.get("headers") or {})
    )
    return _parse_html_to_items(html, source, url)


def scrape_with_browser_session(cfg: dict) -> tuple[list[ScrapedItem], dict[str, bytes]]:
    """Scrape with a persistent browser profile (logged-in session) and fetch images in the same session.
    Use browser_user_data_dir in config so you can log in once (e.g. web.telegram.org or t.me) in that browser.
    Returns (items, image_bytes_by_url). image_bytes is empty if not using a session."""
    source = cfg.get("source", {}) or {}
    url = source.get("url", "")
    if not url:
        raise ValueError("config.yml: source.url is required")
    use_browser = bool(source.get("use_browser", False))
    user_data_dir = (source.get("browser_user_data_dir") or "").strip()
    if not use_browser or not user_data_dir:
        return scrape_once(cfg), {}

    from playwright.sync_api import sync_playwright

    wait_ms = int(source.get("browser_wait_ms", 1500))
    image_bytes: dict[str, bytes] = {}
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(user_data_dir, headless=True)
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(url, wait_until="domcontentloaded")
        if wait_ms:
            page.wait_for_timeout(wait_ms)
        html = page.content()
        items = _parse_html_to_items(html, source, url)
        seen: set[str] = set()
        n = 0
        for item in items:
            for img_url in item.image_urls:
                if n >= _MAX_SESSION_IMAGES or img_url in seen:
                    continue
                seen.add(img_url)
                try:
                    resp = page.goto(img_url, wait_until="commit", timeout=_SESSION_IMAGE_TIMEOUT_MS)
                    if resp and resp.ok:
                        body = resp.body()
                        if body:
                            image_bytes[img_url] = body
                            n += 1
                except Exception:
                    pass
        context.close()
    return items, image_bytes


def poll(cfg: dict) -> Iterable[list[ScrapedItem]]:
    interval = int(((cfg.get("refresh") or {}).get("interval_seconds")) or 10)
    while True:
        yield scrape_once(cfg)
        time.sleep(max(1, interval))

