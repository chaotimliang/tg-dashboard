"""
One-time helper: open a persistent browser profile so you can log in to Telegram.
After logging in, close the browser. Then the Telegram viewer will use this profile
when browser_user_data_dir is set in config_telegram.yml, and images will load with your session.

Usage:
  python telegram_login_browser.py
  (or set TELEGRAM_PROFILE env to a path, default: ./telegram_browser_profile)
"""
from __future__ import annotations

import os
from pathlib import Path

def main() -> None:
    profile = os.environ.get("TELEGRAM_PROFILE") or str(Path(__file__).parent / "telegram_browser_profile")
    Path(profile).mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(profile, headless=False)
        page = context.pages[0] if context.pages else context.new_page()
        page.goto("https://web.telegram.org/k/", wait_until="domcontentloaded")
        input("Log in in the browser window, then press Enter here to close…")
        context.close()

if __name__ == "__main__":
    main()
