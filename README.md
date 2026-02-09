## Live Scraper Viewer

This is a small **scraper + live reader window**:

- Scrapes **text + images** from a page (forum thread, blog, etc.)
- **Dedupes** items it has already shown
- Refreshes on a timer and renders in a clean UI
- Uses **Playwright** automatically if the page needs JavaScript to render

### Important notes

- **Respect the site’s Terms of Service and robots.txt**, and only scrape content you’re allowed to access.
- Keep refresh rates reasonable to avoid overloading sites.

### Setup (Windows PowerShell)

From this folder:

```bash
cd "C:\Users\chaot\scraper_viewer"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
python -m playwright install chromium
```

### Configure

Edit `config.yml`. The defaults try to work on “generic” pages, but forums usually need custom selectors.

### Run the live viewer

```bash
cd "C:\Users\chaot\scraper_viewer"
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

---


## Telegram Viewer (two-panel GUI + translation, Russian → English)

Same layout as the Weibo viewer, optimized for **Telegram** public channels: split-panel GUI with **raw content** on the left and **English translation** on the right. Focused on **Russian-language** content; translation is Russian → English. The left panel is **collapsible**.

Uses the **t.me/s/channel** public preview page (no login). Enter a channel name or URL; the app fetches the preview and translates the text.

### Run the Telegram viewer (from source)

```bash
cd "C:\Users\chaot\scraper_viewer"
.\.venv\Scripts\Activate.ps1
python run_telegram_viewer.py
```

Or double-click `run_telegram_viewer.py`.

- **Left panel:** Enter a Telegram channel or URL (e.g. `@durov`, `durov`, or `t.me/s/durov`), click **Fetch**. Raw scraped content appears below. Use **Hide original** to collapse the left panel; **Show original** to bring it back.
- **Right panel:** Same content translated to English. Translation is tried in this order:
  1. **Local MarianMT** (offline) — if you have `transformers` and `torch` installed.
  2. **Local Ollama** — if Ollama is running (e.g. `ollama run gemma2`).
  3. **Online** (Google / MyMemory via `deep-translator`) — fallback if local options aren’t available or fail.

Config: `config_telegram.yml`.

### Local translation (no internet)

- **Option A — Ollama (local LLM):** Install [Ollama](https://ollama.com), then run a model. The viewer tries **Gemma** first, then fallbacks:
  ```bash
  ollama run gemma2
  ```
  Other sizes: `ollama run gemma2:2b` (1.6GB), `ollama run gemma2:27b` (16GB). If Gemma isn’t installed, it will try `llama3.2` etc.
- **Option B — MarianMT (offline model):** `pip install transformers torch` (first run will download the model). The viewer will use Helsinki-NLP/opus-mt-ru-en for translation with no external service.

### Logged-in client (images that need your session)

If images don’t load (e.g. Telegram CDN requires auth), you can plug in a **persistent browser profile** where you’re already logged in:

1. In `config_telegram.yml`, set under `source`:
   ```yaml
   browser_user_data_dir: "./telegram_browser_profile"
   ```
2. **One-time login:** run the helper so a visible browser opens and you can log in:
   ```bash
   python telegram_login_browser.py
   ```
   Log in at [web.telegram.org](https://web.telegram.org) in that window, then press Enter in the terminal to close.
3. Next runs: when you fetch a channel in the viewer, it uses that profile; scraping and image loading happen in the same session, so images that require your login will load.

### Build Telegram Viewer .exe

```bash
pyinstaller telegram_viewer.spec
# Run: dist\TelegramViewer.exe
```

Same Playwright/Chromium requirement: run `python -m playwright install chromium` once per machine if needed.

