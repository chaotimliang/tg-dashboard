# TG Dashboard

A custom Telegram channel viewer that uses a **local LLM** (via Ollama) to provide real-time translation, AI-assisted search, and automated analytics and insights from channel posts — all running offline on your machine.

---

## Features

- **Dual-panel viewer** — original (Russian) on the left, English translation on the right
- **Local translation** — Argos Translate, MarianMT, or Ollama (no internet required)
- **AI-assisted search** — uses a local vision model to find images relevant to your query (e.g. searching "Ka-52" finds images of the helicopter even if it isn't mentioned in the text)
- **Analytics panel** — extracts equipment losses, casualties, and operational intelligence from post text using a local LLM, displayed as charts
- **Grid view** — thumbnail gallery for browsing media-heavy channels
- **Live mode** — continuously polls a channel for new posts matching a search query
- **Media preview** — click any image to view full size with AI description and metadata

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with at least one model (e.g. `gemma2`, `llama3.2`)
- For vision/image analysis: a vision model installed (e.g. `ollama pull llava`)
- Telegram API credentials (free — see setup below)

---

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Telegram API credentials

1. Go to [my.telegram.org](https://my.telegram.org) and log in
2. Click **API development tools**
3. Create an app — you'll get an `api_id` and `api_hash`
4. Create `telegram_credentials.txt` in the project folder:
   ```
   api_id=YOUR_API_ID
   api_hash=YOUR_API_HASH
   ```

This file is excluded from git (`.gitignore`) — never commit it.

### Ollama models

Install [Ollama](https://ollama.com), then pull the models you want:

```bash
ollama pull gemma2          # text analysis and translation
ollama pull llava           # vision model for image analysis
```

---

<<<<<<< HEAD
## Usage

```bash
=======

## Telegram Viewer (two-panel GUI + translation, Russian → English)

Same layout as the Weibo viewer, optimized for **Telegram** public channels: split-panel GUI with **raw content** on the left and **English translation** on the right. Focused on **Russian-language** content; translation is Russian → English. The left panel is **collapsible**.

Uses the **t.me/s/channel** public preview page (no login). Enter a channel name or URL; the app fetches the preview and translates the text.

### Run the Telegram viewer (from source)

```bash
cd "C:\Users\chaot\scraper_viewer"
.\.venv\Scripts\Activate.ps1
>>>>>>> 2b19ad324f909cb088d9325d7562a23db9b37e88
python run_telegram_viewer.py
```

1. Enter a channel name (e.g. `okspn`) in the **Channel** field
2. Set a date range and click **Fetch**
3. Use the **Search** or **Live Search** bar to filter posts
4. Enable the **AI** checkbox on Live Search to find visually relevant images
5. Check **Analytics** in the filter bar to open the analytics panel, then click **Analyze**

---

## Analytics

The analytics panel uses the local LLM to scan loaded posts and extract:

- **Equipment losses** — type, category (tank, helicopter, UAV, etc.), side (Russian/Ukrainian), and outcome (destroyed/damaged/captured)
- **Casualties** — killed, wounded, captured by side
- **Visualizations** — bar chart by equipment category, pie chart by side

Results are displayed as charts in a collapsible right-side panel. No data leaves your machine.

---

## Translation

Translation is attempted in this order, falling back as needed:

1. **Argos Translate** — fast, fully offline
2. **MarianMT** — offline transformer model (`pip install transformers torch`)
3. **Ollama** — local LLM fallback
4. **Google Translate / MyMemory** — online fallback

---

## Project Structure

| File | Description |
|------|-------------|
| `viewer_gui_telegram.py` | Main application GUI |
| `telegram_client.py` | Telegram API client (Telethon) |
| `translate_text.py` | Multi-backend translation |
| `analytics_models.py` | Data structures for analytics |
| `analytics_extractor.py` | LLM-powered extraction logic |
| `analytics_panel.py` | Analytics UI panel with charts |
| `run_telegram_viewer.py` | Entry point |

