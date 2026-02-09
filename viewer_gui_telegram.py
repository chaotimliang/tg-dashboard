"""
Telegram Viewer using Telethon API.
Features: date range queries, search, fast loading, media grid, download, translation.
"""
from __future__ import annotations

import concurrent.futures
import io
import os
import queue
import threading
import tkinter as tk
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tkinter import ttk, messagebox, font as tkfont, filedialog
from typing import Optional

from PIL import Image, ImageTk
from tkcalendar import DateEntry

from telegram_client import get_client, TelegramMessage, TelegramClientSync
from translate_text import to_english_from_russian
from analytics_panel import AnalyticsPanel

# Defaults
DEFAULT_CHANNEL = "okspn"
MAX_IMAGE_HEIGHT = 180
GRID_THUMB_SIZE = 200  # Larger thumbnails for better visibility
GRID_PADDING = 8
IMAGE_WORKERS = 6
AI_WORKERS = 4  # Parallel AI analysis workers (tune based on your GPU/Ollama capacity)
LIVE_POLL_INTERVAL_MS = 15000  # 15 seconds for live updates

# Translation cache
_translation_cache: dict[str, str] = {}
_reverse_translation_cache: dict[str, str] = {}  # English → Russian


def _translate_to_russian(text: str) -> str | None:
    """Translate English text to Russian for bilingual search."""
    if not text:
        return None
    if text in _reverse_translation_cache:
        return _reverse_translation_cache[text]

    try:
        # Try Argos Translate first (fast)
        try:
            import argostranslate.translate
            from translate_text import _ensure_argos_language

            if _ensure_argos_language("en", "ru"):
                installed = argostranslate.translate.get_installed_languages()
                from_lang = next((l for l in installed if l.code == "en"), None)
                to_lang = next((l for l in installed if l.code == "ru"), None)
                if from_lang and to_lang:
                    translation = from_lang.get_translation(to_lang)
                    if translation:
                        result = translation.translate(text)
                        if result:
                            _reverse_translation_cache[text] = result
                            return result
        except ImportError:
            pass

        # Fallback to Ollama
        try:
            import requests
            from translate_text import OLLAMA_BASE, OLLAMA_MODELS, OLLAMA_TIMEOUT

            prompt = f"Translate the following English text to Russian. Output only the translation, no explanation.\n\n{text}"
            for model in OLLAMA_MODELS:
                try:
                    r = requests.post(
                        f"{OLLAMA_BASE}/api/generate",
                        json={"model": model, "prompt": prompt, "stream": False},
                        timeout=OLLAMA_TIMEOUT,
                    )
                    r.raise_for_status()
                    out = r.json()
                    response = (out.get("response") or "").strip()
                    if response:
                        _reverse_translation_cache[text] = response
                        return response
                except Exception:
                    continue
        except Exception:
            pass
    except Exception:
        pass

    return None


def _bytes_to_photo(data: bytes, max_height: int = MAX_IMAGE_HEIGHT, is_video: bool = False) -> Optional[ImageTk.PhotoImage]:
    """Convert image bytes to Tk PhotoImage. Adds play icon overlay for videos."""
    try:
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        w, h = img.size
        if h > max_height:
            w = int(w * max_height / h)
            h = max_height
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            img = img.resize((w, h), resample)

        # Add play icon overlay for videos
        if is_video:
            img = _add_play_overlay(img)

        return ImageTk.PhotoImage(img)
    except Exception:
        return None


def _add_play_overlay(img: Image.Image) -> Image.Image:
    """Add a semi-transparent play button overlay to an image."""
    from PIL import ImageDraw

    # Create a copy to avoid modifying original
    img = img.copy()
    draw = ImageDraw.Draw(img, 'RGBA')

    w, h = img.size
    # Draw semi-transparent circle
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 6

    # Draw circle background
    draw.ellipse(
        [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
        fill=(0, 0, 0, 128)
    )

    # Draw play triangle
    triangle_size = radius * 0.6
    points = [
        (center_x - triangle_size * 0.4, center_y - triangle_size),
        (center_x - triangle_size * 0.4, center_y + triangle_size),
        (center_x + triangle_size * 0.8, center_y),
    ]
    draw.polygon(points, fill=(255, 255, 255, 200))

    return img


def _bytes_to_thumbnail(data: bytes, size: int = GRID_THUMB_SIZE, is_video: bool = False) -> Optional[ImageTk.PhotoImage]:
    """Convert image bytes to square thumbnail for grid view."""
    try:
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        # Crop to square
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        # Resize
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        img = img.resize((size, size), resample)

        # Add play icon overlay for videos
        if is_video:
            img = _add_play_overlay(img)

        return ImageTk.PhotoImage(img)
    except Exception:
        return None


def _save_image(data: bytes, path: Path, msg_id: int, idx: int = 0) -> bool:
    """Save image bytes to file. Returns True on success."""
    try:
        img = Image.open(io.BytesIO(data))
        ext = "jpg" if img.format in ("JPEG", None) else img.format.lower()
        filename = path / f"msg_{msg_id}_{idx}.{ext}"
        img.save(filename)
        return True
    except Exception:
        return False


def _bytes_to_full_image(data: bytes, max_width: int = 1200, max_height: int = 800, is_video: bool = False) -> Optional[ImageTk.PhotoImage]:
    """Convert image bytes to full-size Tk PhotoImage for preview."""
    try:
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        w, h = img.size

        # Scale to fit within max dimensions while preserving aspect ratio
        scale = min(max_width / w, max_height / h, 1.0)
        if scale < 1.0:
            w = int(w * scale)
            h = int(h * scale)
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            img = img.resize((w, h), resample)

        if is_video:
            img = _add_play_overlay(img)

        return ImageTk.PhotoImage(img)
    except Exception:
        return None


# AI image description cache (msg_id -> description)
_image_description_cache: dict[int, str] = {}
# Lock for thread-safe cache access
_ai_cache_lock = threading.Lock()


def _describe_image_with_ai(data: bytes, msg_id: int, force: bool = False) -> str | None:
    """Use Ollama vision model to describe image content."""
    with _ai_cache_lock:
        if not force and msg_id in _image_description_cache:
            return _image_description_cache[msg_id]

    try:
        import base64
        import requests
        from translate_text import OLLAMA_BASE, OLLAMA_TIMEOUT

        # Encode image as base64
        img_b64 = base64.b64encode(data).decode('utf-8')

        # Try LLaVA or other vision models
        vision_models = ["llava", "llava:13b", "bakllava", "moondream"]

        prompt = (
            "Describe this image in detail. Focus on: military equipment (aircraft, helicopters, vehicles, weapons), "
            "any visible text or markings, location features, and notable objects. "
            "Be specific about aircraft/vehicle types if identifiable (e.g., Ka-52, Mi-28, T-72, BMP, BTR). "
            "Keep response under 100 words."
        )

        for model in vision_models:
            try:
                r = requests.post(
                    f"{OLLAMA_BASE}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "images": [img_b64],
                        "stream": False
                    },
                    timeout=OLLAMA_TIMEOUT,
                )
                if r.status_code == 200:
                    out = r.json()
                    response = (out.get("response") or "").strip()
                    if response:
                        with _ai_cache_lock:
                            _image_description_cache[msg_id] = response
                        return response
            except Exception:
                continue
    except Exception:
        pass

    return None


def _check_image_relevance(data: bytes, msg_id: int, search_query: str) -> tuple[bool, str | None]:
    """
    Ask AI if image is relevant to the search query.
    Returns (is_relevant, description).
    Uses a semantic approach rather than exact string matching.
    """
    try:
        import base64
        import requests
        from translate_text import OLLAMA_BASE, OLLAMA_TIMEOUT

        # Encode image as base64
        img_b64 = base64.b64encode(data).decode('utf-8')

        # Try LLaVA or other vision models
        vision_models = ["llava", "llava:13b", "bakllava", "moondream"]

        # Ask AI to evaluate relevance - be generous with matching
        prompt = f"""Look at this image and determine if it relates to: "{search_query}"

Be GENEROUS with matching. Answer YES if the image shows:
- The exact thing mentioned ("{search_query}")
- Similar or related equipment/objects (e.g., for "Ka-52" also match any attack helicopter, Russian helicopter, Kamov helicopter)
- The broader category (e.g., for "tank" match any armored vehicle)
- Related military scenes, operations, or combat footage that might include it
- Wreckage, debris, or destroyed versions of it

Only answer NO if the image is completely unrelated (e.g., a cat photo when searching for tanks).

Then briefly describe what you see (under 50 words).

Format your response EXACTLY like this:
RELEVANT: YES or NO
DESCRIPTION: [your description]"""

        for model in vision_models:
            try:
                r = requests.post(
                    f"{OLLAMA_BASE}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "images": [img_b64],
                        "stream": False
                    },
                    timeout=OLLAMA_TIMEOUT,
                )
                if r.status_code == 200:
                    out = r.json()
                    response = (out.get("response") or "").strip()
                    if response:
                        # Parse the response
                        is_relevant = "RELEVANT: YES" in response.upper() or "RELEVANT:YES" in response.upper()

                        # Extract description
                        desc = response
                        if "DESCRIPTION:" in response.upper():
                            desc_start = response.upper().find("DESCRIPTION:")
                            desc = response[desc_start + 12:].strip()

                        # Cache the description for later use
                        with _ai_cache_lock:
                            _image_description_cache[msg_id] = desc

                        return is_relevant, desc
            except Exception:
                continue
    except Exception:
        pass

    return False, None


def _batch_describe_images(media_dict: dict[int, bytes], progress_callback=None) -> dict[int, str]:
    """Analyze multiple images with AI. Returns {msg_id: description}."""
    results = {}
    total = len(media_dict)

    for idx, (msg_id, data) in enumerate(media_dict.items()):
        if progress_callback:
            progress_callback(idx + 1, total, msg_id)

        # Check cache first
        with _ai_cache_lock:
            if msg_id in _image_description_cache:
                results[msg_id] = _image_description_cache[msg_id]
                continue

        # Analyze with AI
        desc = _describe_image_with_ai(data, msg_id)
        if desc:
            results[msg_id] = desc

    return results


class MediaPreviewWindow:
    """Modal window for previewing media at full size."""

    def __init__(self, parent: tk.Tk, msg: TelegramMessage, data: bytes, translation: str = ""):
        self.parent = parent
        self.msg = msg
        self.data = data
        self.translation = translation
        self._photo_ref = None

        # Create toplevel window
        self.window = tk.Toplevel(parent)
        self.window.title(f"Media Preview - {msg.timestamp_display}")
        self.window.transient(parent)
        self.window.grab_set()

        # Get screen size for positioning
        screen_w = parent.winfo_screenwidth()
        screen_h = parent.winfo_screenheight()
        max_w = int(screen_w * 0.85)
        max_h = int(screen_h * 0.85)

        self.window.geometry(f"{max_w}x{max_h}")
        self.window.configure(bg="#1a1a1a")

        self._build_ui(max_w, max_h)

        # Center window
        self.window.update_idletasks()
        x = (screen_w - self.window.winfo_width()) // 2
        y = (screen_h - self.window.winfo_height()) // 2
        self.window.geometry(f"+{x}+{y}")

        # Bind escape to close
        self.window.bind("<Escape>", lambda e: self.window.destroy())
        self.window.bind("<Button-1>", self._on_click)

    def _build_ui(self, max_w: int, max_h: int):
        # Main frame
        main = ttk.Frame(self.window)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Image area (takes most space)
        img_frame = ttk.Frame(main)
        img_frame.pack(fill=tk.BOTH, expand=True)

        is_video = self.msg.media_type == "video"
        self._photo_ref = _bytes_to_full_image(self.data, max_w - 40, max_h - 250, is_video=is_video)

        if self._photo_ref:
            img_label = tk.Label(img_frame, image=self._photo_ref, bg="#1a1a1a", cursor="hand2")
            img_label.pack(expand=True)
            if is_video:
                img_label.bind("<Button-1>", self._open_video_url)

        # Info panel at bottom
        info_frame = ttk.Frame(main)
        info_frame.pack(fill=tk.X, pady=(10, 0))

        # Timestamp and URL
        info_text = f"[{self.msg.timestamp_display}]  "
        if self.msg.url:
            info_text += self.msg.url

        ttk.Label(info_frame, text=info_text, font=("Segoe UI", 9)).pack(anchor="w")

        # Message text (if any)
        if self.msg.text:
            text_frame = ttk.Frame(info_frame)
            text_frame.pack(fill=tk.X, pady=(5, 0))

            # Original text
            orig_label = ttk.Label(text_frame, text=self.msg.text[:200] + ("..." if len(self.msg.text) > 200 else ""),
                                   font=("Segoe UI", 9), foreground="gray", wraplength=max_w - 60)
            orig_label.pack(anchor="w")

            # Translated text
            if self.translation and self.translation != self.msg.text:
                trans_label = ttk.Label(text_frame, text=self.translation[:200] + ("..." if len(self.translation) > 200 else ""),
                                        font=("Segoe UI", 9), wraplength=max_w - 60)
                trans_label.pack(anchor="w", pady=(2, 0))

        # AI description area
        self._ai_frame = ttk.LabelFrame(info_frame, text="AI Analysis")
        self._ai_label = ttk.Label(self._ai_frame, text="", wraplength=max_w - 80, font=("Segoe UI", 9))
        self._ai_label.pack(padx=5, pady=5, anchor="w")

        # Show cached AI description if available
        cached_desc = _image_description_cache.get(self.msg.id)
        if cached_desc:
            self._ai_label.configure(text=cached_desc)
            self._ai_frame.pack(fill=tk.X, pady=(5, 0))

        # Buttons
        btn_frame = ttk.Frame(info_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(btn_frame, text="Open in Browser", command=self._open_url).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Save Image", command=self._save_image).pack(side=tk.LEFT, padx=(0, 5))
        analyze_text = "Re-analyze with AI" if cached_desc else "Analyze with AI"
        ttk.Button(btn_frame, text=analyze_text, command=self._analyze_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Close", command=self.window.destroy).pack(side=tk.RIGHT)

    def _on_click(self, event):
        # Click outside content area closes window
        pass

    def _open_url(self):
        if self.msg.url:
            import webbrowser
            webbrowser.open(self.msg.url)

    def _open_video_url(self, event=None):
        if self.msg.url:
            import webbrowser
            webbrowser.open(self.msg.url)

    def _save_image(self):
        folder = filedialog.askdirectory(title="Select Folder", parent=self.window)
        if folder:
            if _save_image(self.data, Path(folder), self.msg.id):
                messagebox.showinfo("Saved", f"Image saved to {folder}", parent=self.window)

    def _analyze_image(self):
        self._ai_frame.pack(fill=tk.X, pady=(5, 0))
        self._ai_label.configure(text="Analyzing image...")
        self.window.update()

        def do_analyze():
            # Force re-analysis
            desc = _describe_image_with_ai(self.data, self.msg.id, force=True)
            self.window.after(0, lambda: self._update_ai_result(desc))

        threading.Thread(target=do_analyze, daemon=True).start()

    def _update_ai_result(self, desc: str | None):
        """Update the AI label with the result (called from main thread)."""
        if desc:
            self._ai_label.configure(text=desc)
        else:
            self._ai_label.configure(text="Could not analyze image. Make sure Ollama is running with a vision model (llava, bakllava, moondream).")


def _translate_cached(text: str) -> str:
    """Translate with caching."""
    if not text:
        return ""
    if text in _translation_cache:
        return _translation_cache[text]
    result = to_english_from_russian(text) or text
    _translation_cache[text] = result
    return result


def _translate_batch_parallel(texts: list[str], max_workers: int = 4) -> list[str]:
    """Translate multiple texts in parallel."""
    if not texts:
        return []

    results = [""] * len(texts)

    def translate_one(idx_text):
        idx, text = idx_text
        return idx, _translate_cached(text)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(translate_one, enumerate(texts))
        for idx, translated in futures:
            results[idx] = translated

    return results


def _fetch_and_translate(
    client: TelegramClientSync,
    channel: str,
    result_queue: queue.Queue,
    limit: int = 100,
    min_date: Optional[datetime] = None,
    max_date: Optional[datetime] = None,
    search_query: Optional[str] = None,
    offset_date: Optional[datetime] = None,
    do_translate: bool = True,
) -> None:
    """Background task: fetch messages, download media, optionally translate."""
    try:
        # Send progress update
        result_queue.put(("progress", "Fetching messages..."))

        # Fetch messages
        messages = client.get_messages(
            channel,
            limit=limit,
            min_date=min_date,
            max_date=max_date,
            search_query=search_query,
            offset_date=offset_date,
        )

        if not messages:
            result_queue.put(("ok", [], [], {}))
            return

        result_queue.put(("progress", f"Found {len(messages)} messages..."))

        # Download media in parallel
        msg_ids_with_media = [m.id for m in messages if m.media_ids]
        media_bytes: dict[int, bytes] = {}
        if msg_ids_with_media:
            result_queue.put(("progress", f"Downloading {len(msg_ids_with_media)} images..."))
            media_bytes = client.download_media(channel, msg_ids_with_media)

        # Translate messages in parallel (if enabled)
        if do_translate:
            result_queue.put(("progress", f"Translating {len(messages)} messages..."))
            texts = [msg.text for msg in messages]
            translations = _translate_batch_parallel(texts, max_workers=4)
        else:
            translations = [msg.text for msg in messages]  # Use original text

        result_queue.put(("ok", messages, translations, media_bytes))

    except Exception as e:
        result_queue.put(("err", str(e), [], {}))


class TelegramViewerApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Telegram Viewer — Telethon API")
        self.root.minsize(900, 600)
        self.root.geometry("1300x800")

        self._client = get_client()
        self._result_queue: queue.Queue = queue.Queue()
        self._photo_refs: list = []
        self._all_messages: list[TelegramMessage] = []
        self._all_translations: list[str] = []
        self._all_media: dict[int, bytes] = {}

        # State
        self._current_channel = ""
        self._loading = False
        self._syncing_scroll = False
        self._left_visible = True
        self._authenticated = False

        # Live mode state
        self._live_mode = False
        self._live_timer_id: str | None = None
        self._live_search_query: str | None = None
        self._live_ai_assist: bool = False  # AI-assisted live search
        self._live_oldest_id: int | None = None  # Track oldest message for live search pagination
        self._seen_message_ids: set[int] = set()  # Track seen messages to avoid duplicates

        # Grid resize debounce
        self._grid_resize_timer: str | None = None
        self._last_grid_cols: int = 0

        self._build_ui()
        self._check_auth()

    def _check_auth(self):
        """Check if we're authenticated."""
        if not self._client.is_configured():
            self._status_var.set("Setup required. Run: python telegram_setup.py")
            self._fetch_btn.configure(state=tk.DISABLED)
            return

        try:
            self._authenticated = self._client.connect()
            if self._authenticated:
                self._status_var.set("Connected. Enter a channel and click Fetch.")
            else:
                self._status_var.set("Auth required. Run: python telegram_setup.py")
                self._fetch_btn.configure(state=tk.DISABLED)
        except Exception as e:
            self._status_var.set(f"Connection error: {e}")
            self._fetch_btn.configure(state=tk.DISABLED)

    def _build_ui(self) -> None:
        # Top control bar
        control_frame = ttk.Frame(self.root, padding=6)
        control_frame.pack(fill=tk.X)

        # Channel input
        ttk.Label(control_frame, text="Channel:").pack(side=tk.LEFT, padx=(0, 4))
        self._url_var = tk.StringVar(value=DEFAULT_CHANNEL)
        self._url_entry = ttk.Entry(control_frame, textvariable=self._url_var, width=20)
        self._url_entry.pack(side=tk.LEFT, padx=(0, 8))

        # Date range
        ttk.Label(control_frame, text="From:").pack(side=tk.LEFT, padx=(8, 4))
        self._date_from = DateEntry(control_frame, width=10, date_pattern="yyyy-mm-dd")
        self._date_from.set_date(datetime.now() - timedelta(days=7))
        self._date_from.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(control_frame, text="To:").pack(side=tk.LEFT, padx=(0, 4))
        self._date_to = DateEntry(control_frame, width=10, date_pattern="yyyy-mm-dd")
        self._date_to.set_date(datetime.now())
        self._date_to.pack(side=tk.LEFT, padx=(0, 8))

        # Limit
        ttk.Label(control_frame, text="Limit:").pack(side=tk.LEFT, padx=(8, 4))
        self._limit_var = tk.StringVar(value="100")
        limit_entry = ttk.Entry(control_frame, textvariable=self._limit_var, width=6)
        limit_entry.pack(side=tk.LEFT, padx=(0, 8))

        # Translate checkbox
        self._translate_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Translate", variable=self._translate_var).pack(side=tk.LEFT, padx=(4, 8))

        # Live mode checkbox
        self._live_var = tk.BooleanVar(value=False)
        self._live_check = ttk.Checkbutton(control_frame, text="Live", variable=self._live_var, command=self._on_live_toggled)
        self._live_check.pack(side=tk.LEFT, padx=(4, 8))

        # Fetch button
        self._fetch_btn = ttk.Button(control_frame, text="Fetch", command=self._on_fetch)
        self._fetch_btn.pack(side=tk.LEFT, padx=(12, 0))

        # Search bar (right side)
        self._live_search_btn = ttk.Button(control_frame, text="Live Search", command=self._on_live_search)
        self._live_search_btn.pack(side=tk.RIGHT, padx=(4, 0))

        # AI Assist checkbox for live search (analyzes images to find matches)
        self._ai_assist_var = tk.BooleanVar(value=False)
        self._ai_assist_check = ttk.Checkbutton(control_frame, text="AI", variable=self._ai_assist_var)
        self._ai_assist_check.pack(side=tk.RIGHT, padx=(4, 0))

        self._search_btn = ttk.Button(control_frame, text="Search", command=self._on_search)
        self._search_btn.pack(side=tk.RIGHT, padx=(4, 0))
        self._search_var = tk.StringVar()
        self._search_entry = ttk.Entry(control_frame, textvariable=self._search_var, width=25)
        self._search_entry.pack(side=tk.RIGHT, padx=(0, 4))
        ttk.Label(control_frame, text="Search:").pack(side=tk.RIGHT, padx=(8, 4))

        # Filter entry for loaded messages
        self._filter_var = tk.StringVar()
        self._filter_var.trace_add("write", self._on_filter_changed)

        filter_frame = ttk.Frame(self.root, padding=(6, 0, 6, 4))
        filter_frame.pack(fill=tk.X)
        ttk.Label(filter_frame, text="Filter loaded:").pack(side=tk.LEFT, padx=(0, 4))
        filter_entry = ttk.Entry(filter_frame, textvariable=self._filter_var, width=30)
        filter_entry.pack(side=tk.LEFT)

        # Include AI checkbox (filter includes AI descriptions)
        self._include_ai_var = tk.BooleanVar(value=True)
        self._include_ai_check = ttk.Checkbutton(filter_frame, text="Include AI", variable=self._include_ai_var,
                                                  command=self._on_filter_changed)
        self._include_ai_check.pack(side=tk.LEFT, padx=(8, 0))

        # Translate Now button (for translating after loading)
        self._translate_now_btn = ttk.Button(filter_frame, text="Translate Now", command=self._on_translate_now)
        self._translate_now_btn.pack(side=tk.RIGHT, padx=(8, 0))

        # Download Media button
        self._download_btn = ttk.Button(filter_frame, text="Download Media", command=self._on_download_media)
        self._download_btn.pack(side=tk.RIGHT, padx=(8, 0))

        # View mode toggle
        self._view_mode = tk.StringVar(value="list")
        ttk.Radiobutton(filter_frame, text="List", variable=self._view_mode, value="list",
                        command=self._on_view_mode_changed).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Radiobutton(filter_frame, text="Grid", variable=self._view_mode, value="grid",
                        command=self._on_view_mode_changed).pack(side=tk.RIGHT, padx=(8, 0))

        # Media filter
        self._media_filter = tk.StringVar(value="all")
        ttk.Label(filter_frame, text="Show:").pack(side=tk.RIGHT, padx=(8, 4))
        media_combo = ttk.Combobox(filter_frame, textvariable=self._media_filter, width=12, state="readonly",
                                   values=["all", "with media", "images only", "videos only"])
        media_combo.pack(side=tk.RIGHT)
        media_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_display())

        # Analytics toggle
        self._analytics_visible = tk.BooleanVar(value=False)
        self._analytics_toggle = ttk.Checkbutton(
            filter_frame, text="Analytics", variable=self._analytics_visible,
            command=self._toggle_analytics
        )
        self._analytics_toggle.pack(side=tk.RIGHT, padx=(12, 0))

        # Main content area with analytics panel
        self._main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self._main_paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Left side: message viewer content
        main = ttk.Frame(self._main_paned)
        self._main_paned.add(main, weight=3)

        # Right side: analytics panel (created but not added until toggled)
        self._analytics_frame = ttk.LabelFrame(self._main_paned, text="", padding=4)
        self._analytics_panel = AnalyticsPanel(self._analytics_frame, self._get_loaded_data)

        self._paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)

        # Left panel
        self._left_frame = ttk.LabelFrame(main, text="Original (Russian)", padding=4)
        self._paned.add(self._left_frame, weight=1)

        left_text_frame = ttk.Frame(self._left_frame)
        left_text_frame.pack(fill=tk.BOTH, expand=True)

        self._raw_text = tk.Text(left_text_frame, wrap=tk.WORD, font=("Segoe UI", 10))
        self._left_scrollbar = ttk.Scrollbar(left_text_frame, orient=tk.VERTICAL, command=self._on_left_scroll)
        self._raw_text.configure(yscrollcommand=self._on_left_scrollbar_move)

        self._left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._raw_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        collapse_btn = ttk.Button(self._left_frame, text="< Hide", command=self._toggle_left)
        collapse_btn.pack(pady=(4, 0))

        # Right panel
        right_frame = ttk.LabelFrame(main, text="Translated (English)", padding=4)
        self._paned.add(right_frame, weight=1)

        right_text_frame = ttk.Frame(right_frame)
        right_text_frame.pack(fill=tk.BOTH, expand=True)

        self._translated_text = tk.Text(right_text_frame, wrap=tk.WORD, font=("Segoe UI", 10))
        self._right_scrollbar = ttk.Scrollbar(right_text_frame, orient=tk.VERTICAL, command=self._on_right_scroll)
        self._translated_text.configure(yscrollcommand=self._on_right_scrollbar_move)

        self._right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._translated_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._paned.pack(fill=tk.BOTH, expand=True)

        # Grid view (hidden by default)
        self._grid_frame = ttk.Frame(main)
        self._grid_canvas = tk.Canvas(self._grid_frame, bg="#1a1a1a", highlightthickness=0)
        self._grid_scrollbar = ttk.Scrollbar(self._grid_frame, orient=tk.VERTICAL, command=self._grid_canvas.yview)
        self._grid_inner = ttk.Frame(self._grid_canvas)

        self._grid_canvas.configure(yscrollcommand=self._grid_scrollbar.set)
        self._grid_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._grid_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._grid_window_id = self._grid_canvas.create_window((0, 0), window=self._grid_inner, anchor="nw")
        self._grid_inner.bind("<Configure>", self._on_grid_inner_configure)
        self._grid_canvas.bind("<Configure>", self._on_grid_canvas_configure)

        # Mouse wheel scrolling for grid (bind to canvas and inner frame)
        def _on_mousewheel(event):
            self._grid_canvas.yview_scroll(-1 * (event.delta // 120), "units")
        self._grid_canvas.bind("<MouseWheel>", _on_mousewheel)
        self._grid_inner.bind("<MouseWheel>", _on_mousewheel)
        self._grid_canvas.bind("<Button-4>", lambda e: self._grid_canvas.yview_scroll(-1, "units"))
        self._grid_canvas.bind("<Button-5>", lambda e: self._grid_canvas.yview_scroll(1, "units"))

        # Status bar
        self._status_var = tk.StringVar(value="Initializing...")
        ttk.Label(self.root, textvariable=self._status_var, padding=4).pack(fill=tk.X)

        # Configure text tags
        self._raw_text.tag_configure("highlight", background="yellow", foreground="black")
        self._translated_text.tag_configure("highlight", background="yellow", foreground="black")
        self._raw_text.tag_configure("timestamp", foreground="gray")
        self._translated_text.tag_configure("timestamp", foreground="gray")

        self.root.after(100, self._poll_queue)

    def _on_left_scroll(self, *args):
        self._raw_text.yview(*args)
        if not self._syncing_scroll:
            self._syncing_scroll = True
            self._translated_text.yview(*args)
            self._syncing_scroll = False

    def _on_right_scroll(self, *args):
        self._translated_text.yview(*args)
        if not self._syncing_scroll:
            self._syncing_scroll = True
            self._raw_text.yview(*args)
            self._syncing_scroll = False

    def _on_left_scrollbar_move(self, first, last):
        self._left_scrollbar.set(first, last)
        if not self._syncing_scroll:
            self._syncing_scroll = True
            self._translated_text.yview_moveto(first)
            self._right_scrollbar.set(first, last)
            self._syncing_scroll = False

    def _on_right_scrollbar_move(self, first, last):
        self._right_scrollbar.set(first, last)
        if not self._syncing_scroll:
            self._syncing_scroll = True
            self._raw_text.yview_moveto(first)
            self._left_scrollbar.set(first, last)
            self._syncing_scroll = False

    def _toggle_left(self) -> None:
        self._left_visible = not self._left_visible
        if self._left_visible:
            if hasattr(self, "_collapse_placeholder"):
                try:
                    self._paned.remove(self._collapse_placeholder)
                except tk.TclError:
                    pass
            self._paned.insert(0, self._left_frame, weight=1)
        else:
            self._paned.remove(self._left_frame)
            if not hasattr(self, "_collapse_placeholder"):
                self._collapse_placeholder = ttk.Frame(self._paned)
                self._show_btn = ttk.Button(self._collapse_placeholder, text=">", command=self._toggle_left)
                self._show_btn.pack(pady=8, padx=4)
            self._paned.insert(0, self._collapse_placeholder, weight=0)

    def _toggle_analytics(self) -> None:
        """Toggle the analytics panel visibility."""
        if self._analytics_visible.get():
            # Show analytics panel
            self._main_paned.add(self._analytics_frame, weight=1)
        else:
            # Hide analytics panel
            try:
                self._main_paned.remove(self._analytics_frame)
            except tk.TclError:
                pass

    def _get_loaded_data(self) -> tuple:
        """Return loaded data for analytics panel."""
        return (
            self._all_messages,
            self._all_translations,
            self._all_media,
            _image_description_cache.copy(),
        )

    def _on_fetch(self) -> None:
        """Fetch messages for date range, or stop live mode if active."""
        if self._live_mode:
            self._stop_live_mode()
            self._status_var.set(f"Stopped. {len(self._all_messages)} messages loaded.")
            return
        self._do_fetch(use_search=False)

    def _on_search(self) -> None:
        """Search messages with bilingual support (English + Russian)."""
        search_query = self._search_var.get().strip()
        if not search_query:
            messagebox.showinfo("Search", "Enter a search term.", parent=self.root)
            return
        self._stop_live_mode()
        self._do_fetch(use_search=True)

    def _on_live_toggled(self) -> None:
        """Handle live mode checkbox toggle."""
        if self._live_var.get():
            self._start_live_mode()
        else:
            self._stop_live_mode()

    def _start_live_mode(self) -> None:
        """Start live polling for newest posts."""
        if self._live_mode:
            return
        self._live_mode = True
        self._live_search_query = None
        self._seen_message_ids = set(m.id for m in self._all_messages)
        self._status_var.set("Live mode: fetching newest posts...")
        self._fetch_btn.configure(text="Stop Live")
        self._do_live_fetch()

    def _stop_live_mode(self) -> None:
        """Stop live polling."""
        self._live_mode = False
        self._live_search_query = None
        self._live_ai_assist = False
        self._live_oldest_id = None
        if self._live_timer_id:
            self.root.after_cancel(self._live_timer_id)
            self._live_timer_id = None
        self._live_var.set(False)
        self._fetch_btn.configure(text="Fetch")
        self._live_search_btn.configure(state=tk.NORMAL)

    def _do_live_fetch(self) -> None:
        """Fetch newest posts for live mode."""
        if not self._live_mode:
            return

        channel = self._url_var.get().strip()
        if not channel:
            self._stop_live_mode()
            return

        def fetch_live():
            try:
                # Fetch latest 20 messages
                messages = self._client.get_messages(
                    channel,
                    limit=20,
                    search_query=self._live_search_query,
                )

                if not messages:
                    self._result_queue.put(("live_update", [], [], {}))
                    return

                # Filter to new messages only
                new_messages = [m for m in messages if m.id not in self._seen_message_ids]

                if not new_messages:
                    self._result_queue.put(("live_update", [], [], {}))
                    return

                # Download media for new messages
                msg_ids_with_media = [m.id for m in new_messages if m.media_ids]
                media_bytes: dict[int, bytes] = {}
                if msg_ids_with_media:
                    media_bytes = self._client.download_media(channel, msg_ids_with_media)

                # Translate if enabled
                if self._translate_var.get():
                    texts = [msg.text for msg in new_messages]
                    translations = _translate_batch_parallel(texts, max_workers=4)
                else:
                    translations = [msg.text for msg in new_messages]

                self._result_queue.put(("live_update", new_messages, translations, media_bytes))

            except Exception as e:
                self._result_queue.put(("live_error", str(e)))

        threading.Thread(target=fetch_live, daemon=True).start()

    def _on_live_search(self) -> None:
        """Start live search mode - continuously fetch posts matching search term."""
        search_query = self._search_var.get().strip()
        if not search_query:
            messagebox.showinfo("Search", "Enter a search term.", parent=self.root)
            return

        # Check if AI assist is enabled
        ai_assist = self._ai_assist_var.get()

        # Translate to Russian for bilingual search
        russian_term = _translate_to_russian(search_query)
        status_parts = [f"'{search_query}'"]
        if russian_term:
            status_parts.append(f"'{russian_term}'")
        if ai_assist:
            status_parts.append("+ AI vision")
        self._status_var.set(f"Live search: {' + '.join(status_parts)}")

        # Clear and prepare for live search
        self._raw_text.delete("1.0", tk.END)
        self._translated_text.delete("1.0", tk.END)
        self._photo_refs.clear()
        self._all_messages.clear()
        self._all_translations.clear()
        self._all_media.clear()
        self._seen_message_ids.clear()
        self._live_oldest_id = None

        self._live_mode = True
        self._live_search_query = search_query
        self._live_ai_assist = ai_assist  # Store AI assist flag
        self._live_var.set(True)
        self._fetch_btn.configure(text="Stop Live")
        self._live_search_btn.configure(state=tk.DISABLED)

        # Start fetching
        self._do_live_search_fetch(search_query, russian_term)

    def _do_live_search_fetch(self, english_term: str, russian_term: str | None) -> None:
        """Fetch search results going backward in time, optionally with AI analysis."""
        if not self._live_mode:
            return

        channel = self._url_var.get().strip()
        if not channel:
            self._stop_live_mode()
            return

        ai_assist = getattr(self, '_live_ai_assist', False)

        # Calculate offset_date for pagination (use oldest seen message timestamp)
        offset_date = None
        if self._all_messages:
            oldest_msg = min(self._all_messages, key=lambda m: m.timestamp or datetime.max.replace(tzinfo=timezone.utc))
            if oldest_msg.timestamp:
                offset_date = oldest_msg.timestamp

        def fetch_search():
            try:
                all_new_messages = []
                text_match_ids = set()

                # Search for English term
                messages_en = self._client.get_messages(
                    channel,
                    limit=50,
                    search_query=english_term,
                    offset_date=offset_date,
                )

                # Search for Russian term if available
                messages_ru = []
                if russian_term:
                    messages_ru = self._client.get_messages(
                        channel,
                        limit=50,
                        search_query=russian_term,
                        offset_date=offset_date,
                    )

                # Merge text-matched messages and deduplicate
                seen_ids = set()
                for msg in messages_en + messages_ru:
                    if msg.id not in seen_ids and msg.id not in self._seen_message_ids:
                        seen_ids.add(msg.id)
                        text_match_ids.add(msg.id)
                        all_new_messages.append(msg)

                # AI-assisted search: fetch additional messages with media and analyze them
                ai_matched_messages = []
                ai_media_bytes: dict[int, bytes] = {}

                if ai_assist:
                    self._result_queue.put(("progress", f"Fetching media for AI analysis..."))

                    # Fetch more messages to scan with AI (larger batch for better coverage)
                    media_messages = self._client.get_messages(
                        channel,
                        limit=100,  # Larger batch for AI scanning
                        offset_date=offset_date,
                    )

                    # Filter to messages with media that weren't already matched by text
                    media_candidates = [
                        m for m in media_messages
                        if m.media_ids and m.id not in seen_ids and m.id not in self._seen_message_ids
                    ]

                    if media_candidates:
                        self._result_queue.put(("progress", f"Found {len(media_candidates)} images to analyze..."))

                        # Download media for AI analysis
                        candidate_ids = [m.id for m in media_candidates]
                        ai_media_bytes = self._client.download_media(channel, candidate_ids)

                        # Use semantic relevance check - ask AI directly if image relates to query
                        # PARALLEL AI ANALYSIS for speed
                        total_to_analyze = len(ai_media_bytes)
                        self._result_queue.put(("progress", f"AI analyzing {total_to_analyze} images in parallel..."))

                        # Prepare tasks: (msg, data, search_query)
                        ai_tasks = []
                        for msg in media_candidates:
                            if msg.id in ai_media_bytes:
                                ai_tasks.append((msg, ai_media_bytes[msg.id], english_term))

                        # Run AI analysis in parallel
                        completed = 0
                        matches_found = 0

                        def analyze_one(task):
                            msg, data, query = task
                            is_relevant, desc = _check_image_relevance(data, msg.id, query)
                            return msg, is_relevant, desc

                        with concurrent.futures.ThreadPoolExecutor(max_workers=AI_WORKERS) as executor:
                            futures = {executor.submit(analyze_one, task): task for task in ai_tasks}

                            for future in concurrent.futures.as_completed(futures):
                                completed += 1
                                try:
                                    msg, is_relevant, desc = future.result()
                                    if is_relevant:
                                        ai_matched_messages.append(msg)
                                        seen_ids.add(msg.id)
                                        matches_found += 1
                                        self._result_queue.put(("progress", f"AI: {completed}/{total_to_analyze} done, {matches_found} matches"))
                                    elif completed % 5 == 0:  # Update progress every 5 images
                                        self._result_queue.put(("progress", f"AI: {completed}/{total_to_analyze} done, {matches_found} matches"))
                                except Exception:
                                    pass  # Skip failed analyses

                        self._result_queue.put(("progress", f"AI complete: {matches_found} matches in {total_to_analyze} images"))

                # Add AI-matched messages to results
                all_new_messages.extend(ai_matched_messages)

                # Sort by timestamp (newest first)
                all_new_messages.sort(key=lambda m: m.timestamp or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

                if not all_new_messages:
                    self._result_queue.put(("live_search_done", 0))
                    return

                # Download media for text-matched messages
                msg_ids_with_media = [m.id for m in all_new_messages if m.media_ids and m.id not in ai_media_bytes]
                media_bytes: dict[int, bytes] = {}
                if msg_ids_with_media:
                    media_bytes = self._client.download_media(channel, msg_ids_with_media)

                # Merge with AI-analyzed media
                media_bytes.update(ai_media_bytes)

                # Translate if enabled
                if self._translate_var.get():
                    texts = [msg.text for msg in all_new_messages]
                    translations = _translate_batch_parallel(texts, max_workers=4)
                else:
                    translations = [msg.text for msg in all_new_messages]

                # Report AI matches in result
                ai_match_count = len(ai_matched_messages)
                self._result_queue.put(("live_search_batch", all_new_messages, translations, media_bytes, ai_match_count))

            except Exception as e:
                self._result_queue.put(("live_error", str(e)))

        batch_num = (len(self._all_messages) // 50) + 1
        self._status_var.set(f"Searching batch {batch_num}...")
        threading.Thread(target=fetch_search, daemon=True).start()

    def _do_fetch(self, use_search: bool = False) -> None:
        channel = self._url_var.get().strip()
        if not channel:
            messagebox.showinfo("Input", "Enter a Telegram channel.", parent=self.root)
            return

        try:
            limit = int(self._limit_var.get())
        except ValueError:
            limit = 100

        # Clear state
        self._status_var.set("Fetching...")
        self._fetch_btn.configure(state=tk.DISABLED)
        self._search_btn.configure(state=tk.DISABLED)
        self._raw_text.delete("1.0", tk.END)
        self._translated_text.delete("1.0", tk.END)
        self._photo_refs.clear()
        self._all_messages.clear()
        self._all_translations.clear()
        self._all_media.clear()
        self._current_channel = channel
        self._loading = True

        # Date range (with timezone)
        min_date = datetime.combine(
            self._date_from.get_date(),
            datetime.min.time()
        ).replace(tzinfo=timezone.utc)
        max_date = datetime.combine(
            self._date_to.get_date(),
            datetime.max.time()
        ).replace(tzinfo=timezone.utc)

        search_query = self._search_var.get().strip() if use_search else None

        # For bilingual search, translate English to Russian
        russian_search = None
        if search_query:
            russian_search = _translate_to_russian(search_query)
            if russian_search:
                self._status_var.set(f"Searching: '{search_query}' + '{russian_search}'...")

        do_translate = self._translate_var.get()

        def fetch_with_bilingual():
            """Fetch with both English and Russian search terms."""
            try:
                self._result_queue.put(("progress", "Fetching messages..."))

                all_messages = []
                seen_ids = set()

                # Fetch with English term
                messages_en = self._client.get_messages(
                    channel,
                    limit=limit,
                    min_date=min_date,
                    max_date=max_date,
                    search_query=search_query,
                    offset_date=max_date,
                )
                for m in messages_en:
                    if m.id not in seen_ids:
                        seen_ids.add(m.id)
                        all_messages.append(m)

                # Fetch with Russian term if available
                if russian_search:
                    self._result_queue.put(("progress", f"Searching Russian: '{russian_search}'..."))
                    messages_ru = self._client.get_messages(
                        channel,
                        limit=limit,
                        min_date=min_date,
                        max_date=max_date,
                        search_query=russian_search,
                        offset_date=max_date,
                    )
                    for m in messages_ru:
                        if m.id not in seen_ids:
                            seen_ids.add(m.id)
                            all_messages.append(m)

                # Sort by timestamp (newest first)
                all_messages.sort(key=lambda m: m.timestamp or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

                # Limit results
                all_messages = all_messages[:limit]

                if not all_messages:
                    self._result_queue.put(("ok", [], [], {}))
                    return

                self._result_queue.put(("progress", f"Found {len(all_messages)} messages..."))

                # Download media
                msg_ids_with_media = [m.id for m in all_messages if m.media_ids]
                media_bytes: dict[int, bytes] = {}
                if msg_ids_with_media:
                    self._result_queue.put(("progress", f"Downloading {len(msg_ids_with_media)} images..."))
                    media_bytes = self._client.download_media(channel, msg_ids_with_media)

                # Translate
                if do_translate:
                    self._result_queue.put(("progress", f"Translating {len(all_messages)} messages..."))
                    texts = [msg.text for msg in all_messages]
                    translations = _translate_batch_parallel(texts, max_workers=4)
                else:
                    translations = [msg.text for msg in all_messages]

                self._result_queue.put(("ok", all_messages, translations, media_bytes))

            except Exception as e:
                self._result_queue.put(("err", str(e), [], {}))

        if use_search and search_query:
            threading.Thread(target=fetch_with_bilingual, daemon=True).start()
        else:
            threading.Thread(
                target=_fetch_and_translate,
                args=(
                    self._client,
                    channel,
                    self._result_queue,
                    limit,
                    min_date,
                    max_date,
                    search_query,
                    max_date,
                    do_translate,
                ),
                daemon=True
            ).start()

    def _on_translate_now(self) -> None:
        """Translate loaded messages that haven't been translated."""
        if not self._all_messages:
            return

        self._translate_now_btn.configure(state=tk.DISABLED)
        self._status_var.set("Translating...")

        def do_translate():
            texts = [msg.text for msg in self._all_messages]
            translations = _translate_batch_parallel(texts, max_workers=4)
            self._result_queue.put(("translated", translations))

        threading.Thread(target=do_translate, daemon=True).start()

    def _on_download_media(self) -> None:
        """Download all media from loaded messages to a folder."""
        if not self._all_media:
            messagebox.showinfo("Download", "No media loaded.", parent=self.root)
            return

        # Ask for folder
        folder = filedialog.askdirectory(title="Select Download Folder", parent=self.root)
        if not folder:
            return

        folder_path = Path(folder)
        self._status_var.set("Downloading media...")
        self._download_btn.configure(state=tk.DISABLED)

        def do_download():
            count = 0
            for msg in self._all_messages:
                if msg.id in self._all_media:
                    data = self._all_media[msg.id]
                    if _save_image(data, folder_path, msg.id):
                        count += 1
            self._result_queue.put(("download_done", count))

        threading.Thread(target=do_download, daemon=True).start()

    def _on_view_mode_changed(self) -> None:
        """Switch between list and grid view."""
        mode = self._view_mode.get()
        if mode == "grid":
            self._paned.pack_forget()
            self._grid_frame.pack(fill=tk.BOTH, expand=True)
            # Delay grid refresh to allow canvas to get proper size
            self.root.after(50, self._refresh_grid_view)
        else:
            self._grid_frame.pack_forget()
            self._paned.pack(fill=tk.BOTH, expand=True)
            self._refresh_display()

    def _on_grid_inner_configure(self, event):
        """Update scroll region when inner frame changes size."""
        self._grid_canvas.configure(scrollregion=self._grid_canvas.bbox("all"))

    def _on_grid_canvas_configure(self, event):
        """When canvas resizes, adjust inner frame width and refresh grid layout."""
        # Make inner frame at least as wide as canvas
        canvas_width = event.width
        self._grid_canvas.itemconfig(self._grid_window_id, width=canvas_width)

        # Check if column count would change
        cell_size = GRID_THUMB_SIZE + GRID_PADDING * 2
        new_cols = max(1, canvas_width // cell_size)

        # Only refresh if column count changed and we have content
        if new_cols != self._last_grid_cols and self._all_messages and self._view_mode.get() == "grid":
            self._last_grid_cols = new_cols
            # Debounce: cancel pending refresh and schedule a new one
            if self._grid_resize_timer:
                self.root.after_cancel(self._grid_resize_timer)
            self._grid_resize_timer = self.root.after(150, self._refresh_grid_view)

    def _open_media_preview(self, msg: TelegramMessage) -> None:
        """Open a media preview modal for the given message."""
        if msg.id not in self._all_media:
            return

        # Get translation for this message
        translation = ""
        try:
            idx = next(i for i, m in enumerate(self._all_messages) if m.id == msg.id)
            if idx < len(self._all_translations):
                translation = self._all_translations[idx]
        except StopIteration:
            pass

        data = self._all_media[msg.id]
        MediaPreviewWindow(self.root, msg, data, translation)

    def _refresh_grid_view(self):
        """Populate the grid view with media thumbnails."""
        # Clear existing
        for widget in self._grid_inner.winfo_children():
            widget.destroy()
        self._photo_refs.clear()

        # Filter messages
        filter_term = self._filter_var.get().strip().lower()
        media_filter = self._media_filter.get()
        include_ai = self._include_ai_var.get()

        filtered_msgs = []
        for i, msg in enumerate(self._all_messages):
            # Apply text filter (includes AI descriptions if enabled)
            if filter_term:
                trans = self._all_translations[i] if i < len(self._all_translations) else msg.text
                text_match = filter_term in msg.text.lower() or filter_term in trans.lower()
                ai_match = False
                if include_ai and msg.id in _image_description_cache:
                    ai_desc = _image_description_cache[msg.id].lower()
                    ai_match = filter_term in ai_desc
                if not text_match and not ai_match:
                    continue
            # Apply media filter
            if media_filter == "with media":
                if msg.id not in self._all_media:
                    continue
            elif media_filter == "images only":
                if msg.id not in self._all_media or msg.media_type != "photo":
                    continue
            elif media_filter == "videos only":
                if msg.id not in self._all_media or msg.media_type != "video":
                    continue
            filtered_msgs.append((i, msg))  # Keep track of index for translation lookup

        # Calculate columns based on canvas width
        canvas_width = self._grid_canvas.winfo_width()
        if canvas_width < 100:  # Not yet properly sized
            canvas_width = 800
        cell_size = GRID_THUMB_SIZE + GRID_PADDING * 2
        cols = max(1, canvas_width // cell_size)

        # Configure grid columns to expand evenly
        for c in range(cols):
            self._grid_inner.columnconfigure(c, weight=1, minsize=cell_size)

        media_count = 0
        for grid_idx, (orig_idx, msg) in enumerate(filtered_msgs):
            if msg.id not in self._all_media:
                continue

            data = self._all_media[msg.id]
            is_video = msg.media_type == "video"
            thumb = _bytes_to_thumbnail(data, size=GRID_THUMB_SIZE, is_video=is_video)
            if thumb:
                self._photo_refs.append(thumb)
                row = media_count // cols
                col = media_count % cols
                media_count += 1

                # Create card-like frame for each item
                # Use different border color if AI-analyzed
                has_ai = msg.id in _image_description_cache
                bg_color = "#2a4a2a" if has_ai else "#2a2a2a"  # Greenish tint if AI analyzed

                frame = tk.Frame(self._grid_inner, bg=bg_color, padx=4, pady=4)
                frame.grid(row=row, column=col, padx=GRID_PADDING, pady=GRID_PADDING, sticky="nsew")

                label = tk.Label(frame, image=thumb, bg=bg_color, cursor="hand2")
                label.pack()

                # Bind click to open preview
                label.bind("<Button-1>", lambda e, m=msg: self._open_media_preview(m))
                frame.bind("<Button-1>", lambda e, m=msg: self._open_media_preview(m))

                # Add timestamp and AI indicator
                info_text = msg.timestamp_display or ""
                if has_ai:
                    info_text = "🔍 " + info_text  # AI indicator

                if info_text:
                    ts_label = tk.Label(frame, text=info_text, font=("Segoe UI", 8),
                                       bg=bg_color, fg="#888888", cursor="hand2")
                    ts_label.pack()
                    ts_label.bind("<Button-1>", lambda e, m=msg: self._open_media_preview(m))

                # Bind mousewheel to child widgets for scrolling
                for widget in (frame, label):
                    widget.bind("<MouseWheel>", lambda e: self._grid_canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        # Scroll to top
        self._grid_canvas.yview_moveto(0)
        self._status_var.set(f"Showing {media_count} media items in grid. Click to preview.")

    def _on_filter_changed(self, *args):
        """Filter displayed messages locally."""
        if self._view_mode.get() == "grid":
            self._refresh_grid_view()
        else:
            self._refresh_display()

    def _refresh_display(self):
        """Redraw content with filter applied (list view)."""
        filter_term = self._filter_var.get().strip().lower()
        media_filter = self._media_filter.get()
        include_ai = self._include_ai_var.get()

        self._raw_text.delete("1.0", tk.END)
        self._translated_text.delete("1.0", tk.END)
        self._photo_refs.clear()

        for i, msg in enumerate(self._all_messages):
            trans = self._all_translations[i] if i < len(self._all_translations) else msg.text

            # Apply text filter (includes AI descriptions if enabled)
            if filter_term:
                text_match = filter_term in msg.text.lower() or filter_term in trans.lower()
                ai_match = False
                if include_ai and msg.id in _image_description_cache:
                    ai_desc = _image_description_cache[msg.id].lower()
                    ai_match = filter_term in ai_desc
                if not text_match and not ai_match:
                    continue

            # Apply media filter
            if media_filter == "with media":
                if msg.id not in self._all_media:
                    continue
            elif media_filter == "images only":
                if msg.id not in self._all_media or msg.media_type != "photo":
                    continue
            elif media_filter == "videos only":
                if msg.id not in self._all_media or msg.media_type != "video":
                    continue

            self._insert_message(msg, trans, filter_term)

    def _insert_message(self, msg: TelegramMessage, translation: str, highlight: str = ""):
        """Insert a single message into both panels."""
        # Timestamp
        if msg.timestamp_display:
            self._raw_text.insert(tk.END, f"[{msg.timestamp_display}]\n", "timestamp")
            self._translated_text.insert(tk.END, f"[{msg.timestamp_display}]\n", "timestamp")

        # Text
        self._insert_with_highlight(self._raw_text, msg.text + "\n\n", highlight)
        self._insert_with_highlight(self._translated_text, translation + "\n\n", highlight)

        # Media
        if msg.id in self._all_media:
            data = self._all_media[msg.id]
            is_video = msg.media_type == "video"
            for text_widget in (self._raw_text, self._translated_text):
                photo = _bytes_to_photo(data, is_video=is_video)
                if photo:
                    self._photo_refs.append(photo)
                    img_frame = tk.Frame(text_widget, bg="#1a1a1a")
                    img_label = tk.Label(img_frame, image=photo, bg="#1a1a1a", cursor="hand2")
                    img_label.pack(side=tk.LEFT, padx=2, pady=2)
                    # Bind click to open preview
                    img_label.bind("<Button-1>", lambda e, m=msg: self._open_media_preview(m))
                    text_widget.window_create(tk.END, window=img_frame)
                    text_widget.insert(tk.END, "\n")

        # Separator
        self._raw_text.insert(tk.END, "\n" + "-" * 50 + "\n\n")
        self._translated_text.insert(tk.END, "\n" + "-" * 50 + "\n\n")

    def _insert_with_highlight(self, text_widget: tk.Text, text: str, search_term: str):
        """Insert text with highlighted terms."""
        if not search_term:
            text_widget.insert(tk.END, text)
            return

        text_widget.insert(tk.END, text)
        search_start = "1.0"
        while True:
            pos = text_widget.search(search_term, search_start, tk.END, nocase=True)
            if not pos:
                break
            end_pos = f"{pos}+{len(search_term)}c"
            text_widget.tag_add("highlight", pos, end_pos)
            search_start = end_pos

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self._result_queue.get_nowait()
                if msg[0] == "progress":
                    # Progress update
                    self._status_var.set(msg[1])
                elif msg[0] == "ok":
                    _, messages, translations, media = msg

                    self._all_messages = messages
                    self._all_translations = translations
                    self._all_media = media

                    if messages:
                        self._refresh_display()
                        self._status_var.set(f"Loaded {len(messages)} messages.")
                    else:
                        self._raw_text.insert(tk.END, "(no messages found)")
                        self._translated_text.insert(tk.END, "(no messages found)")
                        self._status_var.set("No messages found in the specified range.")

                    self._loading = False
                    self._fetch_btn.configure(state=tk.NORMAL)
                    self._search_btn.configure(state=tk.NORMAL)
                elif msg[0] == "translated":
                    # Received translations for existing messages
                    _, translations = msg
                    self._all_translations = translations
                    self._refresh_display()
                    self._status_var.set(f"Translated {len(translations)} messages.")
                    self._translate_now_btn.configure(state=tk.NORMAL)
                elif msg[0] == "download_done":
                    # Media download completed
                    _, count = msg
                    self._download_btn.configure(state=tk.NORMAL)
                    self._status_var.set(f"Downloaded {count} images.")
                    messagebox.showinfo("Download Complete", f"Downloaded {count} images.", parent=self.root)
                elif msg[0] == "live_update":
                    # Live mode: new messages received
                    _, new_messages, translations, media = msg
                    if new_messages:
                        # Prepend new messages
                        for m in new_messages:
                            self._seen_message_ids.add(m.id)
                        self._all_messages = new_messages + self._all_messages
                        self._all_translations = translations + self._all_translations
                        self._all_media.update(media)
                        self._refresh_display()
                        self._status_var.set(f"Live: {len(self._all_messages)} messages (+{len(new_messages)} new)")
                    else:
                        self._status_var.set(f"Live: {len(self._all_messages)} messages (no new)")
                    # Schedule next poll
                    if self._live_mode:
                        self._live_timer_id = self.root.after(LIVE_POLL_INTERVAL_MS, self._do_live_fetch)
                elif msg[0] == "live_search_batch":
                    # Live search: batch of results
                    _, new_messages, translations, media, *rest = msg
                    ai_match_count = rest[0] if rest else 0
                    if new_messages:
                        for m in new_messages:
                            self._seen_message_ids.add(m.id)
                        self._all_messages.extend(new_messages)
                        self._all_translations.extend(translations)
                        self._all_media.update(media)
                        self._refresh_display()
                        # Show AI match info if any
                        status = f"Live search: found {len(self._all_messages)} matches"
                        if ai_match_count > 0:
                            status += f" ({ai_match_count} via AI)"
                        status += "..."
                        self._status_var.set(status)
                        # Continue searching (schedule next batch)
                        if self._live_mode and self._live_search_query:
                            russian_term = _translate_to_russian(self._live_search_query)
                            self._live_timer_id = self.root.after(500, lambda: self._do_live_search_fetch(self._live_search_query, russian_term))
                    else:
                        # No more results
                        self._status_var.set(f"Live search complete: {len(self._all_messages)} matches")
                        self._stop_live_mode()
                elif msg[0] == "live_search_done":
                    # Live search finished
                    _, count = msg
                    self._status_var.set(f"Live search complete: {len(self._all_messages)} matches")
                    self._stop_live_mode()
                elif msg[0] == "live_error":
                    # Error during live mode
                    _, err = msg
                    self._status_var.set(f"Live error: {err}")
                    self._stop_live_mode()
                elif msg[0] == "err":
                    _, err = msg[0], msg[1]
                    self._status_var.set(f"Error: {err}")
                    messagebox.showerror("Error", str(err), parent=self.root)
                    self._loading = False
                    self._fetch_btn.configure(state=tk.NORMAL)
                    self._search_btn.configure(state=tk.NORMAL)
                    self._translate_now_btn.configure(state=tk.NORMAL)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            if self._client:
                try:
                    self._client.disconnect()
                except:
                    pass


def main() -> None:
    app = TelegramViewerApp()
    app.run()


if __name__ == "__main__":
    main()
