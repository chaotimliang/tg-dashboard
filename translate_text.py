"""Translate Russian text to English for Telegram viewer.
Supports: Argos Translate (fast), MarianMT (offline), Ollama (LLM), deep-translator (online)."""
from __future__ import annotations

import time

import requests

# Chunk size to stay under Google Translate limit (~5000 chars)
_CHUNK_SIZE = 4500
# MarianMT typical max length (tokens ~512); chunk by chars to be safe
_MARIAN_CHUNK_CHARS = 400
# Delay between chunks to reduce rate-limit / blocking (seconds)
_CHUNK_DELAY = 0.3
# Ollama base URL; try Gemma first, then fallback (run: ollama run gemma2)
OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODELS = ("gemma2", "gemma2:2b", "llama3.2")  # try in order
OLLAMA_TIMEOUT = 120

# Argos Translate cache
_argos_translator_cache: dict[str, object] = {}
_argos_installed_langs: set[str] = set()


def _chunk_text(text: str, max_len: int = _CHUNK_SIZE) -> list[str]:
    """Split text into chunks on paragraph/sentence boundaries where possible."""
    text = (text or "").strip()
    if not text or len(text) <= max_len:
        return [text] if text else []
    chunks = []
    rest = text
    while rest:
        if len(rest) <= max_len:
            chunks.append(rest)
            break
        block = rest[:max_len]
        for sep in ("\n\n", "\n", ". ", "!", "? ", " "):
            idx = block.rfind(sep)
            if idx > max_len // 2:
                block = block[: idx + len(sep)].rstrip()
                rest = rest[len(block) :].lstrip()
                break
        else:
            rest = rest[max_len:]
            block = block.rstrip()
        if block:
            chunks.append(block)
    return chunks


def _translate_chunks(translator, chunks: list[str]) -> str:
    if not chunks:
        return ""
    if len(chunks) == 1:
        out = translator.translate(chunks[0])
        return (out or "").strip()
    parts = []
    for i, c in enumerate(chunks):
        if i > 0:
            time.sleep(_CHUNK_DELAY)
        t = translator.translate(c)
        if t:
            parts.append(t.strip())
    return "\n\n".join(parts).strip()


_LANG_CODE_TO_NAME = {
    "ru": "Russian",
    "uk": "Ukrainian",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
}


def to_english(raw: str, source_lang: str = "ru", prefer_local: bool = True) -> str:
    """Translate text to English. Tries local Ollama/Gemma first if prefer_local, then web services."""
    raw = (raw or "").strip()
    if not raw:
        return ""

    # 1. Try local Ollama/Gemma first (lightweight, offline)
    if prefer_local:
        lang_name = _LANG_CODE_TO_NAME.get(source_lang, source_lang)
        out = _translate_via_ollama(raw, source_lang=lang_name)
        if out:
            return out

    # 2. Online fallback: Google Translate
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        return raw

    chunks = _chunk_text(raw)
    if not chunks:
        return ""

    # Try Google with requested source
    try:
        translator = GoogleTranslator(source=source_lang, target="en")
        out = _translate_chunks(translator, chunks)
        if out:
            return out
    except Exception:
        pass

    # Fallback: auto-detect language (often more reliable when Google blocks direct 'ru')
    try:
        translator = GoogleTranslator(source="auto", target="en")
        out = _translate_chunks(translator, chunks)
        if out:
            return out
    except Exception:
        pass

    # Fallback: MyMemory (free, no key; may have lower limits)
    try:
        from deep_translator import MyMemoryTranslator
        translator = MyMemoryTranslator(source=source_lang, target="en")
        out = _translate_chunks(translator, chunks)
        if out:
            return out
    except Exception:
        pass
    try:
        from deep_translator import MyMemoryTranslator
        translator = MyMemoryTranslator(source="auto", target="en")
        out = _translate_chunks(translator, chunks)
        if out:
            return out
    except Exception:
        pass

    return raw


def _translate_via_ollama(text: str, source_lang: str = "Russian") -> str | None:
    """Use local Ollama LLM to translate text → English. Tries Gemma then fallbacks. Returns None if Ollama not running."""
    text = (text or "").strip()
    if not text:
        return ""
    prompt = (
        f"Translate the following {source_lang} text to English. "
        "Output only the translation, no explanation or preamble.\n\n"
        + text
    )
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
                return response
        except Exception:
            continue
    return None


def _translate_via_marian(text: str) -> str | None:
    """Use Helsinki-NLP opus-mt-ru-en locally (transformers). Returns None if not installed."""
    text = (text or "").strip()
    if not text:
        return ""
    try:
        from transformers import pipeline
    except ImportError:
        return None
    try:
        # Load once and reuse (module-level cache below)
        if _translate_via_marian._pipe is None:
            _translate_via_marian._pipe = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-ru-en",
                device=-1,
            )
        pipe = _translate_via_marian._pipe
    except Exception:
        return None
    # Marian has max length; chunk by rough char count
    chunks = _chunk_text(text, max_len=_MARIAN_CHUNK_CHARS)
    if not chunks:
        return ""
    try:
        if len(chunks) == 1:
            out = pipe(chunks[0])
            return (out[0]["translation_text"] or "").strip() if out else None
        parts = []
        for c in chunks:
            out = pipe(c)
            if out and out[0].get("translation_text"):
                parts.append(out[0]["translation_text"].strip())
        return "\n\n".join(parts).strip() if parts else None
    except Exception:
        return None


_translate_via_marian._pipe = None  # type: ignore[attr-defined]


def _ensure_argos_language(from_code: str, to_code: str = "en") -> bool:
    """Ensure Argos language pack is installed. Returns True if available."""
    global _argos_installed_langs
    key = f"{from_code}-{to_code}"
    if key in _argos_installed_langs:
        return True

    try:
        import argostranslate.package
        import argostranslate.translate

        # Check if already installed
        installed = argostranslate.translate.get_installed_languages()
        from_lang = next((l for l in installed if l.code == from_code), None)
        to_lang = next((l for l in installed if l.code == to_code), None)

        if from_lang and to_lang:
            translation = from_lang.get_translation(to_lang)
            if translation:
                _argos_installed_langs.add(key)
                return True

        # Need to download
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        pkg = next(
            (p for p in available if p.from_code == from_code and p.to_code == to_code),
            None
        )
        if pkg:
            argostranslate.package.install_from_path(pkg.download())
            _argos_installed_langs.add(key)
            return True
    except Exception:
        pass
    return False


def _translate_via_argos(text: str, from_code: str = "ru", to_code: str = "en") -> str | None:
    """Use Argos Translate (fast, offline). Returns None if not available."""
    text = (text or "").strip()
    if not text:
        return ""

    try:
        import argostranslate.translate

        # Ensure language is installed
        if not _ensure_argos_language(from_code, to_code):
            return None

        # Get cached translator or create new
        cache_key = f"{from_code}-{to_code}"
        if cache_key not in _argos_translator_cache:
            installed = argostranslate.translate.get_installed_languages()
            from_lang = next((l for l in installed if l.code == from_code), None)
            to_lang = next((l for l in installed if l.code == to_code), None)
            if from_lang and to_lang:
                _argos_translator_cache[cache_key] = from_lang.get_translation(to_lang)

        translator = _argos_translator_cache.get(cache_key)
        if translator:
            return translator.translate(text)
    except Exception:
        pass
    return None


def to_english_from_russian(raw: str, prefer_local: bool = True, use_fast: bool = True) -> str:
    """Translate Russian text to English.

    Args:
        raw: Text to translate
        prefer_local: Try local translation first
        use_fast: Use fast Argos Translate (recommended)
    """
    raw = (raw or "").strip()
    if not raw:
        return ""

    if prefer_local:
        # 1. Argos Translate (FAST - ~0.1s per message)
        if use_fast:
            out = _translate_via_argos(raw, from_code="ru", to_code="en")
            if out:
                return out

        # 2. Local MarianMT (offline, pip install transformers torch)
        out = _translate_via_marian(raw)
        if out:
            return out

        # 3. Local Ollama/Gemma (slower but higher quality)
        out = _translate_via_ollama(raw, source_lang="Russian")
        if out:
            return out

    # 4. Online fallback (Google / MyMemory)
    return to_english(raw, source_lang="ru")
