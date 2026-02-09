from __future__ import annotations

import time
from collections import deque

import requests
import streamlit as st

from scraper import ScrapedItem, load_config, scrape_once


st.set_page_config(page_title="Live Scraper Viewer", layout="wide")


@st.cache_resource
def _session() -> requests.Session:
    s = requests.Session()
    return s


def _render_item(item: ScrapedItem) -> None:
    if item.title:
        st.markdown(f"### {item.title}")
    if item.text:
        st.write(item.text)
    if item.image_urls:
        # Streamlit can display URLs directly; keep it simple.
        st.image(list(item.image_urls), use_container_width=True)
    st.caption(f"key: `{item.key}`")


st.title("Live Scraper Viewer")

cfg_path = st.sidebar.text_input("Config path", value="config.yml")
auto_refresh = st.sidebar.toggle("Auto refresh", value=True)
refresh_s = st.sidebar.number_input("Refresh seconds", min_value=1, max_value=3600, value=10, step=1)
max_items = st.sidebar.number_input("Max items on screen", min_value=10, max_value=1000, value=100, step=10)

col_a, col_b = st.columns([1, 1])

with col_a:
    if st.button("Scrape now"):
        st.session_state["force_scrape"] = True

with col_b:
    if st.button("Clear seen"):
        st.session_state["seen"] = set()
        st.session_state["items"] = deque(maxlen=int(max_items))

if "seen" not in st.session_state:
    st.session_state["seen"] = set()
if "items" not in st.session_state:
    st.session_state["items"] = deque(maxlen=int(max_items))

cfg = load_config(cfg_path)
cfg.setdefault("refresh", {})
cfg["refresh"]["interval_seconds"] = int(refresh_s)

status = st.empty()
container = st.container()

do_scrape = bool(st.session_state.pop("force_scrape", False)) or auto_refresh

if do_scrape:
    try:
        scraped = scrape_once(cfg)
        new_count = 0
        for it in scraped:
            if it.key in st.session_state["seen"]:
                continue
            st.session_state["seen"].add(it.key)
            st.session_state["items"].appendleft(it)
            new_count += 1

        status.info(f"Fetched {len(scraped)} items, added {new_count}. Showing {len(st.session_state['items'])}.")
    except Exception as e:
        status.error(f"Scrape failed: {e}")

with container:
    for it in list(st.session_state["items"])[: int(max_items)]:
        _render_item(it)
        st.divider()

# Simple live loop: rerun after N seconds
if auto_refresh:
    time.sleep(int(refresh_s))
    st.rerun()

