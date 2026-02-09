"""
Analytics panel UI for Telegram viewer.
Displays equipment losses, casualties, and statistics with matplotlib charts.
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from analytics_extractor import AnalyticsExtractor, StatsAggregator
from analytics_models import (
    AggregatedStats,
    EquipmentCategory,
    Side,
)

if TYPE_CHECKING:
    from telegram_client import TelegramMessage


# Colors for sides
SIDE_COLORS = {
    Side.RUSSIAN: "#e74c3c",  # Red
    Side.UKRAINIAN: "#3498db",  # Blue
    Side.UNKNOWN: "#95a5a6",  # Gray
}

# Category display names
CATEGORY_NAMES = {
    EquipmentCategory.TANK: "Tanks",
    EquipmentCategory.IFV: "IFVs",
    EquipmentCategory.APC: "APCs",
    EquipmentCategory.ARTILLERY: "Artillery",
    EquipmentCategory.MLRS: "MLRS",
    EquipmentCategory.HELICOPTER: "Helicopters",
    EquipmentCategory.AIRCRAFT: "Aircraft",
    EquipmentCategory.UAV: "UAVs/Drones",
    EquipmentCategory.AIR_DEFENSE: "Air Defense",
    EquipmentCategory.LOGISTICS: "Logistics",
    EquipmentCategory.NAVAL: "Naval",
    EquipmentCategory.OTHER: "Other",
}


class AnalyticsPanel:
    """Right-side panel showing analytics with charts."""

    def __init__(self, parent: tk.Frame, get_data_callback):
        """
        Initialize analytics panel.

        Args:
            parent: Parent Tkinter frame
            get_data_callback: Callable that returns (messages, translations, media, image_descriptions)
        """
        self.parent = parent
        self.get_data_callback = get_data_callback

        self._extractor = AnalyticsExtractor()
        self._aggregator = StatsAggregator()
        self._stats: AggregatedStats | None = None

        self._analyzing = False
        self._result_queue: queue.Queue = queue.Queue()

        self._build_ui()
        self._poll_queue()

    def _build_ui(self) -> None:
        """Build the analytics panel UI."""
        # Main scrollable container
        canvas = tk.Canvas(self.parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.parent, orient="vertical", command=canvas.yview)
        self._scroll_frame = ttk.Frame(canvas)

        self._scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Content
        content = self._scroll_frame
        content.columnconfigure(0, weight=1)

        # Title
        title = ttk.Label(content, text="Analytics", font=("TkDefaultFont", 12, "bold"))
        title.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        # Controls frame
        ctrl_frame = ttk.Frame(content)
        ctrl_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        self._analyze_btn = ttk.Button(
            ctrl_frame, text="Analyze", command=self._on_analyze, width=10
        )
        self._analyze_btn.pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(ctrl_frame, text="Clear", command=self._on_clear, width=8).pack(
            side=tk.LEFT, padx=(0, 5)
        )

        # Progress bar
        self._progress_var = tk.DoubleVar(value=0)
        self._progress = ttk.Progressbar(
            ctrl_frame, variable=self._progress_var, maximum=100, length=100
        )
        self._progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # Status label
        self._status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(content, textvariable=self._status_var)
        status_label.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 5))

        # Summary frame
        summary_frame = ttk.LabelFrame(content, text="Summary")
        summary_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        summary_frame.columnconfigure(1, weight=1)

        # Summary labels
        labels = [
            ("Messages loaded:", "_msg_loaded_var"),
            ("Messages analyzed:", "_msg_analyzed_var"),
            ("Equipment losses:", "_equip_losses_var"),
            ("Casualty reports:", "_casualty_reports_var"),
        ]
        for i, (text, var_name) in enumerate(labels):
            ttk.Label(summary_frame, text=text).grid(
                row=i, column=0, sticky="w", padx=5, pady=2
            )
            var = tk.StringVar(value="0")
            setattr(self, var_name, var)
            ttk.Label(summary_frame, textvariable=var, font=("TkDefaultFont", 10, "bold")).grid(
                row=i, column=1, sticky="e", padx=5, pady=2
            )

        # Equipment chart frame
        equip_frame = ttk.LabelFrame(content, text="Equipment Losses by Category")
        equip_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=5)

        self._equip_fig = Figure(figsize=(4, 3), dpi=80)
        self._equip_ax = self._equip_fig.add_subplot(111)
        self._equip_canvas = FigureCanvasTkAgg(self._equip_fig, equip_frame)
        self._equip_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # By side chart frame
        side_frame = ttk.LabelFrame(content, text="Losses by Side")
        side_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=5)

        self._side_fig = Figure(figsize=(4, 2.5), dpi=80)
        self._side_ax = self._side_fig.add_subplot(111)
        self._side_canvas = FigureCanvasTkAgg(self._side_fig, side_frame)
        self._side_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Casualties frame
        cas_frame = ttk.LabelFrame(content, text="Casualties")
        cas_frame.grid(row=6, column=0, sticky="ew", padx=10, pady=5)
        cas_frame.columnconfigure(1, weight=1)

        # Casualty labels by side
        self._cas_ru_var = tk.StringVar(value="Russian: --")
        self._cas_ua_var = tk.StringVar(value="Ukrainian: --")

        ttk.Label(cas_frame, textvariable=self._cas_ru_var, foreground="#e74c3c").grid(
            row=0, column=0, sticky="w", padx=10, pady=5
        )
        ttk.Label(cas_frame, textvariable=self._cas_ua_var, foreground="#3498db").grid(
            row=1, column=0, sticky="w", padx=10, pady=5
        )

        # Initialize empty charts
        self._update_charts()

    def _on_analyze(self) -> None:
        """Start analysis of loaded messages."""
        if self._analyzing:
            return

        # Get data from main app
        try:
            messages, translations, media, image_descs = self.get_data_callback()
        except Exception:
            self._status_var.set("Error: Could not get data")
            return

        if not messages:
            self._status_var.set("No messages loaded")
            return

        self._analyzing = True
        self._analyze_btn.config(state="disabled")
        self._status_var.set("Analyzing...")
        self._progress_var.set(0)

        # Run analysis in background thread
        def analyze():
            try:
                # Build message tuples: (msg_id, text, img_desc, timestamp)
                to_analyze = []
                for i, msg in enumerate(messages):
                    text = translations[i] if i < len(translations) and translations[i] else msg.text
                    img_desc = image_descs.get(msg.id, "")
                    to_analyze.append((msg.id, text, img_desc, msg.timestamp))

                def progress_cb(done, total):
                    pct = (done / total) * 100 if total > 0 else 0
                    self._result_queue.put(("progress", pct, f"Analyzing {done}/{total}..."))

                results = self._extractor.extract_batch(to_analyze, progress_cb)
                stats = self._aggregator.aggregate(results, len(messages))
                self._result_queue.put(("done", stats))

            except Exception as e:
                self._result_queue.put(("error", str(e)))

        threading.Thread(target=analyze, daemon=True).start()

    def _on_clear(self) -> None:
        """Clear analytics results."""
        self._extractor.clear_cache()
        self._stats = None
        self._update_summary()
        self._update_charts()
        self._status_var.set("Cleared")

    def _poll_queue(self) -> None:
        """Poll result queue for updates."""
        try:
            while True:
                item = self._result_queue.get_nowait()
                if item[0] == "progress":
                    _, pct, msg = item
                    self._progress_var.set(pct)
                    self._status_var.set(msg)
                elif item[0] == "done":
                    _, stats = item
                    self._stats = stats
                    self._analyzing = False
                    self._analyze_btn.config(state="normal")
                    self._progress_var.set(100)
                    self._status_var.set(f"Done - {stats.analyzed_messages} messages analyzed")
                    self._update_summary()
                    self._update_charts()
                elif item[0] == "error":
                    _, err = item
                    self._analyzing = False
                    self._analyze_btn.config(state="normal")
                    self._status_var.set(f"Error: {err}")
        except queue.Empty:
            pass

        self.parent.after(100, self._poll_queue)

    def _update_summary(self) -> None:
        """Update summary labels."""
        if not self._stats:
            self._msg_loaded_var.set("0")
            self._msg_analyzed_var.set("0")
            self._equip_losses_var.set("0")
            self._casualty_reports_var.set("0")
            self._cas_ru_var.set("Russian: --")
            self._cas_ua_var.set("Ukrainian: --")
            return

        s = self._stats
        self._msg_loaded_var.set(str(s.total_messages))
        self._msg_analyzed_var.set(str(s.analyzed_messages))

        # Count total equipment losses
        total_equip = sum(
            sum(side_counts.values())
            for side_counts in s.equipment_by_category.values()
        )
        self._equip_losses_var.set(str(total_equip))

        # Count casualty reports (non-zero)
        cas_count = sum(
            1 for side_data in s.casualties_by_side.values()
            if sum(side_data.values()) > 0
        )
        self._casualty_reports_var.set(str(cas_count))

        # Casualty details by side
        ru_cas = s.casualties_by_side.get(Side.RUSSIAN, {})
        ua_cas = s.casualties_by_side.get(Side.UKRAINIAN, {})

        ru_text = f"Russian: {ru_cas.get('killed', 0)} KIA, {ru_cas.get('wounded', 0)} WIA, {ru_cas.get('captured', 0)} POW"
        ua_text = f"Ukrainian: {ua_cas.get('killed', 0)} KIA, {ua_cas.get('wounded', 0)} WIA, {ua_cas.get('captured', 0)} POW"

        self._cas_ru_var.set(ru_text)
        self._cas_ua_var.set(ua_text)

    def _update_charts(self) -> None:
        """Update matplotlib charts."""
        # Equipment by category chart
        self._equip_ax.clear()

        if self._stats and self._stats.equipment_by_category:
            categories = []
            ru_counts = []
            ua_counts = []

            for cat in EquipmentCategory:
                if cat in self._stats.equipment_by_category:
                    side_data = self._stats.equipment_by_category[cat]
                    ru = side_data.get(Side.RUSSIAN, 0)
                    ua = side_data.get(Side.UKRAINIAN, 0)
                    if ru > 0 or ua > 0:
                        categories.append(CATEGORY_NAMES.get(cat, cat.value))
                        ru_counts.append(ru)
                        ua_counts.append(ua)

            if categories:
                x = range(len(categories))
                width = 0.35

                self._equip_ax.barh(
                    [i - width / 2 for i in x],
                    ru_counts,
                    width,
                    label="Russian",
                    color=SIDE_COLORS[Side.RUSSIAN],
                )
                self._equip_ax.barh(
                    [i + width / 2 for i in x],
                    ua_counts,
                    width,
                    label="Ukrainian",
                    color=SIDE_COLORS[Side.UKRAINIAN],
                )

                self._equip_ax.set_yticks(list(x))
                self._equip_ax.set_yticklabels(categories, fontsize=8)
                self._equip_ax.legend(loc="lower right", fontsize=8)
                self._equip_ax.set_xlabel("Count", fontsize=8)
            else:
                self._equip_ax.text(
                    0.5, 0.5, "No data", ha="center", va="center", fontsize=10
                )
                self._equip_ax.set_xlim(0, 1)
                self._equip_ax.set_ylim(0, 1)
        else:
            self._equip_ax.text(
                0.5, 0.5, "No data", ha="center", va="center", fontsize=10
            )
            self._equip_ax.set_xlim(0, 1)
            self._equip_ax.set_ylim(0, 1)

        self._equip_ax.tick_params(axis="both", labelsize=8)
        self._equip_fig.tight_layout()
        self._equip_canvas.draw()

        # By side pie/bar chart
        self._side_ax.clear()

        if self._stats and self._stats.equipment_by_category:
            total_ru = sum(
                side_data.get(Side.RUSSIAN, 0)
                for side_data in self._stats.equipment_by_category.values()
            )
            total_ua = sum(
                side_data.get(Side.UKRAINIAN, 0)
                for side_data in self._stats.equipment_by_category.values()
            )
            total_unk = sum(
                side_data.get(Side.UNKNOWN, 0)
                for side_data in self._stats.equipment_by_category.values()
            )

            if total_ru > 0 or total_ua > 0 or total_unk > 0:
                labels = []
                sizes = []
                colors = []

                if total_ru > 0:
                    labels.append(f"Russian ({total_ru})")
                    sizes.append(total_ru)
                    colors.append(SIDE_COLORS[Side.RUSSIAN])
                if total_ua > 0:
                    labels.append(f"Ukrainian ({total_ua})")
                    sizes.append(total_ua)
                    colors.append(SIDE_COLORS[Side.UKRAINIAN])
                if total_unk > 0:
                    labels.append(f"Unknown ({total_unk})")
                    sizes.append(total_unk)
                    colors.append(SIDE_COLORS[Side.UNKNOWN])

                self._side_ax.pie(
                    sizes,
                    labels=labels,
                    colors=colors,
                    autopct="%1.0f%%",
                    startangle=90,
                    textprops={"fontsize": 8},
                )
            else:
                self._side_ax.text(
                    0.5, 0.5, "No data", ha="center", va="center", fontsize=10
                )
        else:
            self._side_ax.text(
                0.5, 0.5, "No data", ha="center", va="center", fontsize=10
            )

        self._side_fig.tight_layout()
        self._side_canvas.draw()
