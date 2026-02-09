"""
AI-powered analytics extraction from Telegram messages.
Uses Ollama LLM to extract equipment losses, casualties, and other intelligence.
"""
from __future__ import annotations

import concurrent.futures
import json
import re
import threading
from collections import defaultdict
from datetime import datetime
from typing import Callable, Optional

import requests

from analytics_models import (
    AnalyticsResult,
    AggregatedStats,
    CasualtyReport,
    EquipmentCategory,
    EquipmentLoss,
    OutcomeType,
    Side,
)

# Import Ollama config from translate_text
try:
    from translate_text import OLLAMA_BASE, OLLAMA_TIMEOUT
except ImportError:
    OLLAMA_BASE = "http://localhost:11434"
    OLLAMA_TIMEOUT = 120

# Models to try for extraction (in order of preference)
EXTRACTION_MODELS = ("gemma2", "llama3.2", "mistral", "gemma2:2b")

# Number of parallel workers for batch extraction
EXTRACTION_WORKERS = 4

# Extraction prompt template
EXTRACTION_PROMPT = """Analyze this military-related message and extract structured information.

MESSAGE TEXT:
{text}

{image_section}

Extract information as JSON with this EXACT structure:
{{
  "equipment_losses": [
    {{
      "type": "specific name (e.g., T-72B3, Ka-52, BMP-2)",
      "category": "tank|ifv|apc|artillery|mlrs|helicopter|aircraft|uav|air_defense|logistics|naval|other",
      "side": "russian|ukrainian|unknown",
      "outcome": "destroyed|damaged|captured|abandoned|unknown",
      "count": 1,
      "confidence": 0.8
    }}
  ],
  "casualties": {{
    "side": "russian|ukrainian|unknown",
    "killed": 0,
    "wounded": 0,
    "captured": 0,
    "unit": ""
  }},
  "location": "",
  "is_victory_claim": false,
  "is_loss_admission": false
}}

RULES:
1. Only extract explicitly stated or clearly shown information
2. Use "unknown" for side if not determinable
3. For Russian pro-war channels, "enemy" usually means Ukrainian forces
4. Set confidence 0.7-1.0 for explicit statements, 0.3-0.6 for inferred
5. Extract specific equipment models when identifiable
6. Return empty arrays/nulls/zeros if no relevant information
7. Output ONLY valid JSON, no explanation

JSON:"""


class AnalyticsExtractor:
    """Extract analytics from messages using Ollama LLM."""

    def __init__(self):
        self._cache: dict[int, AnalyticsResult] = {}
        self._cache_lock = threading.Lock()
        self._available_model: str | None = None

    def _find_available_model(self) -> str | None:
        """Find first available Ollama model."""
        if self._available_model:
            return self._available_model

        for model in EXTRACTION_MODELS:
            try:
                r = requests.post(
                    f"{OLLAMA_BASE}/api/generate",
                    json={"model": model, "prompt": "test", "stream": False},
                    timeout=10,
                )
                if r.status_code == 200:
                    self._available_model = model
                    return model
            except Exception:
                continue
        return None

    def _call_ollama(self, prompt: str) -> str | None:
        """Call Ollama API with the extraction prompt."""
        model = self._find_available_model()
        if not model:
            return None

        try:
            r = requests.post(
                f"{OLLAMA_BASE}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=OLLAMA_TIMEOUT,
            )
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception:
            return None

    def _parse_response(
        self, response: str, msg_id: int, timestamp: datetime | None
    ) -> AnalyticsResult | None:
        """Parse LLM JSON response into AnalyticsResult."""
        if not response:
            return None

        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r"\{[\s\S]*\}", response)
            if not json_match:
                return None

            data = json.loads(json_match.group())

            # Parse equipment losses
            equipment_losses = []
            for eq in data.get("equipment_losses", []):
                if not eq.get("type"):
                    continue
                try:
                    category_str = eq.get("category", "other").upper()
                    if category_str not in EquipmentCategory.__members__:
                        category_str = "OTHER"

                    side_str = eq.get("side", "unknown").upper()
                    if side_str not in Side.__members__:
                        side_str = "UNKNOWN"

                    outcome_str = eq.get("outcome", "unknown").upper()
                    if outcome_str not in OutcomeType.__members__:
                        outcome_str = "UNKNOWN"

                    loss = EquipmentLoss(
                        equipment_type=eq.get("type", "Unknown"),
                        category=EquipmentCategory[category_str],
                        side=Side[side_str],
                        outcome=OutcomeType[outcome_str],
                        count=max(1, int(eq.get("count", 1))),
                        confidence=float(eq.get("confidence", 0.5)),
                        source_msg_id=msg_id,
                        timestamp=timestamp,
                        location=data.get("location", ""),
                    )
                    equipment_losses.append(loss)
                except (KeyError, ValueError):
                    continue

            # Parse casualties
            casualties = []
            cas_data = data.get("casualties", {})
            if cas_data and isinstance(cas_data, dict):
                killed = cas_data.get("killed") or 0
                wounded = cas_data.get("wounded") or 0
                captured = cas_data.get("captured") or 0

                if killed or wounded or captured:
                    side_str = cas_data.get("side", "unknown").upper()
                    if side_str not in Side.__members__:
                        side_str = "UNKNOWN"

                    casualty = CasualtyReport(
                        side=Side[side_str],
                        killed=int(killed),
                        wounded=int(wounded),
                        captured=int(captured),
                        confidence=0.6,
                        source_msg_id=msg_id,
                        timestamp=timestamp,
                        unit=cas_data.get("unit", ""),
                    )
                    casualties.append(casualty)

            return AnalyticsResult(
                msg_id=msg_id,
                equipment_losses=equipment_losses,
                casualties=casualties,
                location=data.get("location", ""),
                is_victory_claim=bool(data.get("is_victory_claim", False)),
                is_loss_admission=bool(data.get("is_loss_admission", False)),
                raw_ai_response=response,
                processed_at=datetime.now(),
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return None

    def extract_single(
        self,
        msg_id: int,
        text: str,
        image_description: str = "",
        timestamp: datetime | None = None,
        force: bool = False,
    ) -> AnalyticsResult | None:
        """Extract analytics from a single message."""
        # Check cache
        with self._cache_lock:
            if not force and msg_id in self._cache:
                return self._cache[msg_id]

        # Skip empty messages
        if not text and not image_description:
            return None

        # Build prompt
        image_section = ""
        if image_description:
            image_section = f"\nIMAGE DESCRIPTION:\n{image_description}\n"

        prompt = EXTRACTION_PROMPT.format(text=text, image_section=image_section)

        # Call LLM
        response = self._call_ollama(prompt)
        result = self._parse_response(response, msg_id, timestamp)

        # Cache result
        if result:
            with self._cache_lock:
                self._cache[msg_id] = result

        return result

    def extract_batch(
        self,
        messages: list[tuple[int, str, str, datetime | None]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[int, AnalyticsResult]:
        """
        Extract analytics from multiple messages in parallel.

        Args:
            messages: List of (msg_id, text, image_description, timestamp)
            progress_callback: Optional callback(completed, total)

        Returns:
            Dict mapping msg_id to AnalyticsResult
        """
        results: dict[int, AnalyticsResult] = {}
        total = len(messages)

        if not messages:
            return results

        def process_one(item: tuple) -> tuple[int, AnalyticsResult | None]:
            msg_id, text, img_desc, ts = item
            result = self.extract_single(msg_id, text, img_desc, ts)
            return msg_id, result

        completed = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=EXTRACTION_WORKERS
        ) as executor:
            futures = {executor.submit(process_one, m): m for m in messages}

            for future in concurrent.futures.as_completed(futures):
                completed += 1
                try:
                    msg_id, result = future.result()
                    if result:
                        results[msg_id] = result
                except Exception:
                    pass

                if progress_callback:
                    progress_callback(completed, total)

        return results

    def get_cached_results(self) -> dict[int, AnalyticsResult]:
        """Get all cached results."""
        with self._cache_lock:
            return self._cache.copy()

    def clear_cache(self) -> None:
        """Clear the results cache."""
        with self._cache_lock:
            self._cache.clear()


class StatsAggregator:
    """Aggregate analytics results into statistics."""

    def aggregate(
        self, results: dict[int, AnalyticsResult], total_messages: int = 0
    ) -> AggregatedStats:
        """Create aggregated statistics from results."""
        stats = AggregatedStats(
            total_messages=total_messages,
            analyzed_messages=len(results),
        )

        # Equipment by category
        equipment_by_category: dict[EquipmentCategory, dict[Side, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        equipment_by_type: dict[str, dict[Side, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Casualties by side
        casualties_by_side: dict[Side, dict[str, int]] = defaultdict(
            lambda: {"killed": 0, "wounded": 0, "captured": 0}
        )

        for result in results.values():
            # Count equipment
            for eq in result.equipment_losses:
                equipment_by_category[eq.category][eq.side] += eq.count
                equipment_by_type[eq.equipment_type][eq.side] += eq.count

            # Count casualties
            for cas in result.casualties:
                casualties_by_side[cas.side]["killed"] += cas.killed
                casualties_by_side[cas.side]["wounded"] += cas.wounded
                casualties_by_side[cas.side]["captured"] += cas.captured

            # Count claims
            if result.is_victory_claim:
                stats.victory_claims += 1
            if result.is_loss_admission:
                stats.loss_admissions += 1

        stats.equipment_by_category = dict(equipment_by_category)
        stats.equipment_by_type = dict(equipment_by_type)
        stats.casualties_by_side = dict(casualties_by_side)

        return stats
