"""GPT-4o Vision-Language Model perception for the Go2 quadruped robot.

Calls the OpenRouter API (https://openrouter.ai/api/v1) with model
openai/gpt-4o to analyze camera frames.  No ROS2 dependency — pure Python.

Typical usage::

    from vector_os_nano.perception.vlm_go2 import Go2VLMPerception

    vlm = Go2VLMPerception()                       # key from env
    vlm = Go2VLMPerception(config={"api_key": ".."})  # key from config

    scene = vlm.describe_scene(rgb_frame)
    room  = vlm.identify_room(rgb_frame)
    objs  = vlm.find_objects(rgb_frame, query="chair")
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import httpx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost model — GPT-4o via OpenRouter (as of 2026-04)
# ---------------------------------------------------------------------------
_COST_PER_INPUT_TOKEN: float = 2.50 / 1_000_000   # USD per input token
_COST_PER_OUTPUT_TOKEN: float = 10.00 / 1_000_000  # USD per output token

_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
_MODEL: str = "openai/gpt-4o"
_TIMEOUT_S: float = 30.0
_MAX_RETRIES: int = 3
_JPEG_QUALITY: int = 85


# ---------------------------------------------------------------------------
# Result dataclasses (immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectedObject:
    """A single object identified by the VLM."""

    name: str
    description: str
    confidence: float  # 0.0 – 1.0


@dataclass(frozen=True)
class SceneDescription:
    """Full scene analysis returned by describe_scene()."""

    summary: str                    # One-line summary
    objects: list[DetectedObject]
    room_type: str                  # e.g. "kitchen", "bedroom", "hallway"
    details: str                    # Full free-form description


@dataclass(frozen=True)
class RoomIdentification:
    """Room classification returned by identify_room()."""

    room: str           # Canonical room name (kitchen, living_room, …)
    confidence: float   # 0.0 – 1.0
    reasoning: str      # Model's rationale


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_DESCRIBE_SCENE_PROMPT: str = (
    "You are a robot dog's vision system (camera at 30cm height). "
    "Describe what you see in this first-person camera view from inside a house. "
    "Focus on: furniture (sofas, tables, beds, desks), appliances, floor type, "
    "wall colors, and any distinctive objects. These help identify which room this is. "
    'Respond ONLY in JSON: {"summary": "...", "objects": [{"name": "...", '
    '"description": "...", "confidence": 0.9}], "room_type": "...", "details": "..."}'
)

_IDENTIFY_ROOM_PROMPT: str = (
    "You are a robot dog (30cm tall) navigating a house. "
    "Based on this first-person camera view, identify which room you are in. "
    "Room features to look for:\n"
    "- living_room: blue sofa, coffee table, TV, rug, bookshelves\n"
    "- dining_room: dining table with chairs, pendant light\n"
    "- kitchen: counter, island, fridge (steel), range hood\n"
    "- study: desk with monitor, office chair, bookshelf\n"
    "- master_bedroom: large bed with frame, wardrobe, nightstands\n"
    "- guest_bedroom: bed, dresser\n"
    "- bathroom: bathtub, vanity, small tiles on floor\n"
    "- hallway: open corridor, no major furniture, connecting space\n\n"
    "Look at furniture, floor texture, colors, and objects to decide. "
    'Respond ONLY in JSON: {"room": "...", "confidence": 0.9, "reasoning": "..."}'
)

_FIND_OBJECTS_PROMPT_BASE: str = (
    "List all objects visible in this robot camera view"
)
_FIND_OBJECTS_PROMPT_SUFFIX: str = (
    '. Respond in JSON: {"objects": [{"name": "...", "description": "...", '
    '"confidence": 0.0}]}'
)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class Go2VLMPerception:
    """Vision-Language Model perception interface for the Go2 robot.

    Thread-safe.  Multiple threads may call describe_scene / identify_room /
    find_objects concurrently.  Cumulative cost is tracked under a lock.

    Args:
        config: Optional dict with keys ``api_key`` (str).
                Falls back to environment variable OPENROUTER_API_KEY.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        api_key: str | None = cfg.get("api_key") or os.environ.get(
            "OPENROUTER_API_KEY"
        )
        if not api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Pass config={'api_key': '...'} or set OPENROUTER_API_KEY."
            )
        self._api_key: str = api_key
        self._cost_usd: float = 0.0
        self._cost_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def describe_scene(self, frame: np.ndarray) -> SceneDescription:
        """Analyze a camera frame and return a full scene description.

        Args:
            frame: (H, W, 3) uint8 RGB array.

        Returns:
            SceneDescription with summary, objects, room_type, and details.
        """
        raw = self._call_vlm(frame, _DESCRIBE_SCENE_PROMPT)
        data = _parse_json_response(raw)

        raw_objects: list[dict[str, Any]] = data.get("objects") or []
        objects = [_parse_detected_object(o) for o in raw_objects]

        return SceneDescription(
            summary=str(data.get("summary", "")),
            objects=objects,
            room_type=str(data.get("room_type", "unknown")),
            details=str(data.get("details", "")),
        )

    def identify_room(self, frame: np.ndarray) -> RoomIdentification:
        """Identify which room the robot is currently in.

        Args:
            frame: (H, W, 3) uint8 RGB array.

        Returns:
            RoomIdentification with room name, confidence, and reasoning.
        """
        raw = self._call_vlm(frame, _IDENTIFY_ROOM_PROMPT)
        data = _parse_json_response(raw)

        return RoomIdentification(
            room=str(data.get("room", "unknown")),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=str(data.get("reasoning", "")),
        )

    def find_objects(
        self, frame: np.ndarray, query: str | None = None
    ) -> list[DetectedObject]:
        """List objects visible in the camera frame.

        Args:
            frame: (H, W, 3) uint8 RGB array.
            query: Optional filter string, e.g. ``"chair"`` or ``"red objects"``.
                   If omitted, all visible objects are returned.

        Returns:
            List of DetectedObject sorted by descending confidence.
        """
        if query:
            prompt = (
                f"{_FIND_OBJECTS_PROMPT_BASE} that match: {query}"
                f"{_FIND_OBJECTS_PROMPT_SUFFIX}"
            )
        else:
            prompt = _FIND_OBJECTS_PROMPT_BASE + _FIND_OBJECTS_PROMPT_SUFFIX

        raw = self._call_vlm(frame, prompt)
        data = _parse_json_response(raw)

        raw_objects: list[dict[str, Any]] = data.get("objects") or []
        objects = [_parse_detected_object(o) for o in raw_objects]
        return sorted(objects, key=lambda o: o.confidence, reverse=True)

    # ------------------------------------------------------------------
    # Cost tracking
    # ------------------------------------------------------------------

    @property
    def cumulative_cost_usd(self) -> float:
        """Total USD spent on API calls since this instance was created."""
        with self._cost_lock:
            return self._cost_usd

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode a numpy RGB frame as a base64 JPEG string.

        Args:
            frame: (H, W, 3) uint8 RGB array.

        Returns:
            Base64-encoded JPEG bytes (no data URI prefix).
        """
        pil_image = Image.fromarray(frame)
        buf = BytesIO()
        pil_image.save(buf, format="JPEG", quality=_JPEG_QUALITY)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _call_vlm(self, frame: np.ndarray, prompt: str) -> str:
        """Send a vision request to GPT-4o via OpenRouter.

        Retries up to _MAX_RETRIES times on transient failures (network errors,
        5xx status codes).  4xx errors (bad auth, rate limit) are not retried.

        Args:
            frame: (H, W, 3) uint8 RGB array.
            prompt: System + user prompt to send alongside the image.

        Returns:
            Raw text content from the model's first choice.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
        """
        image_b64 = self._encode_frame(frame)

        payload: dict[str, Any] = {
            "model": _MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            t_start = time.monotonic()
            try:
                with httpx.Client(timeout=_TIMEOUT_S) as client:
                    response = client.post(
                        f"{_OPENROUTER_BASE_URL}/chat/completions",
                        json=payload,
                        headers=headers,
                    )

                elapsed = time.monotonic() - t_start

                # 4xx — auth/quota errors, do not retry
                if 400 <= response.status_code < 500:
                    raise RuntimeError(
                        f"OpenRouter API client error {response.status_code}: "
                        f"{response.text}"
                    )

                response.raise_for_status()
                data = response.json()

                usage: dict[str, int] = data.get("usage") or {}
                input_tokens: int = int(usage.get("prompt_tokens", 0))
                output_tokens: int = int(usage.get("completion_tokens", 0))
                call_cost = (
                    input_tokens * _COST_PER_INPUT_TOKEN
                    + output_tokens * _COST_PER_OUTPUT_TOKEN
                )

                with self._cost_lock:
                    self._cost_usd += call_cost

                logger.info(
                    "VLM call ok | attempt=%d elapsed=%.2fs "
                    "input_tokens=%d output_tokens=%d cost=$%.6f",
                    attempt,
                    elapsed,
                    input_tokens,
                    output_tokens,
                    call_cost,
                )

                choices: list[dict[str, Any]] = data.get("choices") or []
                if not choices:
                    raise RuntimeError("OpenRouter returned no choices.")

                return str(
                    choices[0].get("message", {}).get("content", "")
                )

            except (httpx.TransportError, httpx.TimeoutException) as exc:
                elapsed = time.monotonic() - t_start
                logger.warning(
                    "VLM call failed (attempt %d/%d, %.2fs): %s",
                    attempt,
                    _MAX_RETRIES,
                    elapsed,
                    exc,
                )
                last_exc = exc

            except RuntimeError:
                raise

            except Exception as exc:
                elapsed = time.monotonic() - t_start
                logger.warning(
                    "VLM call unexpected error (attempt %d/%d, %.2fs): %s",
                    attempt,
                    _MAX_RETRIES,
                    elapsed,
                    exc,
                )
                last_exc = exc

        raise RuntimeError(
            f"VLM API failed after {_MAX_RETRIES} attempts. "
            f"Last error: {last_exc}"
        )


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions)
# ---------------------------------------------------------------------------


def _parse_json_response(text: str) -> dict[str, Any]:
    """Parse a JSON object from the model response.

    Tries direct json.loads first.  Falls back to extracting a ```json
    fenced block if the model wrapped the JSON in Markdown.

    Args:
        text: Raw model output string.

    Returns:
        Parsed dict.  Returns empty dict on any parse failure.
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)  # type: ignore[return-value]
    except json.JSONDecodeError:
        pass

    # Try ```json ... ``` block
    match = re.search(r"```json\s*(.*?)\s*```", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))  # type: ignore[return-value]
        except json.JSONDecodeError:
            pass

    # Try any ``` ... ``` block
    match = re.search(r"```\s*(.*?)\s*```", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))  # type: ignore[return-value]
        except json.JSONDecodeError:
            pass

    logger.error("Failed to parse JSON from VLM response: %.200s", text)
    return {}


def _parse_detected_object(raw: dict[str, Any]) -> DetectedObject:
    """Construct a DetectedObject from a raw dict, clamping confidence.

    Args:
        raw: Dict with keys ``name``, ``description``, ``confidence``.

    Returns:
        DetectedObject with confidence clamped to [0.0, 1.0].
    """
    confidence = float(raw.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    return DetectedObject(
        name=str(raw.get("name", "")),
        description=str(raw.get("description", "")),
        confidence=confidence,
    )
