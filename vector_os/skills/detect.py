"""DetectSkill — detect objects in workspace using perception backend.

Ported from perception_skills._detect_all_cb() + the detect-all logic.
The actual VLM/detector call is delegated to context.perception.detect(query)
so the skill is decoupled from the specific perception implementation.

No ROS2 imports.
"""
from __future__ import annotations

import logging
import time

from vector_os.core.skill import Skill, SkillContext
from vector_os.core.types import SkillResult
from vector_os.core.world_model import ObjectState

logger = logging.getLogger(__name__)


class DetectSkill:
    """Detect objects in the workspace using VLM perception.

    Calls context.perception.detect(query) and updates the world model
    with the returned Detection objects.  If context.perception is None
    (no camera / headless mode), the skill fails gracefully.

    Parameters:
        query (str, required): What to detect, e.g. "red cup" or "all objects".
    """

    name: str = "detect"
    description: str = "Detect objects in the workspace using VLM"
    parameters: dict = {
        "query": {
            "type": "string",
            "required": True,
            "description": "What to detect (e.g. 'red cup', 'all objects')",
        }
    }
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {}  # World model updated inside execute via add_object

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Run detection and populate the world model.

        The world model is updated by calling add_object() for each Detection
        returned by perception.  Existing objects with the same ID are
        overwritten so that stale positions are refreshed.

        Object IDs are synthesised as "<label>_<index>" (e.g. "red_cup_0").

        Args:
            params: must contain "query" key.
            context: SkillContext providing perception and world_model access.

        Returns:
            SkillResult with result_data["objects"] listing detected labels
            and confidences, or failure with error_message if no perception
            backend is available.
        """
        if context.perception is None:
            logger.warning("[DETECT] No perception backend available")
            return SkillResult(
                success=False,
                error_message="No perception backend available",
            )

        query: str = params.get("query", "all objects")
        logger.info("[DETECT] Running detection for query: %r", query)

        try:
            detections = context.perception.detect(query)
        except Exception as exc:
            logger.error("[DETECT] Perception error: %s", exc)
            return SkillResult(
                success=False,
                error_message=f"Perception error: {exc}",
            )

        if not detections:
            logger.info("[DETECT] No objects detected")
            return SkillResult(
                success=True,
                result_data={"objects": [], "count": 0},
            )

        now = time.time()
        object_summaries: list[dict] = []

        for idx, det in enumerate(detections):
            # Synthesise a stable object_id from label + index
            safe_label = det.label.replace(" ", "_").lower()
            obj_id = f"{safe_label}_{idx}"

            # Build ObjectState — bbox centre is used for 3D position when
            # a 3D position is not available (bbox is 2D pixel coords here)
            obj = ObjectState(
                object_id=obj_id,
                label=det.label,
                x=0.0,   # unknown until camera→base transform is applied
                y=0.0,
                z=0.0,
                confidence=det.confidence,
                state="on_table",
                last_seen=now,
            )
            context.world_model.add_object(obj)

            object_summaries.append({
                "object_id": obj_id,
                "label": det.label,
                "confidence": round(det.confidence, 4),
            })
            logger.debug(
                "[DETECT] Object %s: label=%r confidence=%.3f",
                obj_id, det.label, det.confidence,
            )

        logger.info("[DETECT] Detected %d object(s)", len(detections))
        return SkillResult(
            success=True,
            result_data={
                "objects": object_summaries,
                "count": len(object_summaries),
            },
        )
