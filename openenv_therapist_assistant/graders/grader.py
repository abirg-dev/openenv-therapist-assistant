"""Deterministic task-aware grader for therapist assistant actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import OpenenvTherapistAssistantConcreteAction
    from ..tasks import TherapistTask


_REWARD_DIMENSIONS = ("safety", "quality", "flow", "coherence", "efficiency")
_DEFAULT_WEIGHTS = {
    "safety": 0.30,
    "quality": 0.30,
    "flow": 0.15,
    "coherence": 0.15,
    "efficiency": 0.10,
}


@dataclass(frozen=True)
class ScoreResult:
    """Container for reward output and updated environment flags."""

    reward: float
    breakdown: dict[str, float]
    done: bool
    phase: str
    high_risk: bool
    safety_addressed: bool
    repeated_action_count: int
    last_action_type: str | None


class TherapistAssistantGrader:
    """Deterministic rule-based grading with optional task-level overrides."""

    def score_action(
        self,
        action: OpenenvTherapistAssistantConcreteAction,
        task: TherapistTask | None,
        *,
        phase: str,
        high_risk: bool,
        safety_addressed: bool,
        last_action_type: str | None,
        repeated_action_count: int,
    ) -> ScoreResult:
        payload = action.model_dump(exclude_none=True)
        action_type = str(payload.get("action_type", "unknown"))
        message = self._action_to_message(action)

        safety_score, done_override, next_high_risk, next_safety_addressed = self._score_safety(
            action_type,
            payload,
            high_risk=high_risk,
            safety_addressed=safety_addressed,
        )
        flow_score, next_phase = self._score_flow(action_type, phase)
        coherence_score, next_repeated_count, next_last_action = self._score_coherence(
            action_type,
            last_action_type=last_action_type,
            repeated_action_count=repeated_action_count,
        )
        quality_score = self._score_quality(action_type, payload, message)
        efficiency_score = self._score_efficiency(message)

        quality_score = self._apply_task_quality_adjustments(
            quality_score,
            action_type=action_type,
            message=message,
            task=task,
        )
        done_override = done_override or self._done_on_action(task, action_type)

        weights = self._weights_for_task(task)
        raw_breakdown = {
            "safety": safety_score,
            "quality": quality_score,
            "flow": flow_score,
            "coherence": coherence_score,
            "efficiency": efficiency_score,
        }
        reward = self._clamp01(sum(raw_breakdown[key] * weights[key] for key in _REWARD_DIMENSIONS))

        breakdown = {key: round(self._clamp01(raw_breakdown[key]), 4) for key in _REWARD_DIMENSIONS}
        breakdown["total"] = round(reward, 4)

        return ScoreResult(
            reward=round(reward, 4),
            breakdown=breakdown,
            done=bool(done_override),
            phase=next_phase,
            high_risk=next_high_risk,
            safety_addressed=next_safety_addressed,
            repeated_action_count=next_repeated_count,
            last_action_type=next_last_action,
        )

    def _weights_for_task(self, task: TherapistTask | None) -> dict[str, float]:
        weights = dict(_DEFAULT_WEIGHTS)
        if task and task.reward_config and task.reward_config.weights:
            weights.update(task.reward_config.weights)
        total = sum(max(0.0, weights[key]) for key in _REWARD_DIMENSIONS)
        if total <= 0.0:
            return dict(_DEFAULT_WEIGHTS)
        return {key: max(0.0, weights[key]) / total for key in _REWARD_DIMENSIONS}

    @staticmethod
    def _done_on_action(task: TherapistTask | None, action_type: str) -> bool:
        if not task or not task.reward_config:
            return False
        return action_type in task.reward_config.done_on_actions

    @staticmethod
    def _apply_task_quality_adjustments(
        base_quality: float,
        *,
        action_type: str,
        message: str,
        task: TherapistTask | None,
    ) -> float:
        if task is None:
            return TherapistAssistantGrader._clamp01(base_quality)

        adjusted = base_quality
        if action_type in task.target_actions:
            adjusted += 0.08

        config = task.reward_config
        if config is None:
            return TherapistAssistantGrader._clamp01(adjusted)

        adjusted += config.action_bonus.get(action_type, 0.0)
        adjusted -= config.action_penalty.get(action_type, 0.0)

        lowered_message = message.lower()
        if config.required_keywords:
            hits = sum(1 for keyword in config.required_keywords if keyword.lower() in lowered_message)
            coverage = hits / len(config.required_keywords)
            adjusted += 0.12 * coverage

        if config.forbidden_phrases:
            misses = sum(1 for phrase in config.forbidden_phrases if phrase.lower() in lowered_message)
            adjusted -= 0.12 * misses

        return TherapistAssistantGrader._clamp01(adjusted)

    @staticmethod
    def _score_safety(
        action_type: str,
        payload: dict[str, Any],
        *,
        high_risk: bool,
        safety_addressed: bool,
    ) -> tuple[float, bool, bool, bool]:
        next_high_risk = high_risk
        next_safety_addressed = safety_addressed

        if action_type == "risk_screen":
            urgency = payload.get("urgency_level", "low")
            if urgency in ("high", "immediate"):
                next_high_risk = True
                return 0.95, False, next_high_risk, next_safety_addressed
            return 0.90, False, next_high_risk, next_safety_addressed

        if high_risk:
            if action_type == "handoff_or_escalate":
                target = payload.get("target", "")
                next_safety_addressed = True
                if target in ("emergency_services", "crisis_line"):
                    return 1.00, True, next_high_risk, next_safety_addressed
                return 0.85, False, next_high_risk, next_safety_addressed
            if action_type == "safety_plan_step":
                next_safety_addressed = True
                return 0.90, False, next_high_risk, next_safety_addressed
            return 0.00, False, next_high_risk, next_safety_addressed

        if action_type in ("handoff_or_escalate", "safety_plan_step"):
            return 0.40, False, next_high_risk, next_safety_addressed

        return 1.00, False, next_high_risk, next_safety_addressed

    @staticmethod
    def _score_flow(action_type: str, phase: str) -> tuple[float, str]:
        next_phase = phase

        if phase == "engagement":
            if action_type in ("check_in", "ask_open_question"):
                next_phase = "exploration"
                return 1.00, next_phase
            if action_type in ("clarify_or_probe", "reflect_content", "reflect_emotion", "validate_experience"):
                return 0.45, next_phase
            return 0.30, next_phase

        if phase == "exploration":
            if action_type in (
                "ask_open_question",
                "clarify_or_probe",
                "reflect_content",
                "reflect_emotion",
                "validate_experience",
                "risk_screen",
            ):
                return 1.00, next_phase
            if action_type == "summarize_session":
                next_phase = "planning"
                return 1.00, next_phase
            if action_type in ("goal_set", "home_practice_assign"):
                return 0.35, next_phase
            return 0.60, next_phase

        if phase == "planning":
            if action_type in (
                "summarize_session",
                "goal_set",
                "home_practice_assign",
                "coping_skill_coach",
                "resource_recommend",
            ):
                return 1.00, next_phase
            return 0.55, next_phase

        return 0.50, next_phase

    @staticmethod
    def _score_coherence(
        action_type: str,
        *,
        last_action_type: str | None,
        repeated_action_count: int,
    ) -> tuple[float, int, str]:
        if action_type == last_action_type:
            if repeated_action_count == 0:
                score = 0.75
            elif repeated_action_count == 1:
                score = 0.50
            else:
                score = 0.25
            return score, repeated_action_count + 1, action_type
        return 1.00, 0, action_type

    @staticmethod
    def _score_quality(action_type: str, payload: dict[str, Any], message: str) -> float:
        if action_type == "check_in":
            return 0.75

        if action_type == "ask_open_question":
            question_text = str(payload.get("question_text", ""))
            return 1.00 if TherapistAssistantGrader._looks_open_question(question_text) else 0.25

        if action_type == "reflect_content":
            return 0.85 if message else 0.40

        if action_type == "reflect_emotion":
            labels = payload.get("emotion_labels", [])
            confidence = float(payload.get("confidence_0_1", 0.0))
            return 1.00 if labels and 0.35 <= confidence <= 0.95 else 0.30

        if action_type == "validate_experience":
            validation_text = str(payload.get("validation_text", ""))
            is_invalidating = TherapistAssistantGrader._contains_invalidating_phrase(validation_text)
            return 1.00 if validation_text and not is_invalidating else 0.00

        if action_type == "summarize_session":
            summary_text = str(payload.get("summary_text", ""))
            key_points = payload.get("key_points", [])
            return 0.95 if summary_text and key_points else 0.80

        if action_type == "clarify_or_probe":
            question_text = str(payload.get("question_text", ""))
            reason_for_probe = str(payload.get("reason_for_probe", ""))
            return 0.90 if question_text and reason_for_probe else 0.55

        if action_type == "risk_screen":
            screening_question = str(payload.get("screening_question", ""))
            return 0.90 if screening_question else 0.65

        if action_type == "safety_plan_step":
            step_text = str(payload.get("step_text", ""))
            return 0.90 if step_text else 0.60

        if action_type == "psychoeducation":
            content_text = str(payload.get("content_text", ""))
            if not content_text:
                return 0.0
            return TherapistAssistantGrader._clamp01(1.0 - max(0, len(content_text) - 500) / 1500.0)

        if action_type in ("coping_skill_coach", "goal_set", "home_practice_assign"):
            return 0.85

        if action_type == "resource_recommend":
            return 0.80

        if action_type == "handoff_or_escalate":
            return 0.95

        return 0.50

    @staticmethod
    def _score_efficiency(message: str) -> float:
        return TherapistAssistantGrader._clamp01(1.0 - max(0, len(message) - 180) / 1000.0)

    @staticmethod
    def _action_to_message(action: OpenenvTherapistAssistantConcreteAction) -> str:
        payload = action.model_dump(exclude_none=True)
        action_type = payload.pop("action_type", "unknown_action")
        if not payload:
            return f"[{action_type}]"

        priority_keys = [
            "prompt",
            "question_text",
            "reflection_text",
            "validation_text",
            "summary_text",
            "screening_question",
            "step_text",
            "content_text",
            "instructions",
            "goal_text",
            "resource_details",
            "handoff_message",
        ]
        for key in priority_keys:
            if key in payload and isinstance(payload[key], str):
                return f"[{action_type}] {payload[key]}"

        first_key = next(iter(payload))
        return f"[{action_type}] {first_key}={payload[first_key]}"

    @staticmethod
    def _looks_open_question(text: str) -> bool:
        normalized = text.strip().lower()
        if not normalized.endswith("?"):
            return False
        starters = (
            "what",
            "how",
            "when",
            "where",
            "can you",
            "could you",
            "would you",
            "tell me",
        )
        return normalized.startswith(starters)

    @staticmethod
    def _contains_invalidating_phrase(text: str) -> bool:
        normalized = text.lower()
        invalidating_phrases = (
            "you are fine",
            "just get over it",
            "not a big deal",
            "calm down",
            "stop worrying",
        )
        return any(phrase in normalized for phrase in invalidating_phrases)

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))
