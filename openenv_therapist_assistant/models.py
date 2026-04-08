# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Openenv Therapist Assistant Environment."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias, Union

from openenv.core.env_server.types import Action, Observation
from pydantic import Discriminator, Field, RootModel, TypeAdapter, model_validator


class CheckInAction(Action):
    """Start a turn by checking the client's current state."""

    action_type: Literal["check_in"] = "check_in"
    prompt: str = Field(min_length=1, max_length=500)
    mood_scale_1_10: int | None = Field(default=None, ge=1, le=10)
    energy_scale_1_10: int | None = Field(default=None, ge=1, le=10)
    sleep_quality_1_10: int | None = Field(default=None, ge=1, le=10)


class AskOpenQuestionAction(Action):
    """Ask a non-leading exploratory question."""

    action_type: Literal["ask_open_question"] = "ask_open_question"
    question_text: str = Field(min_length=1, max_length=1000)
    focus_area: Literal["emotion", "event", "relationship", "coping"]


class ReflectContentAction(Action):
    """Paraphrase the client's content."""

    action_type: Literal["reflect_content"] = "reflect_content"
    reflection_text: str = Field(min_length=1, max_length=1000)
    source_span: str | None = Field(default=None, max_length=500)


class ReflectEmotionAction(Action):
    """Name and reflect likely emotions."""

    action_type: Literal["reflect_emotion"] = "reflect_emotion"
    emotion_labels: list[str] = Field(min_length=1, max_length=6)
    reflection_text: str = Field(min_length=1, max_length=1000)
    confidence_0_1: float = Field(ge=0.0, le=1.0)


class ValidateExperienceAction(Action):
    """Validate the client's emotional experience."""

    action_type: Literal["validate_experience"] = "validate_experience"
    validation_text: str = Field(min_length=1, max_length=1000)
    context_reference: str | None = Field(default=None, max_length=500)


class SummarizeSessionAction(Action):
    """Summarize key points and open items."""

    action_type: Literal["summarize_session"] = "summarize_session"
    summary_text: str = Field(min_length=1, max_length=2000)
    key_points: list[str] = Field(default_factory=list, max_length=10)
    open_items: list[str] = Field(default_factory=list, max_length=10)


class ClarifyOrProbeAction(Action):
    """Ask a targeted clarification or probe question."""

    action_type: Literal["clarify_or_probe"] = "clarify_or_probe"
    question_text: str = Field(min_length=1, max_length=1000)
    reason_for_probe: str = Field(min_length=1, max_length=500)


class RiskScreenAction(Action):
    """Run a structured risk/safety check."""

    action_type: Literal["risk_screen"] = "risk_screen"
    risk_domain: Literal["self_harm", "harm_to_others", "abuse", "substance"]
    screening_question: str = Field(min_length=1, max_length=1000)
    urgency_level: Literal["low", "medium", "high", "immediate"]


class SafetyPlanStepAction(Action):
    """Add a specific safety plan item."""

    action_type: Literal["safety_plan_step"] = "safety_plan_step"
    step_type: Literal[
        "coping_strategy",
        "support_contact",
        "means_restriction",
        "crisis_resource",
    ]
    step_text: str = Field(min_length=1, max_length=1000)


class PsychoeducationAction(Action):
    """Provide brief psychoeducation."""

    action_type: Literal["psychoeducation"] = "psychoeducation"
    topic: str = Field(min_length=1, max_length=120)
    content_text: str = Field(min_length=1, max_length=2000)
    reading_level: Literal["basic", "intermediate", "advanced"] = "basic"


class CopingSkillCoachAction(Action):
    """Guide the client through a coping skill."""

    action_type: Literal["coping_skill_coach"] = "coping_skill_coach"
    skill_name: str = Field(min_length=1, max_length=120)
    instructions: str = Field(min_length=1, max_length=2000)
    duration_minutes: int = Field(ge=1, le=120)


class GoalSetAction(Action):
    """Create a collaborative goal."""

    action_type: Literal["goal_set"] = "goal_set"
    goal_text: str = Field(min_length=1, max_length=1000)
    time_horizon: Literal["this_week", "this_month", "next_3_months", "long_term"]
    success_criteria: str = Field(min_length=1, max_length=1000)


class HomePracticeAssignAction(Action):
    """Assign between-session home practice."""

    action_type: Literal["home_practice_assign"] = "home_practice_assign"
    practice_name: str = Field(min_length=1, max_length=120)
    instructions: str = Field(min_length=1, max_length=2000)
    frequency: str = Field(min_length=1, max_length=120)
    tracking_method: str = Field(min_length=1, max_length=300)


class ResourceRecommendAction(Action):
    """Recommend an external support resource."""

    action_type: Literal["resource_recommend"] = "resource_recommend"
    resource_type: Literal["hotline", "worksheet", "community_service", "app"]
    resource_details: str = Field(min_length=1, max_length=1000)
    reason: str = Field(min_length=1, max_length=500)


class HandoffOrEscalateAction(Action):
    """Escalate or handoff to human support when needed."""

    action_type: Literal["handoff_or_escalate"] = "handoff_or_escalate"
    escalation_reason: str = Field(min_length=1, max_length=500)
    target: Literal["licensed_therapist", "emergency_services", "crisis_line"]
    handoff_message: str = Field(min_length=1, max_length=1000)


OpenenvTherapistAssistantConcreteAction: TypeAlias = Union[
    CheckInAction,
    AskOpenQuestionAction,
    ReflectContentAction,
    ReflectEmotionAction,
    ValidateExperienceAction,
    SummarizeSessionAction,
    ClarifyOrProbeAction,
    RiskScreenAction,
    SafetyPlanStepAction,
    PsychoeducationAction,
    CopingSkillCoachAction,
    GoalSetAction,
    HomePracticeAssignAction,
    ResourceRecommendAction,
    HandoffOrEscalateAction,
]

OpenenvTherapistAssistantAction: TypeAlias = Annotated[
    OpenenvTherapistAssistantConcreteAction,
    Discriminator("action_type"),
]

OpenenvTherapistAssistantActionAdapter: TypeAdapter[OpenenvTherapistAssistantAction] = TypeAdapter(
    OpenenvTherapistAssistantAction
)  # type: ignore[arg-type]


class OpenenvTherapistAssistantActionModel(RootModel[OpenenvTherapistAssistantAction]):
    """HTTP-server wrapper model for discriminated therapist actions."""

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, value: object) -> object:
        """Normalize Hub/UI payload variants before discriminated-union validation."""
        if not isinstance(value, dict):
            return value

        # RootModel payloads may arrive as {"root": {...}}.
        if "root" in value and isinstance(value.get("root"), dict):
            value = value["root"]

        # Hub web UI can submit an empty action object; default to a safe no-op-like action.
        if value == {}:
            return {"action_type": "check_in", "prompt": "hello"}

        return value

OPENENV_THERAPIST_ASSISTANT_ACTION_CLASSES = (
    CheckInAction,
    AskOpenQuestionAction,
    ReflectContentAction,
    ReflectEmotionAction,
    ValidateExperienceAction,
    SummarizeSessionAction,
    ClarifyOrProbeAction,
    RiskScreenAction,
    SafetyPlanStepAction,
    PsychoeducationAction,
    CopingSkillCoachAction,
    GoalSetAction,
    HomePracticeAssignAction,
    ResourceRecommendAction,
    HandoffOrEscalateAction,
)


class OpenenvTherapistAssistantObservation(Observation):
    """Observation from the Openenv Therapist Assistant environment - the echoed message."""

    echoed_message: str = Field(default="", description="The echoed message")
    message_length: int = Field(default=0, description="Length of the echoed message")
