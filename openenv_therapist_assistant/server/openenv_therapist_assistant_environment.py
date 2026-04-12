# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Openenv Therapist Assistant Environment Implementation.

A lightweight therapist-assistant action sandbox.

The environment validates discriminated `action_type` payloads and converts each
action into a compact echoed summary string for fast end-to-end API testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

try:
    from ..graders import TherapistAssistantGrader
    from ..models import (
        OPENENV_THERAPIST_ASSISTANT_ACTION_CLASSES,
        OpenenvTherapistAssistantActionAdapter,
        OpenenvTherapistAssistantObservation,
    )
    from ..tasks import TaskBank
except ImportError:
    from graders import TherapistAssistantGrader
    from models import (
        OPENENV_THERAPIST_ASSISTANT_ACTION_CLASSES,
        OpenenvTherapistAssistantActionAdapter,
        OpenenvTherapistAssistantObservation,
    )
    from tasks import TaskBank

if TYPE_CHECKING:
    from ..models import OpenenvTherapistAssistantAction, OpenenvTherapistAssistantConcreteAction
    from ..tasks import TherapistTask


class OpenenvTherapistAssistantEnvironment(Environment):
    """
    A lightweight action-summary environment for therapist assistant training.

    This environment is designed for testing client/server infrastructure and
    action-shape validation before introducing richer conversational dynamics.
    It maintains minimal state and converts each valid action into a readable
    summary string.

    Example:
        >>> env = OpenenvTherapistAssistantEnvironment()
        >>> obs = await env.reset()
        >>> print(obs.echoed_message)  # "Openenv Therapist Assistant environment ready!"
        >>>
        >>> obs = await env.step({"action_type": "check_in", "prompt": "How are you feeling today?"})
        >>> print(obs.echoed_message)  # "[check_in] How are you feeling today?"
        >>> print(obs.message_length)  # len("[check_in] How are you feeling today?")
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, modality_profile: str = "balanced"):
        """Initialize environment state counters.
        
        Args:
            modality_profile: Therapeutic modality emphasis ('balanced', 'mi_leaning',
                'cbt_leaning', 'psychodynamic_leaning'). Defaults to 'balanced'.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._modality_profile = modality_profile
        self._grader = TherapistAssistantGrader(modality_profile=modality_profile)
        self._reset_count = 0
        self._phase = "engagement"
        self._high_risk = False
        self._safety_addressed = False
        self._last_action_type: str | None = None
        self._repeated_action_count = 0
        self._task_bank = TaskBank()
        self._current_task: TherapistTask | None = None

    def reset(
        self,
        seed: int | None = None,
        difficulty: Literal["easy", "moderate", "hard"] | None = None,
        modality_profile: str | None = None,
    ) -> OpenenvTherapistAssistantObservation:  # type: ignore[override]
        """
        Reset the environment and sample a therapist task.

        Args:
            seed: Optional deterministic seed for task selection.
            difficulty: Optional difficulty filter (easy, moderate, hard).
            modality_profile: Optional therapeutic modality emphasis. If provided, updates
                the grader's profile for this and subsequent episodes.

        Returns:
            OpenenvTherapistAssistantObservation with a ready message
        """
        if modality_profile is not None:
            self._modality_profile = modality_profile
            self._grader = TherapistAssistantGrader(modality_profile=modality_profile)
        return self._reset_impl(seed=seed, difficulty=difficulty)

    def step(self, action: OpenenvTherapistAssistantAction | dict) -> OpenenvTherapistAssistantObservation:  # type: ignore[override]
        """
        Execute one environment step from a typed therapist action.

        Args:
            action: OpenenvTherapistAssistantAction (or a dict payload) with action_type

        Returns:
            OpenenvTherapistAssistantObservation with the action summary and its length
        """
        return self._step_impl(action)

    async def reset_async(
        self,
        seed: int | None = None,
        difficulty: Literal["easy", "moderate", "hard"] | None = None,
        modality_profile: str | None = None,
    ) -> OpenenvTherapistAssistantObservation:
        return self.reset(seed=seed, difficulty=difficulty, modality_profile=modality_profile)

    async def step_async(
        self,
        action: OpenenvTherapistAssistantAction | dict,
    ) -> OpenenvTherapistAssistantObservation:
        return self.step(action)

    def _reset_impl(
        self,
        seed: int | None = None,
        difficulty: Literal["easy", "moderate", "hard"] | None = None,
    ) -> OpenenvTherapistAssistantObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._phase = "engagement"
        self._high_risk = False
        self._safety_addressed = False
        self._last_action_type = None
        self._repeated_action_count = 0

        selected_seed = self._reset_count if seed is None else seed
        self._current_task = self._task_bank.get_task(seed=selected_seed, difficulty=difficulty)
        self._reset_count += 1
        reset_message = f"Openenv Therapist Assistant environment ready! Task: {self._current_task.title}"

        return OpenenvTherapistAssistantObservation(
            echoed_message=reset_message,
            message_length=len(reset_message),
            done=False,
            reward=0.0,
            metadata={
                "episode_id": self._state.episode_id,
                "task": self._task_metadata(),
                "step": self._state.step_count,
                "phase": self._phase,
                "last_action_type": self._last_action_type,
                "modality_profile": self._modality_profile,
                "task_selection": {
                    "seed": selected_seed,
                    "difficulty_requested": difficulty,
                },
            },
        )

    def _step_impl(self, action: OpenenvTherapistAssistantAction | dict) -> OpenenvTherapistAssistantObservation:
        self._state.step_count += 1

        parsed_action = self._parse_action(action)
        message = self._action_to_message(parsed_action)
        length = len(message)

        reward, breakdown, done_override = self._score_action(parsed_action)
        done = bool(done_override)

        return OpenenvTherapistAssistantObservation(
            echoed_message=message,
            message_length=length,
            done=done,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "original_message": message,
                "step": self._state.step_count,
                "phase": self._phase,
                "last_action_type": self._last_action_type,
                "modality_profile": self._modality_profile,
                "high_risk": self._high_risk,
                "safety_addressed": self._safety_addressed,
                "reward_breakdown": breakdown,
                "task": self._task_metadata(),
            },
        )

    def _task_metadata(self) -> dict[str, Any]:
        if self._current_task is None:
            return {}
        return self._current_task.model_dump()

    @staticmethod
    def _parse_action(
        action: OpenenvTherapistAssistantAction | dict[str, Any],
    ) -> OpenenvTherapistAssistantConcreteAction:
        if isinstance(action, dict):
            return OpenenvTherapistAssistantActionAdapter.validate_python(action)
        # OpenEnv HTTP server may pass a RootModel wrapper around the union.
        root_action = getattr(action, "root", None)
        if isinstance(root_action, dict):
            return OpenenvTherapistAssistantActionAdapter.validate_python(root_action)
        if isinstance(root_action, OPENENV_THERAPIST_ASSISTANT_ACTION_CLASSES):
            return root_action
        if isinstance(action, OPENENV_THERAPIST_ASSISTANT_ACTION_CLASSES):
            return action
        if hasattr(action, "model_dump"):
            return OpenenvTherapistAssistantActionAdapter.validate_python(action.model_dump(exclude_none=True))
        raise ValueError("Unsupported action payload for OpenenvTherapistAssistantEnvironment.step")

    @staticmethod
    def _action_to_message(action) -> str:
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

    def _score_action(
        self,
        action: OpenenvTherapistAssistantConcreteAction,
    ) -> tuple[float, dict[str, float], bool]:
        result = self._grader.score_action(
            action,
            self._current_task,
            phase=self._phase,
            high_risk=self._high_risk,
            safety_addressed=self._safety_addressed,
            last_action_type=self._last_action_type,
            repeated_action_count=self._repeated_action_count,
        )

        self._phase = result.phase
        self._high_risk = result.high_risk
        self._safety_addressed = result.safety_addressed
        self._repeated_action_count = result.repeated_action_count
        self._last_action_type = result.last_action_type
        return result.reward, result.breakdown, result.done

    def get_metadata(self) -> EnvironmentMetadata:
        """Expose README content for the OpenEnv web interface metadata panel."""
        readme_path = Path(__file__).resolve().parent.parent / "README.md"
        readme_content: str | None = None
        if readme_path.exists():
            readme_content = readme_path.read_text(encoding="utf-8")

        return EnvironmentMetadata(
            name="openenv_therapist_assistant",
            description="Therapist-assistant training environment with structured actions",
            version="1.0.0",
            readme_content=readme_content,
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
