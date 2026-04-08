# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Openenv Therapist Assistant Environment Client."""

from __future__ import annotations

from typing import Dict, Literal

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import OpenenvTherapistAssistantAction, OpenenvTherapistAssistantObservation


class OpenenvTherapistAssistantEnv(
    EnvClient[OpenenvTherapistAssistantAction, OpenenvTherapistAssistantObservation, State]
):
    """
    Client for the Openenv Therapist Assistant Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> async with OpenenvTherapistAssistantEnv(base_url="http://localhost:8000") as client:
        ...     result = await client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = await client.step(
        ...         {"action_type": "check_in", "prompt": "How have you felt since our last session?"}
        ...     )
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = await OpenenvTherapistAssistantEnv.from_docker_image("openenv_therapist_assistant-env:latest")
        >>> try:
        ...     result = await client.reset()
        ...     result = await client.step(
        ...         {"action_type": "ask_open_question", "question_text": "What feels hardest right now?", "focus_area": "emotion"}
        ...     )
        ... finally:
        ...     await client.close()
    """

    async def reset(
        self,
        seed: int | None = None,
        difficulty: Literal["easy", "moderate", "hard"] | None = None,
    ) -> StepResult[OpenenvTherapistAssistantObservation]:
        """Reset the environment with optional task selection controls."""
        reset_kwargs: Dict[str, object] = {}
        if seed is not None:
            reset_kwargs["seed"] = seed
        if difficulty is not None:
            reset_kwargs["difficulty"] = difficulty
        return await super().reset(**reset_kwargs)

    def _step_payload(self, action: OpenenvTherapistAssistantAction) -> Dict:
        """
        Convert OpenenvTherapistAssistantAction to JSON payload for step message.

        Args:
            action: OpenenvTherapistAssistantAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[OpenenvTherapistAssistantObservation]:
        """
        Parse server response into StepResult[OpenenvTherapistAssistantObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with OpenenvTherapistAssistantObservation
        """
        obs_data = payload.get("observation", {})
        observation = OpenenvTherapistAssistantObservation(
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
