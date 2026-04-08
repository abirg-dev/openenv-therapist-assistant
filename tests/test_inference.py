from __future__ import annotations

from typing import Any

import openenv_therapist_assistant.inference as inference
from openenv_therapist_assistant.inference import compute_episode_score


def test_compute_episode_score_empty_rewards() -> None:
    assert compute_episode_score([]) == 0.0


def test_compute_episode_score_average_in_range() -> None:
    assert compute_episode_score([0.2, 0.6, 1.0]) == 0.6


def test_compute_episode_score_clamps_negative_average() -> None:
    assert compute_episode_score([-0.4, -0.2]) == 0.0


def test_compute_episode_score_clamps_above_one() -> None:
    assert compute_episode_score([1.2, 1.4]) == 1.0


def test_extract_json_payload_from_fenced_block() -> None:
    text = "```json\n{\"action_type\":\"check_in\",\"prompt\":\"hi\"}\n```"
    payload = inference._extract_json_payload(text)

    assert payload == {"action_type": "check_in", "prompt": "hi"}


def test_get_model_action_payload_reports_model_failure(monkeypatch: Any) -> None:
    def boom(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("api down")

    monkeypatch.setattr(inference, "get_model_message", boom)

    payload, error = inference.get_model_action_payload(
        client=None,  # type: ignore[arg-type]
        step=1,
        last_echoed="hello",
        last_reward=0.0,
        history=[],
    )

    assert payload == {"action_type": "check_in", "prompt": "hello"}
    assert error is not None
    assert "Model request failed: api down" in error
