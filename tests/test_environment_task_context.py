from __future__ import annotations

from openenv_therapist_assistant.models import CheckInAction, OpenenvTherapistAssistantActionModel
from openenv_therapist_assistant.server.openenv_therapist_assistant_environment import (
    OpenenvTherapistAssistantEnvironment,
)


def test_reset_includes_task_metadata() -> None:
    env = OpenenvTherapistAssistantEnvironment()

    result = env.reset()

    assert result.metadata is not None
    assert "task" in result.metadata
    task = result.metadata["task"]
    assert task["task_id"]
    assert task["difficulty"] in {"easy", "moderate", "hard"}
    assert task["target_actions"]


def test_step_metadata_preserves_active_task() -> None:
    env = OpenenvTherapistAssistantEnvironment()
    reset_result = env.reset()

    step_result = env.step(CheckInAction(prompt="How have you been feeling this week?"))

    assert step_result.metadata is not None
    assert step_result.metadata["task"]["task_id"] == reset_result.metadata["task"]["task_id"]


def test_reset_progresses_deterministically_through_tasks() -> None:
    env_a = OpenenvTherapistAssistantEnvironment()
    env_b = OpenenvTherapistAssistantEnvironment()

    first_a = env_a.reset().metadata["task"]["task_id"]
    second_a = env_a.reset().metadata["task"]["task_id"]
    first_b = env_b.reset().metadata["task"]["task_id"]
    second_b = env_b.reset().metadata["task"]["task_id"]

    assert first_a == first_b
    assert second_a == second_b


def test_reset_accepts_difficulty_filter() -> None:
    env = OpenenvTherapistAssistantEnvironment()

    result = env.reset(difficulty="hard")

    assert result.metadata["task"]["difficulty"] == "hard"
    assert result.metadata["task_selection"]["difficulty_requested"] == "hard"


def test_reset_seed_is_deterministic_for_selected_pool() -> None:
    env = OpenenvTherapistAssistantEnvironment()

    first = env.reset(seed=2, difficulty="moderate").metadata["task"]["task_id"]
    second = env.reset(seed=2, difficulty="moderate").metadata["task"]["task_id"]

    assert first == second


def test_reset_has_task_metadata() -> None:
    env = OpenenvTherapistAssistantEnvironment()

    result = env.reset()

    assert result.metadata is not None
    assert "task" in result.metadata
    assert result.metadata["task"]["task_id"]


def test_step_preserves_active_task() -> None:
    env = OpenenvTherapistAssistantEnvironment()
    reset_result = env.reset()

    step_result = env.step(CheckInAction(prompt="How have you been feeling this week?"))

    assert step_result.metadata is not None
    assert step_result.metadata["task"]["task_id"] == reset_result.metadata["task"]["task_id"]


def test_same_action_scores_differently_across_tasks() -> None:
    env = OpenenvTherapistAssistantEnvironment()

    easy_rapport = env.reset(seed=0, difficulty="easy")
    reward_rapport = env.step(CheckInAction(prompt="How have you been feeling this week?")).reward

    easy_list_problems = env.reset(seed=1, difficulty="easy")
    reward_list_problems = env.step(CheckInAction(prompt="How have you been feeling this week?")).reward

    assert easy_rapport.metadata["task"]["task_id"] != easy_list_problems.metadata["task"]["task_id"]
    assert reward_rapport != reward_list_problems


def test_step_accepts_root_model_wrapped_action() -> None:
    env = OpenenvTherapistAssistantEnvironment()
    env.reset()

    wrapped_action = OpenenvTherapistAssistantActionModel.model_validate(
        {"action_type": "check_in", "prompt": "How are you feeling today?"}
    )

    result = env.step(wrapped_action)

    assert result.echoed_message.startswith("[check_in]")
    assert result.message_length > 0


def test_action_model_normalizes_empty_payload() -> None:
    wrapped_action = OpenenvTherapistAssistantActionModel.model_validate({})

    assert wrapped_action.root.action_type == "check_in"
    assert wrapped_action.root.prompt == "hello"
