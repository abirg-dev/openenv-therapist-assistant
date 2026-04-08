from __future__ import annotations

from openenv_therapist_assistant.graders import TherapistAssistantGrader
from openenv_therapist_assistant.models import CheckInAction, HandoffOrEscalateAction
from openenv_therapist_assistant.tasks.task_bank import TaskRewardConfig, TherapistTask


def _task_with_config(config: TaskRewardConfig | None) -> TherapistTask:
    return TherapistTask(
        task_id="TEST-001",
        title="Task",
        difficulty="easy",
        client_message="I feel stressed this week.",
        objective="Build rapport.",
        success_criteria=["Warm opening"],
        target_actions=["check_in"],
        risk_level="low",
        reward_config=config,
    )


def test_task_action_bonus_changes_reward() -> None:
    grader = TherapistAssistantGrader()

    with_bonus = _task_with_config(TaskRewardConfig(action_bonus={"check_in": 0.2}))
    without_bonus = _task_with_config(None)

    action = CheckInAction(prompt="How are you feeling this week?")

    boosted = grader.score_action(
        action,
        with_bonus,
        phase="engagement",
        high_risk=False,
        safety_addressed=False,
        last_action_type=None,
        repeated_action_count=0,
    )
    baseline = grader.score_action(
        action,
        without_bonus,
        phase="engagement",
        high_risk=False,
        safety_addressed=False,
        last_action_type=None,
        repeated_action_count=0,
    )

    assert boosted.reward > baseline.reward


def test_done_on_action_triggers_completion() -> None:
    grader = TherapistAssistantGrader()
    task = _task_with_config(TaskRewardConfig(done_on_actions=["handoff_or_escalate"]))

    result = grader.score_action(
        HandoffOrEscalateAction(
            escalation_reason="high risk",
            target="licensed_therapist",
            handoff_message="Connecting you now.",
        ),
        task,
        phase="exploration",
        high_risk=False,
        safety_addressed=False,
        last_action_type=None,
        repeated_action_count=0,
    )

    assert result.done is True


def test_required_and_forbidden_keywords_adjust_quality() -> None:
    grader = TherapistAssistantGrader()
    better_task = _task_with_config(
        TaskRewardConfig(
            required_keywords=["stress", "week"],
            forbidden_phrases=["get over it"],
        )
    )
    worse_task = _task_with_config(
        TaskRewardConfig(
            required_keywords=["stress", "week"],
            forbidden_phrases=["how are you"],
        )
    )
    action = CheckInAction(prompt="How are you feeling with stress this week?")

    better = grader.score_action(
        action,
        better_task,
        phase="engagement",
        high_risk=False,
        safety_addressed=False,
        last_action_type=None,
        repeated_action_count=0,
    )
    worse = grader.score_action(
        action,
        worse_task,
        phase="engagement",
        high_risk=False,
        safety_addressed=False,
        last_action_type=None,
        repeated_action_count=0,
    )

    assert better.breakdown["quality"] > worse.breakdown["quality"]
