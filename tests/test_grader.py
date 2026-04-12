from __future__ import annotations

from openenv_therapist_assistant.graders import TherapistAssistantGrader
from openenv_therapist_assistant.models import (
    CheckInAction,
    ClarifyOrProbeAction,
    HandoffOrEscalateAction,
    SummarizeSessionAction,
    ValidateExperienceAction,
)
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


def test_coercive_validation_is_penalized() -> None:
    grader = TherapistAssistantGrader()
    task = _task_with_config(None)

    coercive = grader.score_action(
        ValidateExperienceAction(validation_text="You have to stop resisting and do as I say."),
        task,
        phase="exploration",
        high_risk=False,
        safety_addressed=False,
        last_action_type=None,
        repeated_action_count=0,
    )
    supportive = grader.score_action(
        ValidateExperienceAction(validation_text="It makes sense this feels frustrating, and we can figure out what helps."),
        task,
        phase="exploration",
        high_risk=False,
        safety_addressed=False,
        last_action_type=None,
        repeated_action_count=0,
    )

    assert coercive.breakdown["quality"] < supportive.breakdown["quality"]


def test_vague_summary_is_penalized() -> None:
    grader = TherapistAssistantGrader()
    task = _task_with_config(None)

    vague = grader.score_action(
        SummarizeSessionAction(
            summary_text="We talked about things and I am not sure what this means.",
            key_points=["things"],
            open_items=[],
        ),
        task,
        phase="exploration",
        high_risk=False,
        safety_addressed=False,
        last_action_type=None,
        repeated_action_count=0,
    )
    actionable = grader.score_action(
        SummarizeSessionAction(
            summary_text="We identified two stress triggers and agreed on a next step plan this week.",
            key_points=["stress trigger 1", "stress trigger 2"],
            open_items=["practice plan"],
        ),
        task,
        phase="exploration",
        high_risk=False,
        safety_addressed=False,
        last_action_type=None,
        repeated_action_count=0,
    )

    assert vague.breakdown["quality"] < actionable.breakdown["quality"]


def test_modality_markers_improve_quality_for_clarify_probe() -> None:
    grader = TherapistAssistantGrader()
    task = _task_with_config(None)

    plain = grader.score_action(
        ClarifyOrProbeAction(
            question_text="Can you say more about that?",
            reason_for_probe="Clarify context.",
        ),
        task,
        phase="exploration",
        high_risk=False,
        safety_addressed=False,
        last_action_type=None,
        repeated_action_count=0,
    )
    modality_aligned = grader.score_action(
        ClarifyOrProbeAction(
            question_text="When that trigger shows up, what thought pattern appears and what coping experiment might fit this week?",
            reason_for_probe="Clarify CBT chain and next step.",
        ),
        task,
        phase="exploration",
        high_risk=False,
        safety_addressed=False,
        last_action_type=None,
        repeated_action_count=0,
    )

    assert modality_aligned.breakdown["quality"] > plain.breakdown["quality"]


def test_overdirective_summary_is_penalized() -> None:
    grader = TherapistAssistantGrader()
    task = _task_with_config(None)

    directive = grader.score_action(
        SummarizeSessionAction(
            summary_text="You should just do this plan and stop that behavior.",
            key_points=["plan"],
            open_items=[],
        ),
        task,
        phase="exploration",
        high_risk=False,
        safety_addressed=False,
        last_action_type=None,
        repeated_action_count=0,
    )
    collaborative = grader.score_action(
        SummarizeSessionAction(
            summary_text="We identified one trigger and agreed on a next step plan you want to try this week.",
            key_points=["trigger", "next step"],
            open_items=["review outcome"],
        ),
        task,
        phase="exploration",
        high_risk=False,
        safety_addressed=False,
        last_action_type=None,
        repeated_action_count=0,
    )

    assert directive.breakdown["quality"] < collaborative.breakdown["quality"]
