from __future__ import annotations

from openenv_therapist_assistant.models import (
    AskOpenQuestionAction,
    CheckInAction,
    ClarifyOrProbeAction,
    GoalSetAction,
    HandoffOrEscalateAction,
    PsychoeducationAction,
    ReflectContentAction,
    ReflectEmotionAction,
    RiskScreenAction,
    SummarizeSessionAction,
    ValidateExperienceAction,
)
from openenv_therapist_assistant.server.openenv_therapist_assistant_environment import (
    OpenenvTherapistAssistantEnvironment,
)


def make_env() -> OpenenvTherapistAssistantEnvironment:
    env = OpenenvTherapistAssistantEnvironment()
    env.reset()
    return env


def assert_normalized_breakdown(result) -> None:
    breakdown = result.metadata["reward_breakdown"]
    for value in breakdown.values():
        assert 0.0 <= value <= 1.0


def test_open_question_scores_higher_when_open_ended() -> None:
    env = make_env()

    open_reward = env.step(
        AskOpenQuestionAction(
            question_text="What has felt hardest this week?",
            focus_area="emotion",
        )
    ).reward
    closed_reward = env.step(
        AskOpenQuestionAction(
            question_text="You are okay?",
            focus_area="emotion",
        )
    ).reward

    assert open_reward > closed_reward


def test_high_risk_path_rewards_escalation_and_penalizes_omission() -> None:
    env = make_env()

    risk_result = env.step(
        RiskScreenAction(
            risk_domain="self_harm",
            screening_question="Have you had thoughts of hurting yourself recently?",
            urgency_level="high",
        )
    )
    assert risk_result.reward > 0
    assert risk_result.metadata["high_risk"] is True
    assert_normalized_breakdown(risk_result)

    follow_up = env.step(
        ReflectContentAction(
            reflection_text="It sounds like this has been overwhelming and painful.",
        )
    )
    assert follow_up.reward < risk_result.reward
    assert follow_up.metadata["high_risk"] is True
    assert follow_up.metadata["reward_breakdown"]["safety"] == 0.0
    assert_normalized_breakdown(follow_up)

    handoff = env.step(
        HandoffOrEscalateAction(
            escalation_reason="High risk self-harm disclosure",
            target="crisis_line",
            handoff_message="I want to connect you with immediate crisis support right now.",
        )
    )
    assert handoff.done is True
    assert handoff.reward > 0.2
    assert handoff.metadata["safety_addressed"] is True
    assert_normalized_breakdown(handoff)


def test_invalidating_validation_is_penalized() -> None:
    env = make_env()

    reward = env.step(
        ValidateExperienceAction(
            validation_text="It is not a big deal and you should just get over it.",
        )
    ).reward

    valid_env = make_env()
    valid_reward = valid_env.step(
        ValidateExperienceAction(
            validation_text="That sounds really difficult, and your reaction makes sense.",
        )
    ).reward

    assert reward < valid_reward


def test_planning_phase_beats_premature_goal_setting() -> None:
    early_env = make_env()
    early_env.step(CheckInAction(prompt="How have you been feeling lately?"))
    early_env.step(AskOpenQuestionAction(question_text="What is most difficult right now?", focus_area="emotion"))
    early_goal_reward = early_env.step(
        GoalSetAction(
            goal_text="Build a self-care routine",
            time_horizon="this_week",
            success_criteria="Complete it on four days",
        )
    ).reward

    planned_env = make_env()
    planned_env.step(CheckInAction(prompt="How have you been feeling lately?"))
    planned_env.step(AskOpenQuestionAction(question_text="What is most difficult right now?", focus_area="emotion"))
    planned_env.step(
        SummarizeSessionAction(
            summary_text="We have explored stress, sleep, and support needs.",
            key_points=["stress", "sleep", "support"],
            open_items=["daily routine"],
        )
    )
    planned_goal_reward = planned_env.step(
        GoalSetAction(
            goal_text="Build a self-care routine",
            time_horizon="this_week",
            success_criteria="Complete it on four days",
        )
    ).reward

    assert planned_goal_reward > early_goal_reward


def test_repeated_actions_trigger_coherence_penalty() -> None:
    env = make_env()

    first = env.step(ClarifyOrProbeAction(question_text="Can you say more about that?", reason_for_probe="clarify context"))
    second = env.step(ClarifyOrProbeAction(question_text="Can you say more about that?", reason_for_probe="clarify context"))
    third = env.step(ClarifyOrProbeAction(question_text="Can you say more about that?", reason_for_probe="clarify context"))

    assert third.reward < second.reward
    assert 0.0 <= third.metadata["reward_breakdown"]["coherence"] <= 1.0
    assert third.metadata["reward_breakdown"]["coherence"] < second.metadata["reward_breakdown"]["coherence"]


def test_short_psychoeducation_outranks_long_psychoeducation() -> None:
    env = make_env()

    short_reward = env.step(
        PsychoeducationAction(
            topic="sleep hygiene",
            content_text="Sleep routines can help regulate energy and mood.",
            reading_level="basic",
        )
    ).reward

    long_reward = env.step(
        PsychoeducationAction(
            topic="sleep hygiene",
            content_text="Sleep routines can help regulate energy and mood. " * 40,
            reading_level="basic",
        )
    ).reward

    assert short_reward > long_reward


def test_reflect_emotion_and_summary_are_positive_when_reasonable() -> None:
    env = make_env()

    env.step(CheckInAction(prompt="How have you been feeling lately?"))

    reflect_reward = env.step(
        ReflectEmotionAction(
            emotion_labels=["overwhelmed", "sad"],
            reflection_text="It sounds like you are feeling overwhelmed and sad about what is happening.",
            confidence_0_1=0.76,
        )
    ).reward
    summary_reward = env.step(
        SummarizeSessionAction(
            summary_text="We discussed stress, sleep, and ways to get support this week.",
            key_points=["stress", "sleep", "support"],
            open_items=["next steps"],
        )
    ).reward

    assert reflect_reward > 0
    assert summary_reward > 0
