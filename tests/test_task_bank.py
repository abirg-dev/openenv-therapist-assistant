from __future__ import annotations

import json
from pathlib import Path

import pytest

from openenv_therapist_assistant import TaskBank, TherapistTask


def test_task_bank_loads_all_tasks() -> None:
    bank = TaskBank()

    tasks = bank.list_tasks()

    assert len(tasks) == 9
    assert all(isinstance(task, TherapistTask) for task in tasks)


def test_task_bank_filters_by_difficulty() -> None:
    bank = TaskBank()

    easy_tasks = bank.list_tasks("easy")
    hard_tasks = bank.list_tasks("hard")

    assert len(easy_tasks) == 3
    assert len(hard_tasks) == 3
    assert {task.difficulty for task in easy_tasks} == {"easy"}
    assert {task.difficulty for task in hard_tasks} == {"hard"}


def test_task_bank_is_deterministic() -> None:
    bank = TaskBank()

    first = bank.get_task(seed=5, difficulty="moderate")
    second = bank.get_task(seed=5, difficulty="moderate")

    assert first == second


def test_task_payloads_include_therapeutic_fields() -> None:
    bank = TaskBank()

    task = bank.get_task(seed=0, difficulty="easy")

    assert task.client_message
    assert task.objective
    assert task.success_criteria
    assert task.target_actions


def test_task_payloads_can_include_reward_config() -> None:
    bank = TaskBank()

    task = bank.get_task(seed=0, difficulty="easy")

    assert task.reward_config is not None
    assert task.reward_config.weights
    assert task.reward_config.action_bonus


def test_invalid_reward_config_weight_key_fails(tmp_path: Path) -> None:
    tasks_dir = tmp_path
    bad_task = {
        "task_id": "EASY-X",
        "title": "Bad reward config",
        "difficulty": "easy",
        "client_message": "hello",
        "objective": "test",
        "target_actions": ["check_in"],
        "reward_config": {
            "weights": {"not_a_dimension": 0.8},
        },
    }
    for name in ("easy.json", "moderate.json", "hard.json"):
        payload = [bad_task] if name == "easy.json" else []
        (tasks_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        TaskBank(tasks_dir=tasks_dir)