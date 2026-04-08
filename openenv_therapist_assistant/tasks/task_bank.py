"""Deterministic therapist task bank loaded from JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, Field, model_validator

TaskDifficulty = Literal["easy", "moderate", "hard"]

_TASKS_DIR = Path(__file__).parent
_REWARD_DIMENSIONS = {"safety", "quality", "flow", "coherence", "efficiency"}


class TaskRewardConfig(BaseModel):
    """Optional per-task reward configuration overrides."""

    model_config = {"frozen": True}

    weights: dict[str, float] = Field(default_factory=dict)
    action_bonus: dict[str, float] = Field(default_factory=dict)
    action_penalty: dict[str, float] = Field(default_factory=dict)
    required_keywords: list[str] = Field(default_factory=list)
    forbidden_phrases: list[str] = Field(default_factory=list)
    done_on_actions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_fields(self) -> "TaskRewardConfig":
        invalid_weights = set(self.weights) - _REWARD_DIMENSIONS
        if invalid_weights:
            raise ValueError(
                f"Invalid reward weight keys: {sorted(invalid_weights)}. "
                f"Allowed keys: {sorted(_REWARD_DIMENSIONS)}"
            )

        for label, mapping in (
            ("weights", self.weights),
            ("action_bonus", self.action_bonus),
            ("action_penalty", self.action_penalty),
        ):
            for key, value in mapping.items():
                if value < 0:
                    raise ValueError(f"{label}[{key}] must be non-negative")

        return self


class TherapistTask(BaseModel):
    """Training task for the therapist assistant environment."""

    model_config = {"frozen": True}

    task_id: str
    title: str
    difficulty: TaskDifficulty
    client_message: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    success_criteria: list[str] = Field(default_factory=list)
    target_actions: list[str] = Field(default_factory=list)
    risk_level: Literal["none", "low", "moderate", "high"] = "none"
    reward_config: TaskRewardConfig | None = None


class TaskBank:
    """Loads therapist tasks and selects them deterministically by seed."""

    def __init__(self, tasks_dir: Path | None = None) -> None:
        root = tasks_dir or _TASKS_DIR
        self._by_difficulty: dict[TaskDifficulty, list[TherapistTask]] = {
            "easy": self._load(root / "easy.json"),
            "moderate": self._load(root / "moderate.json"),
            "hard": self._load(root / "hard.json"),
        }
        self._all: list[TherapistTask] = (
            self._by_difficulty["easy"]
            + self._by_difficulty["moderate"]
            + self._by_difficulty["hard"]
        )
        if not self._all:
            raise ValueError(f"No therapist tasks found in {root}")

    @staticmethod
    def _load(path: Path) -> list[TherapistTask]:
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        return [TherapistTask.model_validate(item) for item in raw]

    def get_task(self, seed: int = 0, difficulty: str | None = None) -> TherapistTask:
        """Select a task deterministically. Same seed -> same task."""
        if difficulty is not None:
            pool = self._by_difficulty.get(difficulty)  # type: ignore[arg-type]
            if not pool:
                raise ValueError(f"No therapist tasks for difficulty '{difficulty}'")
        else:
            pool = self._all
        return pool[seed % len(pool)]

    def list_tasks(self, difficulty: str | None = None) -> Sequence[TherapistTask]:
        """Return all tasks, optionally filtered by difficulty."""
        if difficulty is not None:
            return list(self._by_difficulty.get(difficulty, []))  # type: ignore[arg-type]
        return list(self._all)