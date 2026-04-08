from __future__ import annotations

import asyncio
from typing import Any

from openenv.core import EnvClient

from openenv_therapist_assistant.client import OpenenvTherapistAssistantEnv


def test_client_reset_forwards_seed_and_difficulty(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    async def fake_reset(self: EnvClient, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(EnvClient, "reset", fake_reset)

    client = OpenenvTherapistAssistantEnv(base_url="http://localhost:8000")
    result = asyncio.run(client.reset(seed=7, difficulty="hard"))

    assert result == {"ok": True}
    assert captured == {"seed": 7, "difficulty": "hard"}


def test_client_reset_omits_unset_fields(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    async def fake_reset(self: EnvClient, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(EnvClient, "reset", fake_reset)

    client = OpenenvTherapistAssistantEnv(base_url="http://localhost:8000")
    asyncio.run(client.reset())

    assert captured == {}
