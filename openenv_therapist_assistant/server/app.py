# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Openenv Therapist Assistant Environment.

This module creates an HTTP server that exposes the OpenenvTherapistAssistantEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn openenv_therapist_assistant.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn openenv_therapist_assistant.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m openenv_therapist_assistant.server.app
"""

import json
import os
from pathlib import Path
from typing import Any, Literal

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import (
        OpenenvTherapistAssistantActionModel,
        OpenenvTherapistAssistantObservation,
    )
    from .openenv_therapist_assistant_environment import OpenenvTherapistAssistantEnvironment
except ModuleNotFoundError:
    from models import OpenenvTherapistAssistantActionModel, OpenenvTherapistAssistantObservation
    from server.openenv_therapist_assistant_environment import OpenenvTherapistAssistantEnvironment

from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field


# Ensure the OpenEnv web UI can always locate README content inside the container.
os.environ.setdefault(
    "ENV_README_PATH",
    str(Path(__file__).resolve().parent.parent / "README.md"),
)

# Force the plain FastAPI app so our custom therapist UI is visible on Spaces.
# The framework web UI would otherwise own /web and shadow this app's therapist-facing page.
os.environ["ENABLE_WEB_INTERFACE"] = "false"


# Create the app with web interface and README integration
app = create_app(
    OpenenvTherapistAssistantEnvironment,
    OpenenvTherapistAssistantActionModel,
    OpenenvTherapistAssistantObservation,
    env_name="openenv_therapist_assistant",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


class WebResetRequest(BaseModel):
    """Reset payload for the stateful /web demo endpoints."""

    seed: int | None = Field(default=None, ge=0)
    difficulty: Literal["easy", "moderate", "hard"] | None = None
    modality_profile: Literal["balanced", "mi_leaning", "cbt_leaning", "psychodynamic_leaning"] | None = None


class WebStepRequest(BaseModel):
    """Step payload for the stateful /web demo endpoints."""

    action: dict[str, Any]


_web_env = OpenenvTherapistAssistantEnvironment()


def _serialize_step_result(observation: OpenenvTherapistAssistantObservation) -> dict[str, Any]:
    """Serialize observation without dropping metadata for the custom web UI."""
    obs_payload = observation.model_dump()
    done = bool(obs_payload.pop("done", False))
    reward = obs_payload.pop("reward", None)
    return {
        "observation": obs_payload,
        "reward": reward,
        "done": done,
    }


@app.post("/web/reset")
async def web_reset(request: WebResetRequest) -> dict[str, Any]:
    observation = _web_env.reset(
        seed=request.seed,
        difficulty=request.difficulty,
        modality_profile=request.modality_profile,
    )
    return _serialize_step_result(observation)


@app.post("/web/step")
async def web_step(request: WebStepRequest) -> dict[str, Any]:
    observation = _web_env.step(request.action)
    return _serialize_step_result(observation)


@app.get("/web/state")
async def web_state() -> dict[str, Any]:
    state = _web_env.state
    return {
        "episode_id": state.episode_id,
        "step_count": state.step_count,
    }


_WEB_ACTION_FIELDS: dict[str, list[dict[str, object]]] = {
        "check_in": [
                {"name": "prompt", "kind": "textarea", "required": True, "max": 500},
                {"name": "mood_scale_1_10", "kind": "int", "required": False, "min": 1, "max": 10},
                {"name": "energy_scale_1_10", "kind": "int", "required": False, "min": 1, "max": 10},
                {"name": "sleep_quality_1_10", "kind": "int", "required": False, "min": 1, "max": 10},
        ],
        "ask_open_question": [
                {"name": "question_text", "kind": "textarea", "required": True, "max": 1000},
                {
                        "name": "focus_area",
                        "kind": "enum",
                        "required": True,
                        "options": ["emotion", "event", "relationship", "coping"],
                },
        ],
        "reflect_content": [
                {"name": "reflection_text", "kind": "textarea", "required": True, "max": 1000},
                {"name": "source_span", "kind": "text", "required": False, "max": 500},
        ],
        "reflect_emotion": [
                {"name": "emotion_labels", "kind": "list", "required": True, "max_items": 6},
                {"name": "reflection_text", "kind": "textarea", "required": True, "max": 1000},
                {
                        "name": "confidence_0_1",
                        "kind": "float",
                        "required": True,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                },
        ],
        "validate_experience": [
                {"name": "validation_text", "kind": "textarea", "required": True, "max": 1000},
                {"name": "context_reference", "kind": "text", "required": False, "max": 500},
        ],
        "summarize_session": [
                {"name": "summary_text", "kind": "textarea", "required": True, "max": 2000},
                {"name": "key_points", "kind": "list", "required": False, "max_items": 10},
                {"name": "open_items", "kind": "list", "required": False, "max_items": 10},
        ],
        "clarify_or_probe": [
                {"name": "question_text", "kind": "textarea", "required": True, "max": 1000},
                {"name": "reason_for_probe", "kind": "text", "required": True, "max": 500},
        ],
        "risk_screen": [
                {
                        "name": "risk_domain",
                        "kind": "enum",
                        "required": True,
                        "options": ["self_harm", "harm_to_others", "abuse", "substance"],
                },
                {"name": "screening_question", "kind": "textarea", "required": True, "max": 1000},
                {
                        "name": "urgency_level",
                        "kind": "enum",
                        "required": True,
                        "options": ["low", "medium", "high", "immediate"],
                },
        ],
        "safety_plan_step": [
                {
                        "name": "step_type",
                        "kind": "enum",
                        "required": True,
                        "options": [
                                "coping_strategy",
                                "support_contact",
                                "means_restriction",
                                "crisis_resource",
                        ],
                },
                {"name": "step_text", "kind": "textarea", "required": True, "max": 1000},
        ],
        "psychoeducation": [
                {"name": "topic", "kind": "text", "required": True, "max": 120},
                {"name": "content_text", "kind": "textarea", "required": True, "max": 2000},
                {
                        "name": "reading_level",
                        "kind": "enum",
                        "required": False,
                        "default": "basic",
                        "options": ["basic", "intermediate", "advanced"],
                },
        ],
        "coping_skill_coach": [
                {"name": "skill_name", "kind": "text", "required": True, "max": 120},
                {"name": "instructions", "kind": "textarea", "required": True, "max": 2000},
                {"name": "duration_minutes", "kind": "int", "required": True, "min": 1, "max": 120},
        ],
        "goal_set": [
                {"name": "goal_text", "kind": "textarea", "required": True, "max": 1000},
                {
                        "name": "time_horizon",
                        "kind": "enum",
                        "required": True,
                        "options": ["this_week", "this_month", "next_3_months", "long_term"],
                },
                {"name": "success_criteria", "kind": "textarea", "required": True, "max": 1000},
        ],
        "home_practice_assign": [
                {"name": "practice_name", "kind": "text", "required": True, "max": 120},
                {"name": "instructions", "kind": "textarea", "required": True, "max": 2000},
                {"name": "frequency", "kind": "text", "required": True, "max": 120},
                {"name": "tracking_method", "kind": "text", "required": True, "max": 300},
        ],
        "resource_recommend": [
                {
                        "name": "resource_type",
                        "kind": "enum",
                        "required": True,
                        "options": ["hotline", "worksheet", "community_service", "app"],
                },
                {"name": "resource_details", "kind": "textarea", "required": True, "max": 1000},
                {"name": "reason", "kind": "text", "required": True, "max": 500},
        ],
        "handoff_or_escalate": [
                {"name": "escalation_reason", "kind": "text", "required": True, "max": 500},
                {
                        "name": "target",
                        "kind": "enum",
                        "required": True,
                        "options": ["licensed_therapist", "emergency_services", "crisis_line"],
                },
                {"name": "handoff_message", "kind": "textarea", "required": True, "max": 1000},
        ],
}

_WEB_UI_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Therapist Assistant Demo</title>
    <style>
        :root {
            --bg: #f4f3ef;
            --paper: #fffdf8;
            --ink: #1f2d2b;
            --muted: #5d736e;
            --accent: #0f766e;
            --accent-2: #c2410c;
            --line: #d9d3c7;
            --good: #276749;
            --bad: #9b2c2c;
            --mono: ui-monospace, "Cascadia Mono", "SFMono-Regular", Menlo, Consolas, monospace;
            --serif: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            color: var(--ink);
            background:
                radial-gradient(circle at 0% 0%, #ede7db 0%, transparent 35%),
                radial-gradient(circle at 100% 100%, #f3e9d8 0%, transparent 30%),
                var(--bg);
            font-family: var(--serif);
            min-height: 100vh;
        }
        .shell {
            max-width: 1160px;
            margin: 0 auto;
            padding: 1.25rem 1rem 2rem;
        }
        .hero {
            border: 1px solid var(--line);
            background: linear-gradient(120deg, #fffef9 0%, #f4efe6 100%);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 28px rgba(37, 43, 42, 0.08);
        }
        .hero h1 {
            margin: 0;
            font-size: clamp(1.35rem, 3.2vw, 2rem);
            letter-spacing: 0.01em;
            font-weight: 700;
        }
        .hero p {
            margin: 0.45rem 0 0;
            color: var(--muted);
            font-size: 0.95rem;
        }
        .grid {
            display: grid;
            gap: 0.9rem;
            grid-template-columns: 1.35fr 1fr;
        }
        .card {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: var(--paper);
            padding: 0.9rem;
            box-shadow: 0 6px 20px rgba(31, 45, 43, 0.05);
        }
        .card h2 {
            margin: 0 0 0.65rem;
            font-size: 1rem;
            color: #153530;
        }
        .row {
            display: grid;
            gap: 0.6rem;
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .field {
            margin-bottom: 0.55rem;
        }
        label {
            display: block;
            font-size: 0.78rem;
            color: var(--muted);
            margin: 0 0 0.25rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }
        .required {
            color: var(--accent-2);
        }
        input[type="text"],
        input[type="number"],
        select,
        textarea {
            width: 100%;
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 0.5rem 0.55rem;
            background: #fff;
            color: var(--ink);
            font-family: inherit;
            font-size: 0.94rem;
        }
        textarea {
            min-height: 6rem;
            resize: vertical;
            font-family: var(--mono);
            font-size: 0.82rem;
        }
        .hint {
            font-size: 0.76rem;
            color: var(--muted);
            margin-top: 0.2rem;
        }
        .controls {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: 0.25rem;
        }
        button {
            border: 0;
            border-radius: 999px;
            padding: 0.5rem 0.85rem;
            font-weight: 700;
            cursor: pointer;
            color: #fff;
            background: var(--accent);
            transition: transform 120ms ease, opacity 120ms ease;
        }
        button.secondary {
            background: #3b5f59;
        }
        button.ghost {
            background: #7c5f3c;
        }
        button:disabled {
            opacity: 0.45;
            cursor: not-allowed;
            transform: none;
        }
        button:hover:not(:disabled) {
            transform: translateY(-1px);
        }
        .status {
            margin-top: 0.55rem;
            font-size: 0.85rem;
            padding: 0.45rem 0.6rem;
            border-radius: 8px;
            border: 1px solid #c6d3cf;
            background: #ebf5f2;
        }
        .status.error {
            color: var(--bad);
            border-color: #f0b3b3;
            background: #fff2f2;
        }
        .status.ok {
            color: var(--good);
        }
        .mono {
            font-family: var(--mono);
            font-size: 0.8rem;
            background: #f7f5ef;
            border: 1px solid #e4ded1;
            border-radius: 10px;
            padding: 0.6rem;
            white-space: pre-wrap;
            max-height: 14rem;
            overflow: auto;
        }
        .timeline {
            max-height: 18rem;
        }
        .mini {
            font-size: 0.8rem;
            color: var(--muted);
        }
        @media (max-width: 900px) {
            .grid {
                grid-template-columns: 1fr;
            }
            .row {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="shell">
        <section class="hero">
            <h1>Therapist Assistant Environment Demo</h1>
            <p>Use this workspace to reset episodes, compose any therapist action, and inspect rewards plus metadata in real time.</p>
        </section>

        <div class="grid">
            <section class="card">
                <h2>Episode Controls</h2>
                <div class="row">
                    <div class="field">
                        <label for="seed">seed</label>
                        <input id="seed" type="number" value="0" />
                        <div class="hint">Deterministic task selection.</div>
                    </div>
                    <div class="field">
                        <label for="difficulty">difficulty</label>
                        <select id="difficulty">
                            <option value="">any</option>
                            <option value="easy">easy</option>
                            <option value="moderate">moderate</option>
                            <option value="hard">hard</option>
                        </select>
                        <div class="hint" id="taskTypeHint">Any task type: rotates through easy, moderate, and hard tasks.</div>
                    </div>
                </div>
                <div class="row">
                    <div class="field">
                        <label for="modalityProfile">modality_profile</label>
                        <select id="modalityProfile">
                            <option value="">default (balanced)</option>
                            <option value="balanced">balanced</option>
                            <option value="mi_leaning">mi_leaning</option>
                            <option value="cbt_leaning">cbt_leaning</option>
                            <option value="psychodynamic_leaning">psychodynamic_leaning</option>
                        </select>
                        <div class="hint" id="modalityHint">Balanced profile: equal emphasis across MI, CBT, and psychodynamic markers.</div>
                    </div>
                </div>
                <div class="controls">
                    <button id="btnReset" type="button">Reset</button>
                    <button id="btnState" type="button" class="secondary">State</button>
                    <button id="btnTemplate" type="button" class="ghost">Load Sample</button>
                </div>
                <div class="field" style="margin-top: 0.45rem;">
                    <label for="verboseMode">verbose output</label>
                    <input id="verboseMode" type="checkbox" />
                    <div class="hint">Off by default for therapist-facing summaries. Reward details are always shown.</div>
                </div>
                <div id="status" class="status">Ready.</div>
            </section>

            <section class="card">
                <h2>Session Snapshot</h2>
                <div class="mini" id="snapshotText">No active episode yet.</div>
                <div class="mono" id="taskPanel">{}</div>
            </section>
        </div>

        <div class="grid" style="margin-top: 0.9rem;">
            <section class="card">
                <h2>Action Composer</h2>
                <div class="field">
                    <label for="actionType">action_type <span class="required">required</span></label>
                    <select id="actionType"></select>
                    <div class="hint" id="actionTypeHint">Select an action to see field requirements and sample payloads.</div>
                </div>
                <div id="fieldContainer"></div>
                <div class="controls">
                    <button id="btnStep" type="button">Step</button>
                </div>
                <div class="hint">List fields accept comma-separated values.</div>
            </section>

            <section class="card">
                <h2>Payload Preview</h2>
                <div class="mono" id="payloadPreview">{}</div>
            </section>
        </div>

        <div class="grid" style="margin-top: 0.9rem;">
            <section class="card">
                <h2>Latest Response</h2>
                <div class="mono" id="responsePanel">{}</div>
            </section>
            <section class="card">
                <h2>Timeline</h2>
                <div class="mono timeline" id="timeline"></div>
            </section>
        </div>
    </div>

    <script>
    (() => {
        __ACTION_FIELDS_CODE__;
        const actionTypes = Object.keys(ACTION_FIELDS);
        const state = {
            active: false,
            stepCount: 0,
            episodeId: null,
            phase: null,
            verboseMode: false,
            modalityProfile: "balanced",
            lastActions: [],
            taskTargets: [],
            taskDoneActions: [],
            highRisk: false,
            safetyAddressed: false,
            currentTask: {},
            lastResponse: null,
        };

        const $ = (id) => document.getElementById(id);

        const actionTypeSelect = $("actionType");
        const fieldContainer = $("fieldContainer");
        const statusEl = $("status");
        const payloadPreview = $("payloadPreview");
        const responsePanel = $("responsePanel");
        const timeline = $("timeline");
        const taskPanel = $("taskPanel");
        const snapshotText = $("snapshotText");
        const btnStep = $("btnStep");
        const actionTypeHint = $("actionTypeHint");
        const taskTypeHint = $("taskTypeHint");
        const modalityHint = $("modalityHint");
        const verboseToggle = $("verboseMode");

        const ACTION_TYPE_DESCRIPTIONS = {
            check_in: "Quick emotional/functional check to open the session and establish current state.",
            ask_open_question: "Invite expansive client reflection with a non-leading, open-ended question.",
            reflect_content: "Mirror key facts/themes to show understanding and improve shared clarity.",
            reflect_emotion: "Name and reflect likely emotions to support emotional attunement.",
            validate_experience: "Communicate that the client's reactions make sense in context.",
            summarize_session: "Synthesize key insights and decisions into a coherent summary.",
            clarify_or_probe: "Ask targeted follow-up to resolve ambiguity and deepen case understanding.",
            risk_screen: "Assess risk directly and explicitly when safety concerns may be present.",
            safety_plan_step: "Create one concrete, actionable step in a safety plan.",
            psychoeducation: "Provide concise educational context to normalize and inform coping.",
            coping_skill_coach: "Teach and rehearse a practical coping strategy with clear instructions.",
            goal_set: "Define a specific, measurable near-term therapeutic goal.",
            home_practice_assign: "Assign between-session practice with clear tracking expectations.",
            resource_recommend: "Recommend a relevant external resource and explain why it fits.",
            handoff_or_escalate: "Escalate or hand off care when scope/safety requires higher support.",
        };

        const TASK_TYPE_DESCRIPTIONS = {
            any: "Any task type: rotates through easy, moderate, and hard tasks.",
            easy: "Easy: rapport building, basic problem clarification, and simple goal framing.",
            moderate: "Moderate: ambiguity reduction, prioritization, and structured planning.",
            hard: "Hard: elevated risk, resistance, and complex formulation under pressure.",
        };

        const MODALITY_PROFILE_DESCRIPTIONS = {
            balanced: "Balanced profile: equal emphasis across MI, CBT, and psychodynamic markers.",
            mi_leaning: "MI-leaning: stronger reward for collaborative, autonomy-supportive language.",
            cbt_leaning: "CBT-leaning: stronger reward for trigger-pattern, evidence, and coping framing.",
            psychodynamic_leaning: "Psychodynamic-leaning: stronger reward for meaning, attachment, and defenses.",
        };

        const SAMPLE_BY_ACTION = {
            check_in: {
                prompt: "How has your mood shifted since yesterday?",
                mood_scale_1_10: 4,
                energy_scale_1_10: 5,
            },
            ask_open_question: {
                question_text: "What part of this situation feels the heaviest right now?",
                focus_area: "emotion",
            },
            reflect_content: {
                reflection_text: "You have been trying to hold everything together at work and at home.",
                source_span: "I am exhausted from juggling too many responsibilities.",
            },
            reflect_emotion: {
                emotion_labels: ["anxious", "overwhelmed"],
                reflection_text: "It sounds like you are carrying a lot and feeling stretched thin.",
                confidence_0_1: 0.76,
            },
            validate_experience: {
                validation_text: "Given what you have been dealing with, your reaction makes a lot of sense.",
                context_reference: "high workload and poor sleep",
            },
            summarize_session: {
                summary_text: "Today we identified your stress triggers, clarified early warning signs, and agreed on a short coping plan.",
                key_points: ["stress peaks in late afternoon", "sleep debt worsens mood"],
                open_items: ["review boundary-setting script next session"],
            },
            clarify_or_probe: {
                question_text: "When your stress spikes, what thoughts usually show up first?",
                reason_for_probe: "Clarify automatic thought patterns before choosing interventions.",
            },
            risk_screen: {
                risk_domain: "self_harm",
                screening_question: "Have you had thoughts of hurting yourself this week?",
                urgency_level: "medium",
            },
            safety_plan_step: {
                step_type: "support_contact",
                step_text: "Text your sister and let her know you may need a check-in tonight.",
            },
            psychoeducation: {
                topic: "stress response",
                content_text: "When stress stays high, your body can remain in threat mode, which makes focus and sleep harder. Short grounding drills can help reset that cycle.",
                reading_level: "basic",
            },
            coping_skill_coach: {
                skill_name: "box breathing",
                instructions: "Inhale for 4 counts, hold for 4, exhale for 4, hold for 4. Repeat for 5 rounds while relaxing shoulders.",
                duration_minutes: 5,
            },
            goal_set: {
                goal_text: "Practice one grounding exercise daily after work.",
                time_horizon: "this_week",
                success_criteria: "Complete at least 5 days and record stress level before and after.",
            },
            home_practice_assign: {
                practice_name: "Evening wind-down log",
                instructions: "Spend 5 minutes each night noting stress level, one trigger, and one coping step you used.",
                frequency: "daily",
                tracking_method: "notes app checklist",
            },
            resource_recommend: {
                resource_type: "worksheet",
                resource_details: "CBT thought record worksheet for identifying trigger-thought-feeling links.",
                reason: "Supports between-session reflection and pattern tracking.",
            },
            handoff_or_escalate: {
                escalation_reason: "Client reports escalating risk and limited immediate supports.",
                target: "crisis_line",
                handoff_message: "I want to connect you to trained crisis support right now so you are not carrying this alone.",
            },
        };

        function setStatus(message, isError = false) {
            statusEl.textContent = message;
            statusEl.className = isError ? "status error" : "status ok";
        }

        function unique(items) {
            return Array.from(new Set(items.filter(Boolean)));
        }

        function recommendActionType(phase, lastActions, taskTargets, taskDoneActions, stepCount, highRisk, safetyAddressed) {
            const lastAction = lastActions.length ? lastActions[lastActions.length - 1] : null;
            const recentActions = lastActions.slice(-3);
            const validActions = new Set(Object.keys(SAMPLE_BY_ACTION));
            const actionCounts = {};
            for (const action of lastActions) {
                actionCounts[action] = (actionCounts[action] || 0) + 1;
            }

            const phaseDefaults = {
                engagement: ["check_in", "ask_open_question", "reflect_emotion", "validate_experience"],
                exploration: ["ask_open_question", "clarify_or_probe", "reflect_content", "reflect_emotion", "summarize_session"],
                planning: ["goal_set", "home_practice_assign", "summarize_session", "coping_skill_coach", "resource_recommend"],
            };

            function pickBest(candidates, banned = new Set()) {
                const uniqueCandidates = unique(candidates).filter((candidate) => validActions.has(candidate) && !banned.has(candidate));
                if (!uniqueCandidates.length) {
                    return null;
                }

                const scored = uniqueCandidates
                    .map((candidate) => {
                        const frequencyPenalty = (actionCounts[candidate] || 0) * 10;
                        const immediateRepeatPenalty = candidate === lastAction ? 100 : 0;
                        const recentPenalty = recentActions.includes(candidate) ? 4 : 0;
                        return {
                            candidate,
                            score: frequencyPenalty + immediateRepeatPenalty + recentPenalty,
                        };
                    })
                    .sort((a, b) => a.score - b.score);

                return scored[0].candidate;
            }

            const doneCandidates = Array.isArray(taskDoneActions) ? taskDoneActions.filter((candidate) => validActions.has(candidate)) : [];

            // Break ABAB-style loops by banning the alternating pair temporarily.
            const loopBanned = new Set();
            if (
                lastActions.length >= 4
                && lastActions[lastActions.length - 1] === lastActions[lastActions.length - 3]
                && lastActions[lastActions.length - 2] === lastActions[lastActions.length - 4]
                && lastActions[lastActions.length - 1] !== lastActions[lastActions.length - 2]
            ) {
                loopBanned.add(lastActions[lastActions.length - 1]);
                loopBanned.add(lastActions[lastActions.length - 2]);
            }

            if (highRisk) {
                const crisisCandidates = [
                    safetyAddressed ? "handoff_or_escalate" : "risk_screen",
                    "safety_plan_step",
                    "handoff_or_escalate",
                    ...doneCandidates,
                ];
                const crisisPick = pickBest(crisisCandidates, loopBanned);
                if (crisisPick) {
                    return crisisPick;
                }
            }

            // Once session has progressed, prefer explicit terminating actions from the task.
            if (doneCandidates.length && (phase === "planning" || stepCount >= 4)) {
                const donePick = pickBest(doneCandidates, loopBanned);
                if (donePick) {
                    return donePick;
                }
            }

            let transitionSuggestion = null;
            if (phase === "engagement") {
                if (lastAction === "check_in") transitionSuggestion = "ask_open_question";
                else if (lastAction === "ask_open_question") transitionSuggestion = "reflect_emotion";
                else transitionSuggestion = "check_in";
            } else if (phase === "exploration") {
                if (lastAction === "ask_open_question") transitionSuggestion = "reflect_emotion";
                else if (lastAction === "reflect_emotion") transitionSuggestion = "clarify_or_probe";
                else if (lastAction === "clarify_or_probe") transitionSuggestion = "summarize_session";
                else transitionSuggestion = "ask_open_question";
            } else if (phase === "planning") {
                if (lastAction === "summarize_session") transitionSuggestion = "goal_set";
                else if (lastAction === "goal_set") transitionSuggestion = "home_practice_assign";
                else transitionSuggestion = "goal_set";
            }

            const rankedCandidates = unique([
                transitionSuggestion,
                ...(phaseDefaults[phase] || []),
                ...(Array.isArray(taskTargets) ? taskTargets : []),
                ...doneCandidates,
                "check_in",
                "ask_open_question",
                "reflect_emotion",
                "goal_set",
                "summarize_session",
            ]);

            const rankedPick = pickBest(rankedCandidates, loopBanned);
            if (rankedPick) {
                return rankedPick;
            }

            return rankedCandidates.find((candidate) => candidate in SAMPLE_BY_ACTION) || actionTypeSelect.value;
        }

        function loadSampleForAction(actionType, { showStatus = true } = {}) {
            if (!actionType || !(actionType in SAMPLE_BY_ACTION)) {
                if (showStatus) {
                    setStatus("No sample preset for this action type.", true);
                }
                return null;
            }

            if (actionTypeSelect.value !== actionType) {
                actionTypeSelect.value = actionType;
                renderFields();
            }

            const sample = SAMPLE_BY_ACTION[actionType];
            Object.entries(sample).forEach(([name, value]) => {
                const input = $("f_" + name);
                if (!input) {
                    return;
                }
                if (Array.isArray(value)) {
                    input.value = value.join(", ");
                } else {
                    input.value = String(value);
                }
            });

            syncPreview();
            if (showStatus) {
                setStatus("Sample payload loaded for " + actionType + ".");
            }
            return actionType;
        }

        function preloadRecommendedSample(source) {
            const recommended = recommendActionType(
                state.phase,
                state.lastActions,
                state.taskTargets,
                state.taskDoneActions,
                state.stepCount,
                state.highRisk,
                state.safetyAddressed,
            );
            const loaded = loadSampleForAction(recommended, { showStatus: false });
            if (!loaded) {
                return null;
            }
            return loaded + (source ? " (" + source + ")" : "");
        }

        function asPrettyJson(value) {
            return JSON.stringify(value, null, 2);
        }

        function formatTaskForTherapist(task) {
            const rewardConfig = task && task.reward_config ? task.reward_config : {};
            return {
                task_id: task.task_id || null,
                title: task.title || "",
                client_message: task.client_message || "",
                objective: task.objective || "",
                success_criteria: task.success_criteria || [],
                target_actions: task.target_actions || [],
                risk_level: task.risk_level || "",
                reward_info: {
                    weights: rewardConfig.weights || {},
                    action_bonus: rewardConfig.action_bonus || {},
                    action_penalty: rewardConfig.action_penalty || {},
                    done_on_actions: rewardConfig.done_on_actions || [],
                    required_keywords: rewardConfig.required_keywords || [],
                    forbidden_phrases: rewardConfig.forbidden_phrases || [],
                },
            };
        }

        function formatResponseForTherapist(responseData) {
            const observation = responseData && responseData.observation ? responseData.observation : {};
            const metadata = observation && observation.metadata ? observation.metadata : {};
            const rewardBreakdown = metadata.reward_breakdown || {};

            return {
                therapist_view: {
                    echoed_message: observation.echoed_message || "",
                    done: !!responseData.done,
                    phase: metadata.phase || null,
                    step: metadata.step || null,
                    high_risk: !!metadata.high_risk,
                    safety_addressed: !!metadata.safety_addressed,
                    last_action_type: metadata.last_action_type || null,
                    modality_profile: metadata.modality_profile || null,
                },
                reward_info: {
                    reward: responseData.reward,
                    reward_breakdown: rewardBreakdown,
                    total: typeof rewardBreakdown.total === "number"
                        ? rewardBreakdown.total
                        : responseData.reward,
                },
            };
        }

        function renderTaskPanel(task) {
            if (state.verboseMode) {
                taskPanel.textContent = asPrettyJson(task || {});
                return;
            }
            taskPanel.textContent = asPrettyJson(formatTaskForTherapist(task || {}));
        }

        function renderResponsePanel(data) {
            if (state.verboseMode) {
                responsePanel.textContent = asPrettyJson(data || {});
                return;
            }
            responsePanel.textContent = asPrettyJson(formatResponseForTherapist(data || {}));
        }

        function renderPanels() {
            renderTaskPanel(state.currentTask || {});
            renderResponsePanel(state.lastResponse || {});
        }

        function logTimeline(label, payload) {
            const timelinePayload = state.verboseMode ? payload : formatResponseForTherapist(payload || {});
            const line = "[" + new Date().toLocaleTimeString() + "] " + label + "\\n" + asPrettyJson(timelinePayload) + "\\n\\n";
            timeline.textContent = line + timeline.textContent;
        }

        function updateSnapshot(extra) {
            const text = [
                "active=" + state.active,
                "step_count=" + state.stepCount,
                "episode_id=" + (state.episodeId || "-")
            ];
            if (extra) {
                text.push(extra);
            }
            snapshotText.textContent = text.join(" | ");
            btnStep.disabled = !state.active;
        }

        function updateActionTypeHint() {
            const selected = actionTypeSelect.value;
            actionTypeHint.textContent = ACTION_TYPE_DESCRIPTIONS[selected]
                || "Select an action to see field requirements and sample payloads.";
        }

        function updateTaskTypeHint() {
            const selected = $("difficulty").value || "any";
            taskTypeHint.textContent = TASK_TYPE_DESCRIPTIONS[selected] || TASK_TYPE_DESCRIPTIONS.any;
        }

        function updateModalityHint() {
            const selected = $("modalityProfile").value || "balanced";
            modalityHint.textContent = MODALITY_PROFILE_DESCRIPTIONS[selected] || MODALITY_PROFILE_DESCRIPTIONS.balanced;
        }

        function createInput(field) {
            const wrapper = document.createElement("div");
            wrapper.className = "field";

            const label = document.createElement("label");
            label.setAttribute("for", "f_" + field.name);
            label.innerHTML = field.name + (field.required ? ' <span class="required">required</span>' : "");
            wrapper.appendChild(label);

            let input;
            if (field.kind === "textarea") {
                input = document.createElement("textarea");
            } else if (field.kind === "enum") {
                input = document.createElement("select");
                if (!field.required) {
                    const empty = document.createElement("option");
                    empty.value = "";
                    empty.textContent = "(optional)";
                    input.appendChild(empty);
                }
                (field.options || []).forEach((opt) => {
                    const o = document.createElement("option");
                    o.value = opt;
                    o.textContent = opt;
                    input.appendChild(o);
                });
            } else {
                input = document.createElement("input");
                input.type = field.kind === "int" || field.kind === "float" ? "number" : "text";
            }

            input.id = "f_" + field.name;
            input.dataset.kind = field.kind;
            input.dataset.required = String(!!field.required);
            if (field.max != null) {
                input.maxLength = Number(field.max);
            }
            if (field.min != null) {
                input.min = String(field.min);
            }
            if (field.max != null && (field.kind === "int" || field.kind === "float")) {
                input.max = String(field.max);
            }
            if (field.step != null && field.kind === "float") {
                input.step = String(field.step);
            }
            if (field.kind === "int") {
                input.step = "1";
            }
            if (field.default != null) {
                input.value = String(field.default);
            }

            input.addEventListener("input", syncPreview);
            wrapper.appendChild(input);

            if (field.kind === "list") {
                const hint = document.createElement("div");
                hint.className = "hint";
                const maxItems = field.max_items != null ? " Max " + field.max_items + " items." : "";
                hint.textContent = "Comma-separated list." + maxItems;
                wrapper.appendChild(hint);
            }

            return wrapper;
        }

        function renderFields() {
            fieldContainer.innerHTML = "";
            const selected = actionTypeSelect.value;
            const fields = ACTION_FIELDS[selected] || [];
            fields.forEach((field) => fieldContainer.appendChild(createInput(field)));
            updateActionTypeHint();
            syncPreview();
        }

        function parseFieldValue(field, rawValue) {
            if (rawValue === "") {
                return null;
            }
            if (field.kind === "int") {
                return Number.parseInt(rawValue, 10);
            }
            if (field.kind === "float") {
                return Number.parseFloat(rawValue);
            }
            if (field.kind === "list") {
                return rawValue.split(",").map((x) => x.trim()).filter((x) => x.length > 0);
            }
            return rawValue;
        }

        function buildPayload(strictValidation = false) {
            const actionType = actionTypeSelect.value;
            const payload = { action_type: actionType };
            const fields = ACTION_FIELDS[actionType] || [];
            for (const field of fields) {
                const input = $("f_" + field.name);
                if (!input) {
                    continue;
                }
                const value = parseFieldValue(field, input.value.trim());

                if (field.required && (value == null || value === "" || (Array.isArray(value) && value.length === 0))) {
                    if (strictValidation) {
                        throw new Error("Missing required field: " + field.name);
                    }
                    continue;
                }
                if (value == null) {
                    continue;
                }

                if ((field.kind === "int" || field.kind === "float") && Number.isNaN(value)) {
                    if (strictValidation) {
                        throw new Error("Invalid number for field: " + field.name);
                    }
                    continue;
                }

                if (field.kind === "list" && field.max_items != null && Array.isArray(value) && value.length > Number(field.max_items)) {
                    if (strictValidation) {
                        throw new Error("Too many items for field: " + field.name);
                    }
                    payload[field.name] = value.slice(0, Number(field.max_items));
                    continue;
                }

                payload[field.name] = value;
            }
            return payload;
        }

        function syncPreview() {
            try {
                payloadPreview.textContent = JSON.stringify(buildPayload(false), null, 2);
            } catch (err) {
                payloadPreview.textContent = String(err);
            }
        }

        async function parseResponse(res) {
            const text = await res.text();
            if (!text) {
                return { data: null, raw: "" };
            }
            try {
                return { data: JSON.parse(text), raw: text };
            } catch {
                return { data: null, raw: text };
            }
        }

        function extractStepCount(data) {
            if (data && typeof data.step_count === "number") {
                return data.step_count;
            }
            const metadata = data && data.observation && data.observation.metadata ? data.observation.metadata : {};
            if (typeof metadata.step === "number") {
                return metadata.step;
            }
            return 0;
        }

        function extractEpisodeId(data) {
            if (data && typeof data.episode_id === "string" && data.episode_id) {
                return data.episode_id;
            }
            const metadata = data && data.observation && data.observation.metadata ? data.observation.metadata : {};
            if (typeof metadata.episode_id === "string" && metadata.episode_id) {
                return metadata.episode_id;
            }
            return null;
        }

        async function doReset() {
            const seedRaw = $("seed").value.trim();
            const difficulty = $("difficulty").value;
            const modalityProfile = $("modalityProfile").value;
            const body = {};
            if (seedRaw !== "") {
                body.seed = Number.parseInt(seedRaw, 10);
            }
            if (difficulty) {
                body.difficulty = difficulty;
            }
            if (modalityProfile) {
                body.modality_profile = modalityProfile;
            }

            setStatus("POST /web/reset ...");
            const res = await fetch("/web/reset", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            const parsed = await parseResponse(res);
            if (!res.ok || !parsed.data) {
                setStatus((parsed.data && parsed.data.detail) || parsed.raw || res.statusText, true);
                return;
            }

            const data = parsed.data;
            const metadata = data.observation && data.observation.metadata ? data.observation.metadata : {};
            state.active = true;
            state.stepCount = extractStepCount(data);
            state.episodeId = extractEpisodeId(data) || state.episodeId;
            state.phase = typeof metadata.phase === "string" ? metadata.phase : "engagement";
            state.modalityProfile = typeof metadata.modality_profile === "string" ? metadata.modality_profile : "balanced";
            $("modalityProfile").value = state.modalityProfile;
            state.lastActions = [];
            state.taskTargets = Array.isArray(metadata.task && metadata.task.target_actions)
                ? metadata.task.target_actions
                : [];
            state.taskDoneActions = Array.isArray(
                metadata.task && metadata.task.reward_config && metadata.task.reward_config.done_on_actions
            )
                ? metadata.task.reward_config.done_on_actions
                : [];
            state.highRisk = false;
            state.safetyAddressed = false;
            if (data.done) {
                state.active = false;
            }
            updateSnapshot("reward=" + String(data.reward) + " | modality=" + state.modalityProfile);
            const task = metadata.task || {};
            state.currentTask = task;
            state.lastResponse = data;
            renderPanels();
            logTimeline("RESET", data);
            const preloaded = preloadRecommendedSample("after reset");
            updateModalityHint();
            setStatus(preloaded ? "Reset complete. Preloaded sample: " + preloaded : "Reset complete.");
        }

        async function doState() {
            setStatus("GET /web/state ...");
            const res = await fetch("/web/state");
            const parsed = await parseResponse(res);
            if (!res.ok || !parsed.data) {
                setStatus((parsed.data && parsed.data.detail) || parsed.raw || res.statusText, true);
                return;
            }

            const data = parsed.data;
            const metadata = data.observation && data.observation.metadata ? data.observation.metadata : {};
            state.active = true;
            const serverStepCount = extractStepCount(data);
            if (serverStepCount > 0) {
                state.stepCount = serverStepCount;
            }
            state.episodeId = extractEpisodeId(data) || state.episodeId;
            if (data.done) {
                state.active = false;
            }
            state.taskTargets = Array.isArray(metadata.task && metadata.task.target_actions)
                ? metadata.task.target_actions
                : state.taskTargets;
            state.taskDoneActions = Array.isArray(
                metadata.task && metadata.task.reward_config && metadata.task.reward_config.done_on_actions
            )
                ? metadata.task.reward_config.done_on_actions
                : state.taskDoneActions;
            state.highRisk = !!metadata.high_risk;
            state.safetyAddressed = !!metadata.safety_addressed;
            updateSnapshot("phase=" + String(metadata.phase || "-"));
            if (typeof metadata.modality_profile === "string") {
                state.modalityProfile = metadata.modality_profile;
                $("modalityProfile").value = state.modalityProfile;
                updateModalityHint();
            }
            state.currentTask = metadata.task || state.currentTask;
            state.lastResponse = data;
            renderPanels();
            logTimeline("STATE", data);
            setStatus("State fetched.");
        }

        async function doStep() {
            if (!state.active) {
                setStatus("Reset an episode before stepping.", true);
                return;
            }

            let payload;
            try {
                payload = buildPayload(true);
            } catch (err) {
                setStatus(String(err), true);
                return;
            }

            setStatus("POST /web/step ...");
            const res = await fetch("/web/step", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action: payload }),
            });

            const parsed = await parseResponse(res);
            if (!res.ok || !parsed.data) {
                setStatus((parsed.data && parsed.data.detail) || parsed.raw || res.statusText, true);
                return;
            }

            const data = parsed.data;
            const metadata = data.observation && data.observation.metadata ? data.observation.metadata : {};
            state.stepCount = metadata.step || (state.stepCount + 1);
            state.phase = typeof metadata.phase === "string" ? metadata.phase : state.phase;
            state.modalityProfile = typeof metadata.modality_profile === "string" ? metadata.modality_profile : state.modalityProfile;
            $("modalityProfile").value = state.modalityProfile;
            const latestAction = (typeof metadata.last_action_type === "string" && metadata.last_action_type)
                ? metadata.last_action_type
                : payload.action_type;
            state.lastActions.push(latestAction);
            if (state.lastActions.length > 8) {
                state.lastActions = state.lastActions.slice(-8);
            }
            state.taskTargets = Array.isArray(metadata.task && metadata.task.target_actions)
                ? metadata.task.target_actions
                : state.taskTargets;
            state.taskDoneActions = Array.isArray(
                metadata.task && metadata.task.reward_config && metadata.task.reward_config.done_on_actions
            )
                ? metadata.task.reward_config.done_on_actions
                : state.taskDoneActions;
            state.highRisk = !!metadata.high_risk;
            state.safetyAddressed = !!metadata.safety_addressed;
            state.active = !data.done;
            updateSnapshot("phase=" + String(metadata.phase || "-") + " | modality=" + state.modalityProfile);
            state.currentTask = metadata.task || state.currentTask;
            state.lastResponse = data;
            renderPanels();
            logTimeline("STEP " + payload.action_type, data);
            const preloaded = preloadRecommendedSample("after step");
            updateModalityHint();
            setStatus(preloaded ? "Step complete. Preloaded sample: " + preloaded : "Step complete.");
        }

        function loadSample() {
            loadSampleForAction(actionTypeSelect.value, { showStatus: true });
        }

        actionTypes.forEach((name) => {
            const opt = document.createElement("option");
            opt.value = name;
            opt.textContent = name;
            actionTypeSelect.appendChild(opt);
        });

        actionTypeSelect.addEventListener("change", renderFields);
        $("difficulty").addEventListener("change", updateTaskTypeHint);
        $("modalityProfile").addEventListener("change", updateModalityHint);
        verboseToggle.addEventListener("change", () => {
            state.verboseMode = !!verboseToggle.checked;
            renderPanels();
        });
        $("btnReset").addEventListener("click", () => doReset().catch((err) => setStatus(String(err), true)));
        $("btnState").addEventListener("click", () => doState().catch((err) => setStatus(String(err), true)));
        $("btnStep").addEventListener("click", () => doStep().catch((err) => setStatus(String(err), true)));
        $("btnTemplate").addEventListener("click", loadSample);

        verboseToggle.checked = false;
        state.verboseMode = false;
        renderFields();
        updateTaskTypeHint();
        updateModalityHint();
        renderPanels();
        updateSnapshot();
        setStatus("UI ready. Reset to start a demo episode.");
    })();
    </script>
</body>
</html>
"""

# Generate JavaScript code with the action fields embedded safely
def _build_action_fields_js():
    """Generate safe JavaScript code to define ACTION_FIELDS."""
    lines = []
    lines.append("const ACTION_FIELDS = {")

    action_items = list(_WEB_ACTION_FIELDS.items())
    for idx, (action_type, fields) in enumerate(action_items):
        is_last = idx == len(action_items) - 1
        comma = "" if is_last else ","

        field_items = []
        for field in fields:
            field_parts = []
            for key, value in field.items():
                if isinstance(value, bool):
                    js_value = "true" if value else "false"
                elif isinstance(value, (int, float)):
                    js_value = str(value)
                elif isinstance(value, list):
                    js_value = json.dumps(value)
                else:
                    js_value = json.dumps(value)
                field_parts.append(f'"{key}":{js_value}')
            field_str = "{" + ",".join(field_parts) + "}"
            field_items.append(field_str)

        fields_array_str = "[" + ",".join(field_items) + "]"
        lines.append(f'  "{action_type}":{fields_array_str}{comma}')

    lines.append("};")

    # Indent everything for the script context.
    indented_lines = ["        " + line for line in lines]
    return "\n".join(indented_lines)

_ACTION_FIELDS_JS = _build_action_fields_js()
# Replace the placeholder in the template with actual JavaScript
_WEB_UI_TEMPLATE_WITH_FIELDS = _WEB_UI_TEMPLATE.replace(
    "        __ACTION_FIELDS_CODE__;",
    _ACTION_FIELDS_JS
)
_WEB_UI_HTML = _WEB_UI_TEMPLATE_WITH_FIELDS


@app.get("/web", response_class=HTMLResponse, tags=["Interface"])
def web_ui() -> HTMLResponse:
        """Browser demo UI for exercising therapist actions end-to-end."""
        return HTMLResponse(content=_WEB_UI_HTML)


@app.get("/web/health", include_in_schema=False)
def web_health_alias() -> dict[str, str]:
    """Alias for health checks when the Space is served under /web base paths."""
    return {"status": "ok"}


@app.get("/web/openapi.json", include_in_schema=False)
def web_openapi_alias() -> dict[str, Any]:
    """Alias for OpenAPI schema when the Space is served under /web base paths."""
    return app.openapi()


@app.get("/web/docs", include_in_schema=False)
def web_docs_alias() -> HTMLResponse:
    """Alias for Swagger docs when the Space is served under /web base paths."""
    return get_swagger_ui_html(
        openapi_url="/web/openapi.json",
        title="openenv_therapist_assistant - API Docs",
    )


@app.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    """Send Space visitors directly to the therapist UI."""
    return RedirectResponse(url="/web", status_code=307)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m openenv_therapist_assistant.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn openenv_therapist_assistant.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
