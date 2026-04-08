---
title: Openenv Therapist Assistant Environment Server
emoji: 🖱️
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
    - therapist
    - assistant
    - actions
short_description: Therapist-assistant env with configurable action fields
---

# Openenv Therapist Assistant Environment

A therapist-assistant training environment with structured, discriminated actions keyed by `action_type`.
The current environment behavior converts each action into an echoed summary string so you can quickly validate action payloads and server/client wiring.

## Action Fields

Use these fields to configure actions from the Space frontend:

| action_type | Required fields | Optional fields |
|---|---|---|
| `check_in` | `prompt` | `mood_scale_1_10`, `energy_scale_1_10`, `sleep_quality_1_10` |
| `ask_open_question` | `question_text`, `focus_area` | - |
| `reflect_content` | `reflection_text` | `source_span` |
| `reflect_emotion` | `emotion_labels`, `reflection_text`, `confidence_0_1` | - |
| `validate_experience` | `validation_text` | `context_reference` |
| `summarize_session` | `summary_text` | `key_points`, `open_items` |
| `clarify_or_probe` | `question_text`, `reason_for_probe` | - |
| `risk_screen` | `risk_domain`, `screening_question`, `urgency_level` | - |
| `safety_plan_step` | `step_type`, `step_text` | - |
| `psychoeducation` | `topic`, `content_text` | `reading_level` |
| `coping_skill_coach` | `skill_name`, `instructions`, `duration_minutes` | - |
| `goal_set` | `goal_text`, `time_horizon`, `success_criteria` | - |
| `home_practice_assign` | `practice_name`, `instructions`, `frequency`, `tracking_method` | - |
| `resource_recommend` | `resource_type`, `resource_details`, `reason` | - |
| `handoff_or_escalate` | `escalation_reason`, `target`, `handoff_message` | - |

The Space frontend can use these fields to build action forms, validate payloads, and guide manual testing.

## Quick Start

The simplest way to use the Openenv Therapist Assistant environment is through the `OpenenvTherapistAssistantEnv` class:

```python
import asyncio

from openenv_therapist_assistant import (
    AskOpenQuestionAction,
    CheckInAction,
    OpenenvTherapistAssistantEnv,
    ReflectEmotionAction,
)

async def run_episode() -> None:
    # Create environment from Docker image
    openenv_therapist_assistantenv = await OpenenvTherapistAssistantEnv.from_docker_image(
        "openenv_therapist_assistant-env:latest"
    )

    try:
        # Reset
        result = await openenv_therapist_assistantenv.reset()
        print(f"Reset: {result.observation.echoed_message}")

        # Run a short therapist-assistant interaction flow
        actions = [
            CheckInAction(prompt="How have you been feeling since our last session?"),
            AskOpenQuestionAction(
                question_text="What has felt the hardest this week?",
                focus_area="emotion",
            ),
            ReflectEmotionAction(
                emotion_labels=["overwhelmed", "sad"],
                reflection_text="It sounds like you are carrying a lot and feeling emotionally drained.",
                confidence_0_1=0.78,
            ),
        ]

        for action in actions:
            result = await openenv_therapist_assistantenv.step(action)
            print(f"Action: {action.action_type}")
            print(f"  -> Echoed: '{result.observation.echoed_message}'")
            print(f"  -> Length: {result.observation.message_length}")
            print(f"  -> Reward: {result.reward}")
    finally:
        # Always clean up
        await openenv_therapist_assistantenv.close()


asyncio.run(run_episode())
```

That's it! The `OpenenvTherapistAssistantEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `await close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From the environment directory
docker build -t openenv_therapist_assistant-env:latest -f Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
`OpenenvTherapistAssistantAction` is a discriminated union keyed on `action_type`.

Supported action types:
- `check_in`
- `ask_open_question`
- `reflect_content`
- `reflect_emotion`
- `validate_experience`
- `summarize_session`
- `clarify_or_probe`
- `risk_screen`
- `safety_plan_step`
- `psychoeducation`
- `coping_skill_coach`
- `goal_set`
- `home_practice_assign`
- `resource_recommend`
- `handoff_or_escalate`

### Task Bank

The environment now includes a small therapist task bank for training and evaluation.
Tasks are grouped by difficulty:
- `easy` - rapport building, problem listing, basic goal setting
- `moderate` - clarifying vague problems, prioritizing concerns, coping planning
- `hard` - risk-sensitive work, resistance handling, complex formulation

Each task includes a `client_message`, an `objective`, `success_criteria`, and recommended action types.

Example action payloads:

```python
CheckInAction(
    prompt="How are you feeling today?",
    mood_scale_1_10=4,
)

RiskScreenAction(
    risk_domain="self_harm",
    screening_question="Have you had thoughts of hurting yourself recently?",
    urgency_level="high",
)
```

### Observation
**OpenenvTherapistAssistantObservation**: Contains the echoed action summary and metadata
- `echoed_message` (str) - Human-readable summary generated from the action
- `message_length` (int) - Length of the echoed summary
- `reward` (float) - Reward based on message length (length × 0.1)
- `done` (bool) - Always False for echo environment
- `metadata` (dict) - Additional info like step count

### Reward
The reward is currently calculated as: `message_length × 0.1`
- `{"action_type": "check_in", "prompt": "Hi"}` → echoed summary `[check_in] Hi` → reward: 1.3
- Longer prompts/actions generate larger rewards because the echoed summary is longer.

## Advanced Usage

### Connecting to an Existing Server

If you already have a Openenv Therapist Assistant environment server running, you can connect directly:

```python
from openenv_therapist_assistant import (
    AskOpenQuestionAction,
    OpenenvTherapistAssistantEnv,
)

# Connect to existing server
openenv_therapist_assistantenv = OpenenvTherapistAssistantEnv(base_url="<ENV_HTTP_URL_HERE>")

async def run() -> None:
    async with openenv_therapist_assistantenv:
        # Use as normal
        result = await openenv_therapist_assistantenv.reset()
        result = await openenv_therapist_assistantenv.step(
            AskOpenQuestionAction(
                question_text="What would feel most supportive right now?",
                focus_area="coping",
            )
        )


asyncio.run(run())
```

Note: When connecting to an existing server, `await openenv_therapist_assistantenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from openenv_therapist_assistant import (
    OpenenvTherapistAssistantEnv,
    ReflectContentAction,
)

import asyncio


# Connect with async context manager (auto-connects and closes)
async def run() -> None:
    async with OpenenvTherapistAssistantEnv(base_url="http://localhost:8000") as env:
        result = await env.reset()
        print(f"Reset: {result.observation.echoed_message}")
        # Multiple steps with low latency
        reflections = [
            "You felt dismissed in that conversation.",
            "A part of you wants boundaries, but another part fears conflict.",
            "You are noticing patterns, which is meaningful progress.",
        ]
        for text in reflections:
            result = await env.step(ReflectContentAction(reflection_text=text))
            print(f"Echoed: {result.observation.echoed_message}")


asyncio.run(run())
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    OpenenvTherapistAssistantEnvironment,  # Pass class, not instance
    OpenenvTherapistAssistantAction,
    OpenenvTherapistAssistantObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
import asyncio

from openenv_therapist_assistant import OpenenvTherapistAssistantEnv

async def run_episode(client_id: int):
    async with OpenenvTherapistAssistantEnv(base_url="http://localhost:8000") as env:
        result = await env.reset()
        for i in range(10):
            result = await env.step(
                {
                    "action_type": "check_in",
                    "prompt": f"Client {client_id}, step {i}: How are you feeling now?",
                }
            )
        return client_id, result.observation.message_length

async def run_concurrent() -> None:
    # Run 4 episodes concurrently
    results = await asyncio.gather(*(run_episode(i) for i in range(4)))
    print(results)


asyncio.run(run_concurrent())
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/openenv_therapist_assistant_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
openenv_therapist_assistant/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # OpenenvTherapistAssistantEnv client
├── models.py              # Action and Observation models
├── tasks/                 # Therapist task bank and sample tasks
└── server/
    ├── __init__.py        # Server module exports
    ├── openenv_therapist_assistant_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
