"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import asyncio
import json
import os
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Any, List, Optional

from openai import OpenAI

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openenv_therapist_assistant.client import OpenenvTherapistAssistantEnv
from openenv_therapist_assistant.models import OpenenvTherapistAssistantActionAdapter

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME") or "openenv_therapist_assistant-env:latest"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK_NAME") or "therapist_assistant_episode"
BENCHMARK = os.getenv("BENCHMARK") or "openenv_therapist_assistant"
MAX_STEPS = 12
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]
INFERENCE_DEBUG = os.getenv("INFERENCE_DEBUG", "1").strip().lower() in {"1", "true", "yes", "on"}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a therapist-assistant policy acting in a simulated environment.
Return exactly one JSON object per turn with a valid therapist action.
Do not include markdown.

Valid action schemas:
- check_in: {"action_type":"check_in","prompt":"..."}
- ask_open_question: {"action_type":"ask_open_question","question_text":"...","focus_area":"emotion|event|relationship|coping"}
- reflect_content: {"action_type":"reflect_content","reflection_text":"..."}
- reflect_emotion: {"action_type":"reflect_emotion","emotion_labels":["..."],"reflection_text":"...","confidence_0_1":0.0-1.0}
- validate_experience: {"action_type":"validate_experience","validation_text":"..."}
- summarize_session: {"action_type":"summarize_session","summary_text":"...","key_points":["..."],"open_items":["..."]}
- clarify_or_probe: {"action_type":"clarify_or_probe","question_text":"...","reason_for_probe":"..."}
- risk_screen: {"action_type":"risk_screen","risk_domain":"self_harm|harm_to_others|abuse|substance","screening_question":"...","urgency_level":"low|medium|high|immediate"}
- safety_plan_step: {"action_type":"safety_plan_step","step_type":"coping_strategy|support_contact|means_restriction|crisis_resource","step_text":"..."}
- psychoeducation: {"action_type":"psychoeducation","topic":"...","content_text":"...","reading_level":"basic|intermediate|advanced"}
- coping_skill_coach: {"action_type":"coping_skill_coach","skill_name":"...","instructions":"...","duration_minutes":1-120}
- goal_set: {"action_type":"goal_set","goal_text":"...","time_horizon":"this_week|this_month|next_3_months|long_term","success_criteria":"..."}
- home_practice_assign: {"action_type":"home_practice_assign","practice_name":"...","instructions":"...","frequency":"...","tracking_method":"..."}
- resource_recommend: {"action_type":"resource_recommend","resource_type":"hotline|worksheet|community_service|app","resource_details":"...","reason":"..."}
- handoff_or_escalate: {"action_type":"handoff_or_escalate","escalation_reason":"...","target":"licensed_therapist|emergency_services|crisis_line","handoff_message":"..."}
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error.replace("\n", " ").replace("\r", " ") if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def compute_episode_score(rewards: list[float]) -> float:
    """Compute normalized score as average reward over executed steps."""
    if not rewards:
        return 0.0
    avg_reward = sum(rewards) / len(rewards)
    return min(max(avg_reward, 0.0), 1.0)


def debug_log(message: str) -> None:
    if INFERENCE_DEBUG:
        print(f"[DEBUG] {message}", file=sys.stderr, flush=True)


def _fallback_action_payload() -> dict[str, Any]:
    return {"action_type": "check_in", "prompt": "hello"}


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def get_model_action_payload(
    client: OpenAI,
    step: int,
    last_echoed: str,
    last_reward: float,
    history: List[str],
) -> tuple[dict[str, Any], str | None]:
    try:
        message = get_model_message(client, step, last_echoed, last_reward, history)
    except Exception as exc:
        return _fallback_action_payload(), f"Model request failed: {exc}"

    payload = _extract_json_payload(message)
    if payload is not None:
        return payload, None
    return _fallback_action_payload(), f"Failed to parse JSON from model response: {message!r}"


def build_user_prompt(step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Last echoed message: {last_echoed!r}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Send your next message.
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, last_echoed, last_reward, history)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    text = (completion.choices[0].message.content or "").strip()
    return text if text else "hello"


async def main() -> None:
    debug_log(
        "startup "
        f"image={IMAGE_NAME} "
        f"api_base={API_BASE_URL} "
        f"model={MODEL_NAME} "
        f"task={TASK_NAME} "
        f"benchmark={BENCHMARK} "
        f"max_steps={MAX_STEPS}"
    )
    debug_log(f"api_key_present={bool(API_KEY)}")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await OpenenvTherapistAssistantEnv.from_docker_image(IMAGE_NAME)
    debug_log("environment client created from docker image")

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    runtime_task = TASK_NAME
    start_emitted = False

    try:
        result = await env.reset() # OpenENV.reset()
        debug_log(
            "reset completed "
            f"done={result.done} "
            f"reward={result.reward} "
            f"obs_done={result.observation.done} "
            f"message_length={result.observation.message_length} "
            f"metadata_keys={sorted((result.observation.metadata or {}).keys())}"
        )
        task_metadata = (result.observation.metadata or {}).get("task", {})
        runtime_task = str(task_metadata.get("task_id") or task_metadata.get("title") or runtime_task)
        debug_log(
            "task metadata "
            f"task_id={task_metadata.get('task_id')} "
            f"title={task_metadata.get('title')}"
        )

        log_start(task=runtime_task, env=BENCHMARK, model=MODEL_NAME)
        start_emitted = True

        last_echoed = result.observation.echoed_message
        last_reward = 0.0
        debug_log(f"entering step loop max_steps={MAX_STEPS}")

        for step in range(1, MAX_STEPS + 1):
            debug_log(f"loop top step={step} pre_step_done={result.done}")
            if result.done:
                debug_log("breaking before step because result.done=true")
                break

            payload, error = get_model_action_payload(client, step, last_echoed, last_reward, history)
            action_text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
            debug_log(
                f"model payload step={step} action_type={payload.get('action_type')} "
                f"model_error={error}"
            )

            try:
                action = OpenenvTherapistAssistantActionAdapter.validate_python(payload)
            except Exception as exc:
                error = str(exc)
                debug_log(f"action validation failed step={step}: {error}")
                payload = _fallback_action_payload()
                action_text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
                action = OpenenvTherapistAssistantActionAdapter.validate_python(payload)
                debug_log(f"using fallback action step={step}")

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step
            last_echoed = obs.echoed_message
            last_reward = reward
            debug_log(
                "step result "
                f"step={step} done={done} reward={reward:.3f} "
                f"message_length={obs.message_length} "
                f"metadata_keys={sorted((obs.metadata or {}).keys())}"
            )

            log_step(step=step, action=action_text, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_text} -> reward {reward:+.2f}")

            if done:
                debug_log(f"breaking after step={step} because done=true")
                break

        score = compute_episode_score(rewards)
        success = score >= SUCCESS_SCORE_THRESHOLD
        debug_log(
            f"episode finished success={success} steps_taken={steps_taken} "
            f"score={score:.3f} rewards={rewards}"
        )

    except Exception as exc:
        debug_log(f"unhandled exception in main: {exc}")
        traceback.print_exc(file=sys.stderr)
        if not start_emitted:
            log_start(task=runtime_task, env=BENCHMARK, model=MODEL_NAME)
            start_emitted = True

    finally:
        try:
            await env.close()
            debug_log("environment client closed")
        except Exception as exc:
            debug_log(f"env.close() raised: {exc}")

        if not start_emitted:
            log_start(task=runtime_task, env=BENCHMARK, model=MODEL_NAME)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        debug_log("end line emitted")


if __name__ == "__main__":
    asyncio.run(main())