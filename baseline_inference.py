from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict
from statistics import mean

from openai import APIStatusError, OpenAI

from env.resource import ACTIONS
from env.smart_env import SmartGrowEnv
from tasks.tasks import EpisodeOutcome, get_task, grade_task, list_tasks

HF_OPENAI_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"
ACTION_PATTERN = re.compile(r"\b([0-7])\b")


def _client() -> OpenAI:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for baseline inference.")
    return OpenAI(base_url=HF_OPENAI_BASE_URL, api_key=token)


def _action_catalog() -> str:
    lines = []
    for action_id, effect in sorted(ACTIONS.items()):
        lines.append(f"{action_id}: {effect.label}")
    return "\n".join(lines)


def _state_payload(env: SmartGrowEnv, task_name: str) -> dict[str, object]:
    state = env.state_snapshot()
    return {
        "task": task_name,
        "day": state.day,
        "max_days": env.max_days,
        "soil_moisture": round(state.soil_moisture, 3),
        "nutrients": round(state.nutrients, 3),
        "temperature": round(state.temperature, 3),
        "humidity": round(state.humidity, 3),
        "light": round(state.light, 3),
        "growth": round(state.growth, 3),
        "health": round(state.health, 3),
        "water_tank": round(state.water_tank, 3),
        "nutrient_tank": round(state.nutrient_tank, 3),
        "energy_reserve": round(state.energy_reserve, 3),
        "last_action": env.last_action,
    }


def _system_prompt() -> str:
    return (
        "You are choosing one control action for a deterministic plant simulator.\n"
        "Return JSON only with keys action_id and rationale.\n"
        "Pick exactly one integer action_id from the catalog.\n"
        "Prioritize keeping health above 0.65 while increasing growth efficiently.\n"
        "Avoid risky over-correction when water, nutrients, and energy are already stable.\n"
        "Action catalog:\n"
        f"{_action_catalog()}"
    )


def _user_prompt(env: SmartGrowEnv, task_name: str) -> str:
    payload = _state_payload(env, task_name)
    return (
        "Choose the next action for this state.\n"
        f"{json.dumps(payload, sort_keys=True)}"
    )


def _extract_action_id(response_text: str) -> int:
    try:
        payload = json.loads(response_text)
        action_id = int(payload["action_id"])
        if action_id in ACTIONS:
            return action_id
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        pass

    match = ACTION_PATTERN.search(response_text)
    if not match:
        raise ValueError(f"Unable to parse action_id from model response: {response_text!r}")
    return int(match.group(1))


def _model_response_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()

    choices = getattr(response, "choices", None) or []
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                else:
                    item_text = getattr(item, "text", None)
                    if item_text:
                        text_parts.append(item_text)
            return "".join(text_parts).strip()

    raise ValueError("The OpenAI-compatible response did not include readable text.")


def choose_action(client: OpenAI, model: str, env: SmartGrowEnv, task_name: str) -> int:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        top_p=1,
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_prompt(env, task_name)},
        ],
        response_format={"type": "json_object"},
    )
    return _extract_action_id(_model_response_text(response))


def run_task(task_name: str, model: str, seed: int | None = None) -> tuple[EpisodeOutcome, float]:
    task = get_task(task_name)
    run_seed = task.seed if seed is None else seed
    env = SmartGrowEnv(scenario_name=task.scenario_name, max_days=task.max_days, seed=run_seed)
    client = _client()
    env.reset(seed=run_seed)

    growth_values: list[float] = []
    health_values: list[float] = []
    water_values: list[float] = []
    nutrient_values: list[float] = []
    total_reward = 0.0

    while True:
        action_id = choose_action(client, model, env, task.name)
        step = env.step(action_id)
        total_reward += step.reward
        growth_values.append(env.state.growth)
        health_values.append(env.state.health)
        water_values.append(env.state.soil_moisture)
        nutrient_values.append(env.state.nutrients)
        if step.terminated or step.truncated:
            break

    outcome = EpisodeOutcome(
        total_reward=round(total_reward, 4),
        final_growth=round(env.state.growth, 4),
        final_health=round(env.state.health, 4),
        days_survived=env.state.day,
        average_growth=round(mean(growth_values), 4),
        average_health=round(mean(health_values), 4),
        average_water=round(mean(water_values), 4),
        average_nutrients=round(mean(nutrient_values), 4),
        completed=env.state.growth >= 1.0 and env.state.health >= 0.65,
        failed=env.state.health <= 0.15,
    )
    return outcome, grade_task(task.name, outcome)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reproducible OpenAI-compatible baseline benchmark across SmartGrow tasks.")
    parser.add_argument("--task", choices=list_tasks(), help="Run a single task instead of the full benchmark.", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override the default fixed task seed.")
    parser.add_argument("--model", default=os.getenv("HF_MODEL", DEFAULT_MODEL), help="Hugging Face model id exposed via the OpenAI-compatible router.")
    parser.add_argument("--show-state", action="store_true", help="Print the final state payload for each evaluated task.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    task_names = [args.task] if args.task else list_tasks()
    aggregate_scores: list[float] = []

    for task_name in task_names:
        task = get_task(task_name)
        try:
            outcome, score = run_task(task_name, model=args.model, seed=args.seed)
        except APIStatusError as exc:
            details = exc.body if isinstance(exc.body, dict) else {"error": str(exc)}
            print(
                f"benchmark_stopped task={task.name} model={args.model} "
                f"status_code={exc.status_code} error={json.dumps(details, sort_keys=True)}"
            )
            break
        aggregate_scores.append(score)
        print(
            f"{task.name} difficulty={task.difficulty} scenario={task.scenario_name} "
            f"seed={task.seed if args.seed is None else args.seed} model={args.model} "
            f"score={score:.4f} growth={outcome.final_growth:.4f} health={outcome.final_health:.4f} "
            f"days={outcome.days_survived} reward={outcome.total_reward:.4f}"
        )
        if args.show_state:
            print(json.dumps(asdict(outcome), sort_keys=True))

    if aggregate_scores:
        print(f"aggregate_score={mean(aggregate_scores):.4f}")


if __name__ == "__main__":
    main()
