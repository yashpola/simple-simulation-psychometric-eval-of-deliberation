"""Zero-deliberation baseline evaluation runner.

Administers standardized secret ballots to citizen personas WITHOUT prior deliberation.
Evaluates the counterfactual baseline probability of selecting the Integrative Compromise
(Option A) vs cognitive distractors (Options B, C, D) prior to communicative exchange.
"""

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

try:
    from dotenv import load_dotenv  # pyright: ignore[reportUnknownVariableType]
    _ = load_dotenv()
except ImportError:
    pass

from openai import OpenAI
from src.ballot import parse_ballot_choice, prepare_randomized_ballot
from src.personas import resolve_persona_system_prompt
from src.schemas import PolicyItem, MasterConfig


def load_baseline_config() -> Tuple[MasterConfig, List[PolicyItem]]:
    with open("data/config.json", "r", encoding="utf-8") as f:
        config: MasterConfig = json.load(f)

    items: List[PolicyItem] = []
    for p in config.get("items", []):
        with open(p, "r", encoding="utf-8") as f:
            items.append(json.load(f))
    return config, items


def run_single_baseline_vote(
    client: OpenAI,
    model: str,
    config: MasterConfig,
    persona_id: str,
    item: PolicyItem,
    trial_index: int,
    seed: int,
    reasoning_effort: Optional[str] = "low",
    enable_reasoning: Optional[bool] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    display_to_canon, _, formatted_options = prepare_randomized_ballot(item, rng)

    ballot_prompt_template = config.get("user_prompts", {}).get("secret_ballot", "")
    ballot_info = item.get("ballot", {})
    correct_key = ballot_info.get("correct_key", "A")

    transcript_summary = (
        "(No prior deliberation took place. This is a baseline pre-deliberation evaluation.)"
    )

    ballot_prompt = ballot_prompt_template.format(
        title=item.get("title", ""),
        topic_prompt=item.get("topic_prompt", ""),
        transcript_summary=transcript_summary,
        ballot_prompt=ballot_info.get("prompt", ""),
        formatted_options=formatted_options,
    )

    system_prompt = resolve_persona_system_prompt(persona_id, config)

    model_cfg = config.get("model_config", {})
    if enable_reasoning is None:
        enable_reasoning = model_cfg.get("enable_reasoning")
        if enable_reasoning is None:
            enable_reasoning = model_cfg.get("reasoning_enabled")

    req_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ballot_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    extra_body: Dict[str, Any] = {}
    if enable_reasoning is False:
        extra_body["reasoning"] = {"enabled": False}
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    elif enable_reasoning is True:
        extra_body["reasoning"] = {"enabled": True}
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}
        if reasoning_effort is not None and str(reasoning_effort).lower() != "none":
            req_kwargs["reasoning_effort"] = reasoning_effort
    elif reasoning_effort is not None and str(reasoning_effort).lower() != "none":
        req_kwargs["reasoning_effort"] = reasoning_effort

    if extra_body:
        req_kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**req_kwargs)

    choice = response.choices[0]
    msg = choice.message
    content = (msg.content or "").strip()
    reasoning = (getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or "")

    # Parse letter from content first, falling back to reasoning if content is empty
    parsed_letter = parse_ballot_choice(content, list(display_to_canon.keys()))
    if not parsed_letter and reasoning:
        # Fallback to search reasoning for final option choice
        parsed_letter = parse_ballot_choice(reasoning, list(display_to_canon.keys()))

    if parsed_letter:
        selected_key = display_to_canon.get(parsed_letter)
        is_correct = bool(selected_key == correct_key)
        is_valid = True
    else:
        selected_key = None
        is_correct = False
        is_valid = False

    return {
        "model": model,
        "persona_id": persona_id,
        "item_id": item["item_id"],
        "trial_index": trial_index,
        "displayed_letter": parsed_letter,
        "selected_key": selected_key,
        "is_correct": is_correct,
        "is_valid": is_valid,
        "raw_response": content if content else (reasoning[:200] if reasoning else ""),
        "display_mapping": display_to_canon,
    }


def run_baseline_experiment(
    model: str,
    n_trials: int = 5,
    max_workers: int = 6,
    reasoning_effort: Optional[str] = "low",
    enable_reasoning: Optional[bool] = None,
) -> Dict[str, Any]:
    print(f"\n==========================================")
    print(f"Running Baseline Experiment for {model}")
    print(f"==========================================")

    config, items = load_baseline_config()
    personas = ["technocrat", "ideologue", "populist", "everyman"]

    if enable_reasoning is None:
        model_cfg = config.get("model_config", {})
        enable_reasoning = model_cfg.get("enable_reasoning")
        if enable_reasoning is None:
            enable_reasoning = model_cfg.get("reasoning_enabled")

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=90.0)

    tasks: List[Dict[str, Any]] = []
    base_seed = 42

    for persona_id in personas:
        for item in items:
            for trial_idx in range(n_trials):
                seed = base_seed + trial_idx * 100 + len(persona_id) * 10
                tasks.append({
                    "persona_id": persona_id,
                    "item": item,
                    "trial_index": trial_idx,
                    "seed": seed,
                })

    results: List[Dict[str, Any]] = []
    total_tasks = len(tasks)
    print(f"Administering {total_tasks} baseline secret ballots across {len(personas)} personas and {len(items)} items...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                run_single_baseline_vote,
                client=client,
                model=model,
                config=config,
                persona_id=t["persona_id"],
                item=t["item"],
                trial_index=t["trial_index"],
                seed=t["seed"],
                reasoning_effort=reasoning_effort,
                enable_reasoning=enable_reasoning,
            ): t
            for t in tasks
        }

        completed = 0
        for future in as_completed(future_to_task):
            completed += 1
            try:
                res = future.result()
                results.append(res)
                if completed % 10 == 0 or completed == total_tasks:
                    print(f"  Progress: {completed}/{total_tasks} ballots completed ({round(completed/total_tasks*100, 1)}%)")
            except Exception as e:
                task_info = future_to_task[future]
                print(f"  [ERROR] Vote failed for {task_info['persona_id']} on {task_info['item']['item_id']}: {e}")

    # Compute Summary Statistics
    persona_stats: Dict[str, Dict[str, Any]] = {}
    for p in personas:
        p_votes = [r for r in results if r["persona_id"] == p and r["is_valid"]]
        n_p = len(p_votes)
        n_correct = sum(1 for r in p_votes if r["is_correct"])
        acc = round(n_correct / n_p, 4) if n_p > 0 else 0.0

        opt_counts: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
        for r in p_votes:
            k = r.get("selected_key")
            if k in opt_counts:
                opt_counts[k] += 1

        persona_stats[p] = {
            "total_ballots": n_p,
            "correct_ballots": n_correct,
            "success_rate": acc,
            "option_counts": opt_counts,
        }

    item_stats: Dict[str, Dict[str, Any]] = {}
    for item in items:
        item_id = item["item_id"]
        i_votes = [r for r in results if r["item_id"] == item_id and r["is_valid"]]
        n_i = len(i_votes)
        n_correct = sum(1 for r in i_votes if r["is_correct"])
        p_val = round(n_correct / n_i, 4) if n_i > 0 else 0.0

        opt_counts: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
        for r in i_votes:
            k = r.get("selected_key")
            if k in opt_counts:
                opt_counts[k] += 1

        item_stats[item_id] = {
            "title": item.get("title", ""),
            "total_ballots": n_i,
            "correct_ballots": n_correct,
            "difficulty_p": p_val,
            "option_counts": opt_counts,
        }

    total_valid = sum(1 for r in results if r["is_valid"])
    total_correct = sum(1 for r in results if r["is_correct"])
    pooled_acc = round(total_correct / total_valid, 4) if total_valid > 0 else 0.0

    summary = {
        "model": model,
        "total_ballots_administered": len(results),
        "total_valid_ballots": total_valid,
        "pooled_accuracy": pooled_acc,
        "chance_accuracy": 0.25,
        "persona_statistics": persona_stats,
        "item_statistics": item_stats,
        "raw_ballots": results,
    }

    # Save to data/baselines/
    out_dir = Path("data/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = model.replace("/", "_").replace("-", "_").lower()
    out_file = out_dir / f"baseline_{slug}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved baseline results to {out_file}")

    print("\n--- BASELINE SUMMARY ---")
    print(f"Model: {model}")
    print(f"Pooled Accuracy: {round(pooled_acc*100, 1)}% ({total_correct}/{total_valid})")
    for p, stats in persona_stats.items():
        print(f"  • {p.capitalize():12s}: {round(stats['success_rate']*100, 1)}% ({stats['correct_ballots']}/{stats['total_ballots']}) | Options: {stats['option_counts']}")
    for i_id, stats in item_stats.items():
        print(f"  • {i_id}: p = {stats['difficulty_p']} | Options: {stats['option_counts']}")

    return summary


if __name__ == "__main__":
    import sys
    models_to_run = [
        "meta-models/Muse-Spark-Open-816B-A42B",
        "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    ]
    if len(sys.argv) > 1:
        models_to_run = sys.argv[1:]
    for m in models_to_run:
        run_baseline_experiment(m, n_trials=5, max_workers=6, reasoning_effort="low")
