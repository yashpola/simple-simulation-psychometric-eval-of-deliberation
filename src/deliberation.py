"""Dyadic deliberation engine across diverse citizen groups and policy items.

Adheres to:
- Guideline 2: Functional split for LLM chat loops & prompt injections.
- Guideline 3: Single-responsibility deliberation execution.
- Guideline 4: Zero hardcoded prompt strings; all prompts loaded from MasterConfig with default "".
- Guideline 5: Defensive parsing wrapped in try/except.
"""

from typing import List, Any, Dict
from src.schemas import (
    PolicyItem,
    DeliberationTurn,
    DeliberationTranscript,
    MasterConfig
)
from src.personas import get_persona_metadata, resolve_persona_system_prompt
from src.llm import BaseLLMClient


def build_conversation_history(turns: List[DeliberationTurn]) -> str:
    """Format conversational dialogue history for prompt context."""
    if not turns:
        return ""

    history_lines: List[str] = []
    for turn in turns:
        history_lines.append(f"Turn {turn['turn_index'] + 1} - [{turn['speaker']}]:\n{turn['statement']}\n")
    return "\n".join(history_lines)


def conduct_deliberation(
    client: BaseLLMClient,
    config: MasterConfig,
    item: PolicyItem,
    dyad_id: str,
    agents: List[str],
    trial_index: int,
    turns_per_item: int = 4
) -> DeliberationTranscript:
    """Execute complete dyadic deliberation between two agents for a policy item."""
    turns: List[DeliberationTurn] = []
    user_prompts = config.get("user_prompts", {})
    start_template = user_prompts.get("deliberation_start")
    turn_template = user_prompts.get("deliberation_turn", "")
    model_cfg = config.get("model_config", {})
    temperature = model_cfg.get("temperature", 0.7)
    max_tokens = (
        model_cfg.get("deliberation_max_tokens")
        or model_cfg.get("deliberation_max_new_tokens")
        or model_cfg.get("max_tokens", 512)
    )
    model_seed = model_cfg.get("seed", None)
    reasoning_effort = model_cfg.get("reasoning_effort")
    enable_reasoning = model_cfg.get("enable_reasoning")
    if enable_reasoning is None:
        enable_reasoning = model_cfg.get("reasoning_enabled")

    for turn_idx in range(turns_per_item):
        current_speaker_id = agents[turn_idx % len(agents)]
        persona_meta = get_persona_metadata(current_speaker_id)
        conv_history = build_conversation_history(turns)

        system_prompt = resolve_persona_system_prompt(current_speaker_id, config)
        template = start_template if (turn_idx == 0 and start_template) else turn_template
        turn_prompt = template.format(
            title=item.get("title", ""),
            topic_prompt=item.get("topic_prompt", ""),
            conversation_history=conv_history
        )

        gen_kwargs: dict[str, Any] = {
            "prompt": turn_prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": model_seed,
            "reasoning_effort": reasoning_effort,
        }
        if enable_reasoning is not None:
            try:
                import inspect
                sig = inspect.signature(client.generate)
                if "enable_reasoning" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    gen_kwargs["enable_reasoning"] = enable_reasoning
            except Exception:
                gen_kwargs["enable_reasoning"] = enable_reasoning

        statement = client.generate(**gen_kwargs).strip()

        turn_record: DeliberationTurn = {
            "turn_index": turn_idx,
            "speaker": persona_meta["name"],
            "persona_id": current_speaker_id,
            "statement": statement
        }
        turns.append(turn_record)

    transcript: DeliberationTranscript = {
        "item_id": item["item_id"],
        "item_title": item["title"],
        "dyad_id": dyad_id,
        "agents": agents,
        "trial_index": trial_index,
        "turns": turns
    }
    return transcript
