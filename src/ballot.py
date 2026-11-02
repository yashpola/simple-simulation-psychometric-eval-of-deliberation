"""Secret ballot voting runner and defensive response parser.

Adheres to:
- Guideline 1: Strict typing using TypedDict schemas.
- Guideline 3: Single-responsibility functions.
- Guideline 4: Prompt template loaded from config with fallback to "".
- Guideline 5: Defensive regex parsing with try/except wrappers.
"""

import random
import re
from typing import Dict, Tuple, List, Optional, Any
from src.schemas import (
    PolicyItem,
    BallotVote,
    DyadBallotRecord,
    MasterConfig
)
from src.personas import resolve_persona_system_prompt
from src.llm import BaseLLMClient


def parse_ballot_choice(raw_text: str, valid_options: List[str] = ["A", "B", "C", "D"]) -> Optional[str]:
    """Strictly extract selected option letter from LLM response.

    Only accepts strict single letters (e.g. 'A', '(A)', '[A]', '**A**', 'A.')
    or explicit anchored label prefixes (e.g. 'Option A', 'Choice: B', 'Answer: C', 'Selection: D', 'Vote: A').
    Discards verbose explanations, reasoning scratchpads, and unanchored text.
    """
    try:
        valid_set = set(opt.upper() for opt in valid_options)
        cleaned = raw_text.strip()

        # 1. Exact match on single letter (optionally wrapped in parens, brackets, markdown bold, or period)
        m_exact = re.fullmatch(r"(?:\*{1,2}|\[|\()?\s*([A-D])\s*(?:\*{1,2}|\]|\)|\.)?", cleaned, re.IGNORECASE)
        if m_exact and m_exact.group(1).upper() in valid_set:
            return m_exact.group(1).upper()

        # 2. Strict prefix match: label + letter, anchored to full line/string
        m_prefix = re.fullmatch(
            r"^(?:option|choice|answer|selection|vote|my choice is|my final vote is|i choose option|i choose|i vote for)\s*[:\-\.]?\s*(?:\*{1,2}|\[|\()?\s*([A-D])\s*(?:\*{1,2}|\]|\)|\.)?$",
            cleaned,
            re.IGNORECASE,
        )
        if m_prefix and m_prefix.group(1).upper() in valid_set:
            return m_prefix.group(1).upper()
    except Exception:
        pass

    return None


def prepare_randomized_ballot(
    item: PolicyItem,
    rng: Optional[random.Random] = None
) -> Tuple[Dict[str, str], Dict[str, str], str]:
    """Randomize the display order of ballot options to avoid position bias.

    Returns:
    - display_to_canonical: dict mapping displayed letter (e.g. 'A') -> canonical key (e.g. 'C')
    - canonical_to_display: dict mapping canonical key -> displayed letter
    - formatted_prompt: formatted text of options for display
    """
    if rng is None:
        rng = random.Random()

    ballot_info = item.get("ballot", {})
    options_dict = ballot_info.get("options", {})
    canonical_keys = list(options_dict.keys())
    shuffled_keys = list(canonical_keys)
    rng.shuffle(shuffled_keys)

    display_letters = ["A", "B", "C", "D"][:len(shuffled_keys)]
    display_to_canonical: Dict[str, str] = {}
    canonical_to_display: Dict[str, str] = {}
    options_text_lines: List[str] = []

    for disp_letter, canon_key in zip(display_letters, shuffled_keys):
        display_to_canonical[disp_letter] = canon_key
        canonical_to_display[canon_key] = disp_letter
        option_desc = options_dict.get(canon_key, "")
        clean_desc = re.sub(r"\s*\((?:Key|Distractor)[^)]*\)\s*$", "", option_desc, flags=re.IGNORECASE).strip()
        options_text_lines.append(f"{disp_letter}. {clean_desc}")

    formatted_options = "\n".join(options_text_lines)
    return display_to_canonical, canonical_to_display, formatted_options


def administer_agent_ballot(
    client: BaseLLMClient,
    config: MasterConfig,
    persona_id: str,
    item: PolicyItem,
    dyad_id: str,
    trial_index: int,
    transcript_summary: str,
    rng: Optional[random.Random] = None,
) -> BallotVote:
    """Administer secret ballot to a single agent and record choice."""
    display_to_canon, _, formatted_options = prepare_randomized_ballot(item, rng)

    ballot_prompt_template = config.get("user_prompts", {}).get("secret_ballot", "")
    ballot_info = item.get("ballot", {})
    correct_key = ballot_info.get("correct_key", "A")

    ballot_prompt = ballot_prompt_template.format(
        title=item.get("title", ""),
        topic_prompt=item.get("topic_prompt", ""),
        transcript_summary=transcript_summary,
        ballot_prompt=ballot_info.get("prompt", ""),
        formatted_options=formatted_options
    )

    model_cfg = config.get("model_config", {})
    model_seed = model_cfg.get("seed", None)
    reasoning_effort = model_cfg.get("reasoning_effort")
    enable_reasoning = model_cfg.get("enable_reasoning")
    if enable_reasoning is None:
        enable_reasoning = model_cfg.get("reasoning_enabled")
    ballot_max_tokens = (
        model_cfg.get("ballot_max_tokens")
        or model_cfg.get("ballot_max_new_tokens")
        or 512
    )
    system_prompt = resolve_persona_system_prompt(persona_id, config)
    gen_kwargs: dict[str, Any] = {
        "prompt": ballot_prompt,
        "system_prompt": system_prompt,
        "temperature": 0.3,
        "max_tokens": ballot_max_tokens,
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

    raw_response = client.generate(**gen_kwargs)

    raw_clean = raw_response.strip()
    parsed_display_letter = parse_ballot_choice(raw_clean, list(display_to_canon.keys()))

    if not raw_clean:
        print(
            f"⚠️  [BALLOT WARNING] Agent '{persona_id}' (dyad {dyad_id}, trial {trial_index}) on '{item.get('item_id')}' "
            f"produced an EMPTY raw response. This ballot vote is discarded."
        )
        selected_canonical_key: Optional[str] = None
        is_correct = False
        is_valid = False
    elif not parsed_display_letter:
        print(
            f"⚠️  [BALLOT WARNING] Agent '{persona_id}' (dyad {dyad_id}, trial {trial_index}) on '{item.get('item_id')}' "
            f"produced unparseable ballot response: {repr(raw_clean[:60])}. This ballot vote is discarded."
        )
        selected_canonical_key = None
        is_correct = False
        is_valid = False
    else:
        selected_canonical_key = display_to_canon.get(parsed_display_letter)
        is_correct = bool(selected_canonical_key == correct_key)
        is_valid = True

    vote: BallotVote = {
        "item_id": item["item_id"],
        "dyad_id": dyad_id,
        "trial_index": trial_index,
        "persona_id": persona_id,
        "raw_response": raw_response,
        "displayed_letter": parsed_display_letter,
        "selected_key": selected_canonical_key,
        "is_correct": is_correct,
        "is_valid": is_valid,
        "display_mapping": display_to_canon
    }
    return vote


def administer_dyad_ballot(
    client: BaseLLMClient,
    config: MasterConfig,
    agents: List[str],
    item: PolicyItem,
    dyad_id: str,
    trial_index: int,
    transcript_summary: str,
    mcq_option_order_seed: Optional[int] = None,
    seed: Optional[int] = None,
) -> DyadBallotRecord:
    """Run independent secret ballots for both agents in a dyad and evaluate consensus.

    The mcq_option_order_seed (or legacy seed) determines the pseudo-random shuffling
    of MCQ ballot option letters (A/B/C/D) to mitigate LLM position bias.
    """
    effective_seed = mcq_option_order_seed if mcq_option_order_seed is not None else seed
    votes: Dict[str, BallotVote] = {}
    for idx, persona_id in enumerate(agents):
        agent_seed = (effective_seed + idx * 1000 + trial_index) if effective_seed is not None else None
        rng = random.Random(agent_seed)
        vote = administer_agent_ballot(
            client=client,
            config=config,
            persona_id=persona_id,
            item=item,
            dyad_id=dyad_id,
            trial_index=trial_index,
            transcript_summary=transcript_summary,
            rng=rng
        )
        votes[persona_id] = vote

    # Consensus reached if ALL agents in dyad had valid votes and chose the integrative compromise (target key)
    consensus = bool(all(v.get("is_valid", False) and v.get("is_correct", False) for v in votes.values()))

    ballot_record: DyadBallotRecord = {
        "item_id": item["item_id"],
        "dyad_id": dyad_id,
        "trial_index": trial_index,
        "votes": votes,
        "consensus_reached": consensus
    }
    return ballot_record
