"""Unit tests for secret ballot parser and option display randomization."""

import random
from typing import Optional
from src.ballot import parse_ballot_choice, prepare_randomized_ballot, administer_dyad_ballot
from src.llm import BaseLLMClient
from src.schemas import PolicyItem, MasterConfig


class DummyBallotClient(BaseLLMClient):
    """Test client for ballot evaluation."""

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        seed: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        enable_reasoning: Optional[bool] = None,
        **kwargs,
    ) -> str:
        return "Option A"


def test_parse_ballot_choice_exact():
    assert parse_ballot_choice("A") == "A"
    assert parse_ballot_choice("b") == "B"
    assert parse_ballot_choice(" C ") == "C"
    assert parse_ballot_choice("D\n") == "D"


def test_parse_ballot_choice_prefixed():
    assert parse_ballot_choice("Option A") == "A"
    assert parse_ballot_choice("Choice: B") == "B"
    assert parse_ballot_choice("Answer: C.") == "C"
    assert parse_ballot_choice("Selection: D") == "D"
    assert parse_ballot_choice("Option (B)") == "B"


def test_parse_ballot_choice_markdown_and_tokens():
    assert parse_ballot_choice("**A**") == "A"
    assert parse_ballot_choice("(C)") == "C"
    assert parse_ballot_choice("[B]") == "B"
    assert parse_ballot_choice("I choose Option A.") == "A"
    assert parse_ballot_choice("My final vote is C.") == "C"


def test_parse_ballot_choice_invalid():
    assert parse_ballot_choice("E") is None
    assert parse_ballot_choice("None of the above") is None
    assert parse_ballot_choice("I cannot decide") is None


def test_prepare_randomized_ballot_mappings():
    item: PolicyItem = {
        "item_id": "TEST_01",
        "title": "Test Item",
        "topic_prompt": "Prompt context",
        "ballot": {
            "options": {
                "A": "Integrative Compromise",
                "B": "Status Quo",
                "C": "Zero Sum",
                "D": "Scapegoat"
            },
            "correct_key": "A"
        }
    }

    rng = random.Random(123)
    disp_to_canon, canon_to_disp, formatted_text = prepare_randomized_ballot(item, rng)

    # Validate bijection
    assert set(disp_to_canon.keys()) == {"A", "B", "C", "D"}
    assert set(disp_to_canon.values()) == {"A", "B", "C", "D"}
    for d, c in disp_to_canon.items():
        assert canon_to_disp[c] == d

    # Validate formatted text contains all option descriptions
    for _, desc in item["ballot"]["options"].items():
        assert desc in formatted_text


def test_prepare_randomized_ballot_strips_taxonomy_annotations():
    item: PolicyItem = {
        "item_id": "TEST_02",
        "title": "Annotated Test Item",
        "topic_prompt": "Prompt context",
        "ballot": {
            "options": {
                "A": "Policy Proposal A (Key - Integrative Compromise)",
                "B": "Policy Proposal B (Distractor 1 - Status Quo Inertia)",
                "C": "Policy Proposal C (Distractor 2 - Technocratic Zero-Sum)",
                "D": "Policy Proposal D (Distractor 3 - External Scapegoating/Deflection)",
            },
            "correct_key": "A",
        },
    }

    rng = random.Random(42)
    _, _, formatted_text = prepare_randomized_ballot(item, rng)

    # Validate all taxonomy metadata tags are cleanly stripped from prompt payload
    assert "(Key" not in formatted_text
    assert "Integrative Compromise" not in formatted_text
    assert "(Distractor" not in formatted_text
    assert "Policy Proposal A" in formatted_text
    assert "Policy Proposal B" in formatted_text
    assert "Policy Proposal C" in formatted_text
    assert "Policy Proposal D" in formatted_text


def test_administer_dyad_ballot_seed_parameter_compatibility():
    item: PolicyItem = {
        "item_id": "TEST_01",
        "title": "Test Item",
        "topic_prompt": "Prompt context",
        "ballot": {
            "options": {
                "A": "Integrative Compromise",
                "B": "Status Quo",
                "C": "Zero Sum",
                "D": "Scapegoat"
            },
            "correct_key": "A"
        }
    }
    cfg: MasterConfig = {
        "model_config": {"agent_model": "mock"},
        "user_prompts": {
            "secret_ballot": "{formatted_options}"
        }
    }
    client = DummyBallotClient()

    # Calling with mcq_option_order_seed vs legacy seed produces identical mappings
    res_new = administer_dyad_ballot(
        client=client,
        config=cfg,
        agents=["technocrat", "populist"],
        item=item,
        dyad_id="D1",
        trial_index=0,
        transcript_summary="summary",
        mcq_option_order_seed=42
    )

    res_legacy = administer_dyad_ballot(
        client=client,
        config=cfg,
        agents=["technocrat", "populist"],
        item=item,
        dyad_id="D1",
        trial_index=0,
        transcript_summary="summary",
        seed=42
    )

    for persona in ["technocrat", "populist"]:
        assert res_new["votes"][persona]["display_mapping"] == res_legacy["votes"][persona]["display_mapping"]
        assert res_new["votes"][persona]["displayed_letter"] == res_legacy["votes"][persona]["displayed_letter"]
        assert res_new["votes"][persona]["selected_key"] == res_legacy["votes"][persona]["selected_key"]

