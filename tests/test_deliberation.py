"""Unit tests for dyadic deliberation."""

from typing import Optional
from src.deliberation import (
    build_conversation_history,
    conduct_deliberation
)
from src.schemas import DeliberationTurn, PolicyItem
from src.main import load_master_configuration
from src.llm import BaseLLMClient


class MockLLMClient(BaseLLMClient):
    """Simple test double for deliberation unit tests."""

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
        return "I propose we combine necessary structural reform with direct targeted protections."


def test_build_conversation_history():
    turns: list[DeliberationTurn] = [
        {
            "turn_index": 0,
            "speaker": "Technocrat",
            "persona_id": "technocrat",
            "statement": "Let us look at the budget numbers."
        },
        {
            "turn_index": 1,
            "speaker": "Populist",
            "persona_id": "populist",
            "statement": "What about everyday people?"
        }
    ]

    history = build_conversation_history(turns)
    assert "Let us look at the budget numbers." in history
    assert "What about everyday people?" in history


def test_conduct_deliberation_turn_structure():
    config = load_master_configuration("data/config.json")
    item: PolicyItem = {
        "item_id": "ITEM_01",
        "title": "Test Policy",
        "topic_prompt": "Deliberate on policy",
        "ballot": {"options": {"A": "Key", "B": "Trap"}, "correct_key": "A"}
    }

    client = MockLLMClient()
    transcript = conduct_deliberation(
        client=client,
        config=config,
        item=item,
        dyad_id="D1",
        agents=["technocrat", "ideologue"],
        trial_index=0,
        turns_per_item=4
    )

    assert len(transcript["turns"]) == 4
    assert transcript["turns"][0]["persona_id"] == "technocrat"
    assert transcript["turns"][1]["persona_id"] == "ideologue"
    assert transcript["turns"][2]["persona_id"] == "technocrat"
    assert transcript["turns"][3]["persona_id"] == "ideologue"
    assert "pre_stances" not in transcript
    assert "post_stances" not in transcript

    for t in transcript["turns"]:
        assert len(t["statement"]) > 0


class RecordingLLMClient(BaseLLMClient):
    """Test client that records call parameters."""

    def __init__(self) -> None:
        self.received_seed: Optional[int] = None
        self.received_reasoning_effort: Optional[str] = None
        self.received_enable_reasoning: Optional[bool] = None

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
        self.received_seed = seed
        self.received_reasoning_effort = reasoning_effort
        self.received_enable_reasoning = enable_reasoning
        return "Recorded compromise statement."


def test_conduct_deliberation_forwards_model_seed():
    """Verify that model_config seed is properly forwarded to client.generate."""
    config = load_master_configuration("data/config.json")
    model_cfg = config.get("model_config")
    if model_cfg is not None:
        model_cfg["seed"] = 9999
    else:
        config["model_config"] = {"seed": 9999}
    item: PolicyItem = {
        "item_id": "ITEM_01",
        "title": "Test Policy",
        "topic_prompt": "Deliberate on policy",
        "ballot": {"options": {"A": "Key", "B": "Trap"}, "correct_key": "A"}
    }
    recorder = RecordingLLMClient()
    conduct_deliberation(
        client=recorder,
        config=config,
        item=item,
        dyad_id="D1",
        agents=["technocrat", "ideologue"],
        trial_index=0,
        turns_per_item=1
    )
    assert recorder.received_seed == 9999


def test_conduct_deliberation_forwards_reasoning_effort():
    """Verify that model_config reasoning_effort is properly forwarded to client.generate."""
    config = load_master_configuration("data/config.json")
    model_cfg = config.get("model_config")
    if model_cfg is not None:
        model_cfg["reasoning_effort"] = "low"
    else:
        config["model_config"] = {"reasoning_effort": "low"}
    item: PolicyItem = {
        "item_id": "ITEM_01",
        "title": "Test Policy",
        "topic_prompt": "Deliberate on policy",
        "ballot": {"options": {"A": "Key", "B": "Trap"}, "correct_key": "A"}
    }
    recorder = RecordingLLMClient()
    conduct_deliberation(
        client=recorder,
        config=config,
        item=item,
        dyad_id="D1",
        agents=["technocrat", "ideologue"],
        trial_index=0,
        turns_per_item=1
    )
    assert recorder.received_reasoning_effort == "low"


def test_conduct_deliberation_forwards_enable_reasoning():
    """Verify that model_config enable_reasoning is properly forwarded to client.generate."""
    config = load_master_configuration("data/config.json")
    model_cfg = config.get("model_config")
    if model_cfg is not None:
        model_cfg["enable_reasoning"] = False
    else:
        config["model_config"] = {"enable_reasoning": False}
    item: PolicyItem = {
        "item_id": "ITEM_01",
        "title": "Test Policy",
        "topic_prompt": "Deliberate on policy",
        "ballot": {"options": {"A": "Key", "B": "Trap"}, "correct_key": "A"}
    }
    recorder = RecordingLLMClient()
    conduct_deliberation(
        client=recorder,
        config=config,
        item=item,
        dyad_id="D1",
        agents=["technocrat", "ideologue"],
        trial_index=0,
        turns_per_item=1
    )
    assert recorder.received_enable_reasoning is False


def test_conduct_deliberation_uses_deliberation_start_on_opening_turn():
    """Verify that deliberation_start prompt is used for turn 0 and deliberation_turn for turn 1."""
    config = load_master_configuration("data/config.json")
    item: PolicyItem = {
        "item_id": "ITEM_01",
        "title": "Test Policy",
        "topic_prompt": "Deliberate on policy",
        "ballot": {"options": {"A": "Key", "B": "Trap"}, "correct_key": "A"}
    }
    prompts_received: list[str] = []

    class PromptCapturingClient(BaseLLMClient):
        def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 512,
            seed: Optional[int] = None,
            reasoning_effort: Optional[str] = None,
        ) -> str:
            prompts_received.append(prompt)
            return "Compromise statement."

    conduct_deliberation(
        client=PromptCapturingClient(),
        config=config,
        item=item,
        dyad_id="D1",
        agents=["technocrat", "ideologue"],
        trial_index=0,
        turns_per_item=2
    )

    assert len(prompts_received) == 2
    # Turn 0 should use deliberation_start template
    assert "opening the deliberation" in prompts_received[0].lower()
    # Turn 1 should use deliberation_turn template
    assert "conversation history:" in prompts_received[1].lower()


def test_openai_client_reasoning_payload(monkeypatch):
    """Verify that OpenAIClient correctly formats extra_body when enable_reasoning is toggled."""
    monkeypatch.setenv("OPENAI_API_KEY", "mock_key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.together.xyz/v1")

    from src.llm import OpenAIClient
    from unittest.mock import MagicMock

    # 1. Reasoning disabled
    client_disabled = OpenAIClient(model="Qwen/Qwen3.5-9B", enable_reasoning=False, reasoning_effort="low")
    mock_choice = MagicMock()
    mock_choice.message.content = "Response without reasoning."
    mock_choice.message.reasoning = None
    mock_response = MagicMock(choices=[mock_choice], usage=None)

    client_disabled.client = MagicMock()
    client_disabled.client.chat.completions.create.return_value = mock_response

    resp = client_disabled.generate(prompt="Hello")
    assert resp == "Response without reasoning."
    call_kwargs = client_disabled.client.chat.completions.create.call_args.kwargs
    assert call_kwargs.get("extra_body") == {
        "reasoning": {"enabled": False},
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert "reasoning_effort" not in call_kwargs

    # 2. Reasoning enabled
    client_enabled = OpenAIClient(model="Qwen/Qwen3.5-9B", enable_reasoning=True, reasoning_effort="low")
    client_enabled.client = MagicMock()
    client_enabled.client.chat.completions.create.return_value = mock_response

    resp2 = client_enabled.generate(prompt="Hello")
    assert resp2 == "Response without reasoning."
    call_kwargs2 = client_enabled.client.chat.completions.create.call_args.kwargs
    assert call_kwargs2.get("extra_body") == {
        "reasoning": {"enabled": True},
        "chat_template_kwargs": {"enable_thinking": True}
    }
    assert call_kwargs2.get("reasoning_effort") == "low"

