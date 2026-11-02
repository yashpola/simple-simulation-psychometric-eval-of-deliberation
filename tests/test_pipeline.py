"""End-to-end integration test running the simplified benchmark pipeline."""

import json
from typing import Any, Optional
import pytest
from unittest.mock import patch
from pathlib import Path
from src.main import execute_simulation_run
from src.llm import OpenAIClient


def dummy_openai_generate(
    self: Any,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    seed: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Offline test double for OpenAIClient.generate during automated unit tests."""
    prompt_lower = prompt.lower()
    if "select the single resolution" in prompt_lower or "options (in randomized order)" in prompt_lower or "ballot" in prompt_lower:
        return "A"
    return "I propose an integrative compromise that addresses both constraints."


def test_full_experiment_pipeline(tmp_path: Path) -> None:
    output_dir = tmp_path / "test_exp_output"

    # Prepare test configuration with 1 trial per item and explicit test output directory
    with open("data/config.json", "r", encoding="utf-8") as f:
        test_cfg = json.load(f)
    test_cfg["deliberation_config"]["trials_per_item"] = 1
    test_cfg["deliberation_config"]["mcq_option_order_seed"] = 101
    test_cfg["output_dir"] = str(output_dir)

    test_cfg_path = tmp_path / "test_config.json"
    with open(test_cfg_path, "w", encoding="utf-8") as f:
        json.dump(test_cfg, f)

    # Patch OpenAIClient.generate to execute offline during automated testing
    with patch.object(OpenAIClient, "generate", dummy_openai_generate):
        result_dir = execute_simulation_run(config_path=str(test_cfg_path))

    # 1. Verify directory exists
    assert result_dir.exists()

    # 2. Verify organized subdirectories exist (inputs/, model_outputs/, evals/, metadata/)
    inputs_dir = result_dir / "inputs"
    model_outputs_dir = result_dir / "model_outputs"
    evals_dir = result_dir / "evals"
    metadata_dir = result_dir / "metadata"

    assert inputs_dir.is_dir()
    assert model_outputs_dir.is_dir()
    assert evals_dir.is_dir()
    assert metadata_dir.is_dir()

    # 2a. Verify inputs subdirectory (clean snapshot without runtime/cost)
    assert (inputs_dir / "config.json").exists()
    with open(inputs_dir / "config.json", "r", encoding="utf-8") as f:
        saved_cfg = json.load(f)
    assert "cost_usd" not in saved_cfg.get("experiment_metadata", {})
    assert "runtime_seconds" not in saved_cfg.get("experiment_metadata", {})

    items_dir = inputs_dir / "items"
    assert items_dir.is_dir()
    item_files = list(items_dir.glob("*.json"))
    assert len(item_files) >= 3
    for item_file in item_files:
        assert item_file.name.endswith("_description.json")

    # 2b. Verify model_outputs subdirectory
    assert (model_outputs_dir / "ballots.json").exists()
    transcripts_dir = model_outputs_dir / "transcripts"
    assert transcripts_dir.is_dir()

    transcript_files = list(transcripts_dir.glob("*.json"))
    assert len(transcript_files) >= 3
    for trans_file in transcript_files:
        assert trans_file.name.endswith("_transcript.json")

    transcript_pdfs = list(transcripts_dir.glob("*.pdf"))
    assert len(transcript_pdfs) >= 3
    for pdf_file in transcript_pdfs:
        assert pdf_file.name.endswith("_transcript.pdf")
        assert pdf_file.stat().st_size > 1000

    for t_file in transcript_files:
        with open(t_file, "r", encoding="utf-8") as f:
            item_transcripts = json.load(f)
        assert len(item_transcripts) > 0
        first_t = item_transcripts[0]
        assert "item_id" in first_t
        assert "turns" in first_t
        assert len(first_t["turns"]) == 4
        assert "pre_stances" not in first_t
        assert "post_stances" not in first_t

    # 2c. Verify evals subdirectory
    assert (evals_dir / "psychometric_evals.json").exists()
    assert (evals_dir / "final_report.md").exists()

    # 2d. Verify metadata subdirectory
    assert (metadata_dir / "metadata.json").exists()

    with open(metadata_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta_json = json.load(f)
    assert "cost_usd" in meta_json
    assert "runtime_seconds" in meta_json
    assert "model" in meta_json

    # 3. Verify ballot contents
    with open(model_outputs_dir / "ballots.json", "r", encoding="utf-8") as f:
        ballots_data = json.load(f)
    assert len(ballots_data) > 0
    first_ballot = ballots_data[0]
    assert "consensus_reached" in first_ballot
    assert "votes" in first_ballot

    # 4. Verify CTT & IRT metrics structure
    with open(evals_dir / "psychometric_evals.json", "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    assert "ctt" in metrics_data
    assert "irt" in metrics_data
    assert "items" in metrics_data["irt"]
    assert "deliberators" in metrics_data["irt"]
    assert "dyad_theta" in metrics_data["irt"]["deliberators"]
    assert "individual_theta" in metrics_data["irt"]["deliberators"]
    assert "persona_by_dyad_theta" in metrics_data["irt"]["deliberators"]
    assert "individual_success_rates" in metrics_data
    assert "insights" in metrics_data

    # 5. Verify markdown report contents
    with open(evals_dir / "final_report.md", "r", encoding="utf-8") as f:
        report_text = f.read()
    assert "# Deliberative Item Response Theory (IRT) Benchmark Report" in report_text
    assert "Standardized Policy Battery Psychometrics" in report_text
    assert "Latent Deliberative Ability Breakdown" in report_text
    assert "Cognitive Trap & Distractor Vulnerability Analysis" in report_text


def test_main_cli_execution(tmp_path: Path) -> None:
    """Test invoking main() via CLI argument parsing with --config."""
    import sys
    from src.main import main

    output_dir = tmp_path / "cli_test_exp_output"
    with open("data/config.json", "r", encoding="utf-8") as f:
        test_cfg = json.load(f)
    test_cfg["deliberation_config"]["trials_per_item"] = 1
    test_cfg["output_dir"] = str(output_dir)

    test_cfg_path = tmp_path / "cli_config.json"
    with open(test_cfg_path, "w", encoding="utf-8") as f:
        json.dump(test_cfg, f)

    with patch.object(sys, "argv", ["main.py", "--config", str(test_cfg_path)]):
        with patch.object(OpenAIClient, "generate", dummy_openai_generate):
            main()

    assert (output_dir / "evals" / "final_report.md").exists()
    assert (output_dir / "inputs" / "config.json").exists()
    assert (output_dir / "inputs" / "items").is_dir()
    assert (output_dir / "model_outputs" / "transcripts").is_dir()
    assert (output_dir / "metadata" / "metadata.json").exists()


def test_missing_env_credentials_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that if OPENAI_API_KEY or OPENAI_BASE_URL is missing, it fails fast with ValueError."""
    from src.llm import OpenAIClient

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIClient(model="openai/gpt-oss-120b")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        OpenAIClient(model="openai/gpt-oss-120b")


def test_missing_model_in_config_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that model must be configured in config.json and is never extracted from environment variables."""
    from src.llm import get_llm_client
    from src.schemas import ModelConfig

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.test/v1")
    monkeypatch.setenv("AGENT_MODEL", "rogue-env-model")
    monkeypatch.setenv("DEFAULT_AGENT_MODEL", "rogue-env-model-2")
    monkeypatch.setenv("MODEL_GPT_OSS", "rogue-env-model-3")

    cfg: ModelConfig = {}
    with pytest.raises(ValueError, match="agent_model"):
        get_llm_client(cfg)


def test_empty_deliberations_or_ballots_aborts_report_generation(tmp_path: Path) -> None:
    """Verify that when all responses are empty, simulation saves base diagnostics but aborts report generation."""
    output_dir = tmp_path / "test_abort_output"

    with open("data/config.json", "r", encoding="utf-8") as f:
        test_cfg = json.load(f)
    test_cfg["deliberation_config"]["trials_per_item"] = 1
    test_cfg["output_dir"] = str(output_dir)

    test_cfg_path = tmp_path / "abort_config.json"
    with open(test_cfg_path, "w", encoding="utf-8") as f:
        json.dump(test_cfg, f)

    def empty_generate(self: Any, prompt: str, **kwargs: Any) -> str:
        # Simulate model exhausting tokens on reasoning with empty response content
        return ""

    with patch.object(OpenAIClient, "generate", empty_generate):
        result_dir = execute_simulation_run(config_path=str(test_cfg_path))

    assert result_dir == output_dir
    # Base diagnostic artifacts MUST exist
    assert (result_dir / "inputs" / "config.json").exists()
    assert (result_dir / "model_outputs" / "ballots.json").exists()
    assert (result_dir / "inputs" / "items").is_dir()
    assert (result_dir / "model_outputs" / "transcripts").is_dir()
    assert (result_dir / "metadata" / "metadata.json").exists()

    # Evaluative artifacts MUST NOT exist (aborted to avoid contaminated psychometrics)
    assert not (result_dir / "evals" / "final_report.md").exists()
    assert not (result_dir / "evals" / "psychometric_evals.json").exists()


def test_individual_empty_ballot_discarded_with_warning(tmp_path: Path) -> None:
    """Verify that an individual empty ballot is marked is_valid=False, logged, and discarded from CTT."""
    output_dir = tmp_path / "test_discard_output"

    with open("data/config.json", "r", encoding="utf-8") as f:
        test_cfg = json.load(f)
    test_cfg["deliberation_config"]["trials_per_item"] = 1
    test_cfg["output_dir"] = str(output_dir)

    test_cfg_path = tmp_path / "discard_config.json"
    with open(test_cfg_path, "w", encoding="utf-8") as f:
        json.dump(test_cfg, f)

    call_count = {"ballots": 0}

    def flaky_ballot_generate(self: Any, prompt: str, **kwargs: Any) -> str:
        prompt_lower = prompt.lower()
        if "select the single resolution" in prompt_lower or "ballot" in prompt_lower:
            call_count["ballots"] += 1
            if call_count["ballots"] == 1:
                return ""  # First ballot produces empty response
            return "A"
        return "I propose an integrative compromise that addresses both constraints."

    with patch.object(OpenAIClient, "generate", flaky_ballot_generate):
        result_dir = execute_simulation_run(config_path=str(test_cfg_path))

    # Pipeline completes because not ALL ballots/deliberations are empty
    assert (result_dir / "evals" / "final_report.md").exists()
    assert (result_dir / "evals" / "psychometric_evals.json").exists()

    with open(result_dir / "evals" / "psychometric_evals.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["insights"]["discarded_ballot_count"] == 1
    assert metrics["insights"]["valid_votes_count"] == metrics["insights"]["total_votes_cast"] - 1

    with open(result_dir / "evals" / "final_report.md", "r", encoding="utf-8") as f:
        report = f.read()

    assert "> [!WARNING]" in report
    assert "Data Quality Alert" in report
    assert "1 of" in report


def test_token_limits_forwarded_from_config(tmp_path: Path) -> None:
    """Verify deliberation_max_tokens and ballot_max_tokens are forwarded to client.generate."""
    from src.deliberation import conduct_deliberation
    from src.ballot import administer_agent_ballot
    from src.schemas import PolicyItem

    recorded_tokens: list[int] = []

    class MockClient(OpenAIClient):
        def __init__(self) -> None:
            self.model = "mock"

        def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 512,
            seed: Optional[int] = None,
            reasoning_effort: Optional[str] = None,
            **kwargs: Any,
        ) -> str:
            recorded_tokens.append(max_tokens)
            return "A"

    dummy_client = MockClient()
    mock_item: PolicyItem = {
        "item_id": "TEST_01",
        "title": "Test Title",
        "topic_prompt": "Test Prompt",
        "ballot": {
            "prompt": "Test Ballot",
            "correct_key": "A",
            "options": {"A": "Opt A", "B": "Opt B", "C": "Opt C", "D": "Opt D"}
        }
    }

    mock_config = {
        "model_config": {
            "agent_model": "test-model",
            "deliberation_max_tokens": 777,
            "ballot_max_tokens": 333
        },
        "system_prompts": {"technocrat": "System"},
        "user_prompts": {
            "deliberation_turn": "{title} {topic_prompt} {conversation_history}",
            "secret_ballot": "{title} {topic_prompt} {transcript_summary} {ballot_prompt} {formatted_options}"
        }
    }

    conduct_deliberation(
        client=dummy_client,
        config=mock_config,  # type: ignore[arg-type]
        item=mock_item,
        dyad_id="D1",
        agents=["technocrat", "technocrat"],
        trial_index=0,
        turns_per_item=1
    )
    assert recorded_tokens[-1] == 777

    administer_agent_ballot(
        client=dummy_client,
        config=mock_config,  # type: ignore[arg-type]
        persona_id="technocrat",
        item=mock_item,
        dyad_id="D1",
        trial_index=0,
        transcript_summary="Summary"
    )
    assert recorded_tokens[-1] == 333

