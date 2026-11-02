"""Integration test verifying that the configured agent model is available on Together AI / endpoint."""

import os
import json
import pytest
from openai import APIError, APIConnectionError

try:
    from dotenv import load_dotenv  # pyright: ignore[reportUnknownVariableType]
    _ = load_dotenv()
except ImportError:
    pass

from src.llm import OpenAIClient


def test_configured_model_available_on_endpoint():
    """Verify that the model configured in data/config.json is active and serving on the endpoint."""
    config_path = "data/config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    model_name = config_data.get("model_config", {}).get("agent_model")
    assert model_name, "model_config.agent_model must be defined in data/config.json"

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")

    if not api_key or not base_url:
        pytest.skip("OPENAI_API_KEY and OPENAI_BASE_URL must be configured in environment to probe endpoint.")

    client = OpenAIClient(model=model_name)

    try:
        # Probe model with minimal 1-token prompt
        response = client.generate(
            prompt="Reply with 'OK'.",
            max_tokens=1,
            temperature=0.0
        )
        assert response is not None
    except APIConnectionError as conn_err:
        pytest.skip(f"Offline / network unreachable (sandbox mode): {conn_err}")
    except APIError as api_err:
        err_msg = str(api_err).lower()
        if "model_not_available" in err_msg or "non-serverless" in err_msg or "does not exist" in err_msg:
            pytest.fail(
                f"Configured model '{model_name}' is not available on endpoint ({base_url}): {api_err}"
            )
        raise api_err

