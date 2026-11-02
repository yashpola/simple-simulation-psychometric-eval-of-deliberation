"""OpenAI client abstraction using environment-configured base URL and API key.

Adheres to:
- Guideline 1: Strict built-in typing using TypedDict schemas.
- Guideline 2: Functional split for LLM inference execution.
- Strictly uses OpenAIClient (mock provider removed).
- Base URL and API key loaded strictly from environment (.env).
- Model identifier configured strictly in config.json (model_config.agent_model).
"""

import abc
import os
from typing import Optional, Any, Dict
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

try:
    from dotenv import load_dotenv  # pyright: ignore[reportUnknownVariableType]
    _ = load_dotenv()
except ImportError:
    pass

from src.schemas import ModelConfig


def clean_llm_response(text: str) -> str:
    """Return raw response text stripped of surrounding whitespace without heuristic filtering."""
    return text.strip() if text else ""


# Model pricing table: (USD per 1M input tokens, USD per 1M output tokens)
MODEL_PRICING_TABLE: dict[str, tuple[float, float]] = {
    "deepseek-ai/deepseek-v4.1-flash-1017": (0.30, 1.20),
    "deepseek-ai/deepseek-v4.1-flash": (0.30, 1.20),
    "deepseek-v4.1-flash": (0.30, 1.20),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "gpt-oss-120b": (0.15, 0.60),
    "qwen/qwen3.5-9b": (0.10, 0.10),
    "qwen3.5-9b": (0.10, 0.10),
    "meta-models/muse-spark-open-816b-a42b": (1.25, 4.25),
    "muse-spark": (1.25, 4.25),
    "meta-llama/llama-3.3-70b-instruct": (0.88, 0.88),
    "llama-3.3-70b": (0.88, 0.88),
}


class BaseLLMClient(abc.ABC):
    """Abstract base class for LLM inference."""

    def __init__(self) -> None:
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        seed: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        enable_reasoning: Optional[bool] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from prompt and system prompt."""
        pass

    def get_token_usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def get_cost_usd(self) -> float:
        return 0.0


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI and OpenAI-compatible endpoints (Together AI, vLLM, Ollama, OpenRouter).

    Model is configured strictly in config.json.
    API key and base URL credentials are loaded strictly from the environment (.env).
    """

    def __init__(
        self,
        model: str,
        enable_reasoning: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ):
        super().__init__()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing required environment variable: OPENAI_API_KEY. "
                "Please configure your API key in .env as instructed in README.md."
            )

        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise ValueError(
                "Missing required environment variable: OPENAI_BASE_URL. "
                "Please configure your base URL in .env as instructed in README.md."
            )

        if not model:
            raise ValueError(
                "Missing required model configuration: 'agent_model' in model_config of config.json. "
                "The model must be specified in config.json."
            )

        self.model: str = model
        self.enable_reasoning: Optional[bool] = enable_reasoning
        self.reasoning_effort: Optional[str] = reasoning_effort
        self.client: OpenAI = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        seed: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        enable_reasoning: Optional[bool] = None,
        **kwargs: Any,
    ) -> str:
        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        eff_enable_reasoning = enable_reasoning if enable_reasoning is not None else self.enable_reasoning
        eff_reasoning_effort = reasoning_effort if reasoning_effort is not None else self.reasoning_effort

        req_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            req_kwargs["seed"] = seed

        extra_body: dict[str, Any] = {}
        if eff_enable_reasoning is False:
            extra_body["reasoning"] = {"enabled": False}
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        elif eff_enable_reasoning is True:
            extra_body["reasoning"] = {"enabled": True}
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}
            if eff_reasoning_effort is not None and str(eff_reasoning_effort).lower() != "none":
                req_kwargs["reasoning_effort"] = eff_reasoning_effort
        elif eff_reasoning_effort is not None and str(eff_reasoning_effort).lower() != "none":
            req_kwargs["reasoning_effort"] = eff_reasoning_effort

        if extra_body:
            req_kwargs["extra_body"] = extra_body

        response = self.client.chat.completions.create(**req_kwargs)

        choice = response.choices[0]
        msg = choice.message
        content = (msg.content or "").strip()
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""

        # Track token usage from response or fallback estimate
        usage = getattr(response, "usage", None)
        if usage is not None:
            p_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            c_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            t_tokens = int(getattr(usage, "total_tokens", 0) or (p_tokens + c_tokens))
        else:
            p_tokens = max(1, (len(prompt) + len(system_prompt or "")) // 4)
            c_tokens = max(1, len(content) // 4)
            t_tokens = p_tokens + c_tokens

        self.total_prompt_tokens += p_tokens
        self.total_completion_tokens += c_tokens
        self.total_tokens += t_tokens

        if content:
            return content

        if reasoning:
            return str(reasoning).strip()

        return ""

    def get_cost_usd(self) -> float:
        """Calculate total estimated inference cost in USD based on tracked token usage."""
        model_key = self.model.lower()
        pricing: Optional[tuple[float, float]] = None
        for k, v in MODEL_PRICING_TABLE.items():
            if k in model_key or model_key in k:
                pricing = v
                break
        if not pricing:
            pricing = (0.30, 1.20)

        in_rate, out_rate = pricing
        cost = (self.total_prompt_tokens / 1_000_000.0) * in_rate + (self.total_completion_tokens / 1_000_000.0) * out_rate
        return round(cost, 6)


def get_llm_client(
    config: ModelConfig,
    is_judge: bool = False
) -> OpenAIClient:
    """Factory function returning configured OpenAIClient reading endpoint credentials directly from .env."""
    model = config.get("agent_model")
    if not model:
        raise ValueError(
            "Missing required configuration: 'agent_model' in model_config of config.json. "
            "The model must be specified in config.json."
        )

    enable_reasoning = config.get("enable_reasoning")
    if enable_reasoning is None:
        enable_reasoning = config.get("reasoning_enabled")
    reasoning_effort = config.get("reasoning_effort")

    return OpenAIClient(
        model=model,
        enable_reasoning=enable_reasoning,
        reasoning_effort=reasoning_effort,
    )
