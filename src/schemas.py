"""Strict typing schemas using Python built-in typing module (TypedDict).

Simplified benchmark observing latent deliberative competence and item psychometrics across diverse citizen groups.
"""

from typing import TypedDict, List, Dict, Optional, Any, NotRequired


class ItemBallot(TypedDict):
    options: Dict[str, str]
    correct_key: str
    prompt: NotRequired[str]


class PolicyItem(TypedDict):
    item_id: str
    title: str
    topic_prompt: str
    ballot: ItemBallot


class DyadConfig(TypedDict, total=False):
    id: str
    name: str
    dynamic: str
    agents: List[str]


class DeliberationConfig(TypedDict, total=False):
    turns_per_item: int
    trials_per_item: int
    mcq_option_order_seed: int
    seed: int
    dyads: List[DyadConfig]


class ModelConfig(TypedDict, total=False):
    agent_model: str
    temperature: float
    max_tokens: int
    deliberation_max_tokens: int
    deliberation_max_new_tokens: int
    ballot_max_tokens: int
    ballot_max_new_tokens: int
    reasoning_effort: Optional[str]
    enable_reasoning: Optional[bool]
    reasoning_enabled: Optional[bool]
    seed: Optional[int]


class ExperimentMetadata(TypedDict, total=False):
    name: str
    description: str
    version: str


class ExecutionMetadata(TypedDict, total=False):
    experiment_name: str
    model: str
    cost_usd: float
    runtime_seconds: float
    runtime_formatted: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    timestamp: str


class SystemPrompts(TypedDict, total=False):
    technocrat: str
    ideologue: str
    populist: str
    everyman: str


class UserPrompts(TypedDict, total=False):
    deliberation_start: str
    deliberation_turn: str
    secret_ballot: str


class MasterConfig(TypedDict, total=False):
    experiment_metadata: ExperimentMetadata
    model_config: ModelConfig
    system_prompts: SystemPrompts
    user_prompts: UserPrompts
    items: List[str]
    deliberation_config: DeliberationConfig
    output_dir: Optional[str]


class DeliberationTurn(TypedDict):
    turn_index: int
    speaker: str
    persona_id: str
    statement: str


class DeliberationTranscript(TypedDict):
    item_id: str
    item_title: str
    dyad_id: str
    agents: List[str]
    trial_index: int
    turns: List[DeliberationTurn]


class BallotVote(TypedDict):
    item_id: str
    dyad_id: str
    trial_index: int
    persona_id: str
    raw_response: str
    displayed_letter: Optional[str]
    selected_key: Optional[str]
    is_correct: bool
    is_valid: bool
    display_mapping: Dict[str, str]


class DyadBallotRecord(TypedDict):
    item_id: str
    dyad_id: str
    trial_index: int
    votes: Dict[str, BallotVote]
    consensus_reached: bool


class CTTMetrics(TypedDict):
    item_id: str
    difficulty_p: float
    discrimination_index: float
    distractor_counts: Dict[str, int]
    distractor_ratios: Dict[str, float]


class IRTMetrics(TypedDict):
    item_id: str
    difficulty_b: float
    discrimination_a: float


class IRTDeliberators(TypedDict):
    dyad_theta: Dict[str, float]
    individual_theta: Dict[str, float]
    persona_by_dyad_theta: Dict[str, Dict[str, float]]


class IRTResults(TypedDict):
    items: Dict[str, IRTMetrics]
    deliberators: IRTDeliberators


class PsychometricResults(TypedDict, total=False):
    ctt: Dict[str, CTTMetrics]
    irt: IRTResults
    individual_success_rates: Dict[str, float]
    insights: Dict[str, Any]
