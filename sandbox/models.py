"""Pydantic data models shared across the sandbox.

Per full_design_doc.md §4: all models are Pydantic BaseModel subclasses,
superseding the HLD's original dataclass sketches.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class StanceRecord(BaseModel):
    turn: int = Field(ge=1)
    stance_value: float
    reason_text: str
    interaction_id: str


class Agent(BaseModel):
    agent_id: str
    run_id: str
    persona: str = Field(min_length=1)
    language_condition: str = Field(pattern="^(english|hinglish|hindi)$")
    model_backend_id: str
    initial_stance: float
    stance_history: list[StanceRecord] = Field(default_factory=list)
    memory_window: list[str] = Field(default_factory=list, max_length=5)
    created_at_turn: int = 0

    def current_stance(self) -> float:
        """Last stance_history entry's value, or initial_stance if empty."""
        return self.stance_history[-1].stance_value if self.stance_history else self.initial_stance


class MemeContent(BaseModel):
    meme_id: str
    image_path: str
    caption_text: str = Field(min_length=1)
    stance_label: float
    offensiveness_label: float | None = None
    source_dataset: str


class Interaction(BaseModel):
    interaction_id: str
    run_id: str
    turn: int = Field(ge=1)
    speaker_agent_id: str
    neighbor_agent_ids: list[str] = Field(min_length=1)
    stance_before: float
    stance_after: float
    reason_text: str
    content_type: str = Field(pattern="^(generated_text|meme)$")
    meme_id: str | None = None
    used_vision_fallback: bool = False
    prompt_token_count: int = Field(ge=0)
    completion_token_count: int = Field(ge=0)
    model_backend_id: str
    latency_ms: int = Field(ge=0)
    api_call_status: str = Field(pattern="^(success|retried_success|failed_logged_null)$")
    timestamp_utc: str

    @model_validator(mode="after")
    def check_meme_id_consistency(self) -> "Interaction":
        if self.content_type == "meme" and self.meme_id is None:
            raise ValueError("content_type='meme' requires a non-null meme_id")
        if self.content_type == "generated_text" and self.meme_id is not None:
            raise ValueError("content_type='generated_text' requires meme_id to be null")
        return self

    @property
    def is_valid_for_analysis(self) -> bool:
        return self.api_call_status != "failed_logged_null"


class MemeInjectionConfig(BaseModel):
    enabled: bool = False
    meme_pool_id: str | None = None
    injection_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    injection_schedule: str = Field(default="random", pattern="^(random|fixed_turn)$")
    fixed_turns: set[int] = Field(default_factory=set)

    @model_validator(mode="after")
    def check_enabled_requirements(self) -> "MemeInjectionConfig":
        if self.enabled:
            if self.meme_pool_id is None:
                raise ValueError("meme_pool_id is required when meme_injection.enabled is True")
            if self.injection_schedule == "fixed_turn" and not self.fixed_turns:
                raise ValueError("fixed_turns must be non-empty when injection_schedule='fixed_turn'")
        return self


class ExperimentRun(BaseModel):
    run_id: str
    rq_target: str = Field(pattern="^(RQ1_RQ2|RQ3|RQ1_RQ2_RQ3)$")
    mode: str = Field(default="multi_turn", pattern="^(multi_turn)$")
    topic: str = Field(min_length=1)
    alpha: float = Field(ge=0.0, le=1.0)
    M: int = Field(ge=2)
    N: int = Field(ge=1)
    K: int = Field(ge=1)
    trial_number: int = Field(ge=1)
    language_condition: str = Field(pattern="^(english|hinglish|hindi)$")
    model_backend_id: str = Field(min_length=1)
    stance_scale: list[float] = Field(min_length=2)
    persona_pool_id: str = Field(min_length=1)
    meme_injection: MemeInjectionConfig = Field(default_factory=MemeInjectionConfig)
    seed: int
    temperature: float = Field(ge=0.0, le=2.0)
    max_cost_usd: float | None = Field(default=None, gt=0.0)
    rate_limits: dict[str, int] = Field(default_factory=dict)
    checkpoint_every_n_turns: int = Field(default=1, ge=1)
    git_commit_hash: str = ""
    started_at_utc: str | None = None
    completed_at_utc: str | None = None
    status: str = Field(default="pending", pattern="^(pending|running|completed|failed_partial|failed_total)$")
    total_cost_usd: float = 0.0

    @model_validator(mode="after")
    def check_n_less_than_m(self) -> "ExperimentRun":
        if self.N >= self.M:
            raise ValueError(f"N ({self.N}) must be less than M ({self.M})")
        return self


class CheckpointState(BaseModel):
    run_id: str
    last_completed_turn: int
    agent_snapshot: list[Agent]
