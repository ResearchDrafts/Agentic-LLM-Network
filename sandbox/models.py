"""Pydantic data models shared across the sandbox.

Per full_design_doc.md §4: all models are Pydantic BaseModel subclasses,
superseding the HLD's original dataclass sketches.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StanceRecord(BaseModel):
    turn: int = Field(ge=1)
    stance_value: float
    reason_text: str
    interaction_id: str


class Agent(BaseModel):
    # validate_assignment re-checks field constraints on assignment, not just
    # at construction. This exists for memory_window's max_length=5 (Fix A):
    # without it, `agent.memory_window = [8 items]` silently succeeds and the
    # cap is enforced only by agent_manager.py's [-5:] slice, i.e. by
    # convention rather than by the schema -- which is precisely what
    # full_design_doc.md Sec 4.1 claims is NOT the case.
    #
    # Known limit: this catches assignment, not in-place mutation. Pydantic
    # cannot see `agent.memory_window.append(...)`. Fix A is therefore also
    # re-enforced at the point of use, where prompt_builder.py slices [-5:]
    # when rendering memory into a prompt -- the prompt is the only thing
    # that affects experimental results, so that is the cap that matters.
    model_config = ConfigDict(validate_assignment=True)

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
    # extra="forbid" for the same reason as ExperimentRun below: this block is
    # hand-written YAML, and a typo'd key here silently disables the exact
    # mechanism RQ3 manipulates.
    model_config = ConfigDict(extra="forbid")

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
    # extra="forbid" so an unrecognized YAML key raises instead of vanishing.
    # Twelve of this model's fields are optional, so without it a typo is
    # silently absorbed and the run proceeds on the default. The dangerous
    # case is `meme_injections:` (plural): the config loads, meme injection
    # stays off, and RQ3's meme-present arm runs as a second meme-absent arm,
    # producing a clean null result that reads as "memes do not matter".
    # `max_cost_usd` and `rate_limits` fail the same way, just less quietly.
    model_config = ConfigDict(extra="forbid")

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
    # Endpoint for a self-hosted backend, e.g. "http://localhost:8000/v1" for
    # vLLM. Lives on the run rather than in an environment variable so it is
    # captured in run_config.json and the run stays reproducible: which server
    # produced a dataset is part of its provenance.
    api_base: str | None = None
    stance_scale: list[float] = Field(min_length=2)
    # Anchor text for the two ends of stance_scale, rendered into the prompt
    # by prompt_builder.py. An unanchored numeric scale is interpreted
    # inconsistently by an LLM across turns and personas, so these exist to
    # make a stance number mean the same thing to every agent. Optional with
    # generic defaults, so existing configs stay valid; set them to something
    # topic-appropriate for a real run (phase4.md Q4.4).
    stance_low_label: str = Field(default="strongly opposed", min_length=1)
    stance_high_label: str = Field(default="strongly in favour", min_length=1)
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

    @model_validator(mode="after")
    def check_stance_scale_distinct(self) -> "ExperimentRun":
        """full_design_doc.md Sec 7.1 requires at least 2 DISTINCT values.

        Field(min_length=2) only counts entries, so [3, 3] passed. Then
        min == max, clamp_and_validate_scale accepts exactly one value, every
        reply that is not that value raises StanceParseFailure, and the whole
        population ends up failed_logged_null. Loud once it happens, but
        cheaper to reject at config load.
        """
        if len(set(self.stance_scale)) < 2:
            raise ValueError(
                f"stance_scale needs at least 2 distinct values, got {self.stance_scale}"
            )
        return self


class CheckpointState(BaseModel):
    run_id: str
    last_completed_turn: int
    agent_snapshot: list[Agent]
