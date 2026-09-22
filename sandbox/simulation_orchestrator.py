"""The per-turn simulation loop (full_design_doc.md Sec 3.9, Tier 3).

The only module that calls every other component. Everything below the
integration tier is deliberately unaware of this one.

Ported from Sec 3.9's sample, with the gaps that sample leaves open resolved
per phase4.md Section 2: run-directory creation, the exception taxonomy, the
run-lifecycle fields nothing previously read, SeedManager wiring, and passing
the multimodal prompt through to the gateway.

**Fix F, stated as an invariant because nothing structurally enforces it.**
`AgentManager.all_agents()` hands back live `Agent` references, so the frozen
snapshot is a convention, not a guarantee. The rule: steps 2 and 3 (neighbour
resolution, meme-injection resolution) run to completion before any dispatch
begins, and no dispatch coroutine mutates agent state. All mutation happens in
step 6, after `asyncio.gather` returns. Break that and runs stop being
reproducible, silently.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sandbox.agent_manager import AgentManager
from sandbox.checkpoint_manager import CheckpointManager
from sandbox.cost_tracker import CostTracker
from sandbox.interaction_engine import AlphaSampling, RecommendationStrategy
from sandbox.logging_writer import LoggingWriter
from sandbox.meme_pool_manager import MemePoolManager
from sandbox.model_gateway import FatalGatewayError, ModelGateway
from sandbox.models import Agent, ExperimentRun, Interaction, MemeContent
from sandbox.prompt_builder import PromptBuilder
from sandbox.seed_manager import SeedManager
from sandbox.stance_parser import (
    StanceParseFailure,
    clamp_and_validate_scale,
    parse_stance,
)

# Fix I: one initial attempt plus two retries with an emphasised format block.
MAX_STANCE_ATTEMPTS = 3


class SimulationOrchestrator:
    def __init__(
        self,
        run: ExperimentRun,
        agent_manager: AgentManager,
        interaction_engine: RecommendationStrategy,
        meme_pool: MemePoolManager,
        gateway: ModelGateway,
        logger: LoggingWriter,
        checkpoint_mgr: CheckpointManager,
        prompt_builder: PromptBuilder,
        neighbor_rng: random.Random,
        cost_tracker: CostTracker | None = None,
        runs_dir: Path = Path("runs"),
    ):
        self._run = run
        self._agent_manager = agent_manager
        self._interaction_engine = interaction_engine
        self._meme_pool = meme_pool
        self._gateway = gateway
        self._logger = logger
        self._checkpoint_mgr = checkpoint_mgr
        self._prompt_builder = prompt_builder
        # Must be SeedManager.neighbor_sampling_rng, never a fresh Random and
        # never the meme stream: the three are offset precisely so they do not
        # correlate, and reproducibility depends on each stochastic choice
        # drawing from its own.
        self._neighbor_rng = neighbor_rng
        self._cost_tracker = cost_tracker
        self._run_dir = runs_dir / run.run_id
        # agent_id -> that agent's most recent Interaction. This is what
        # prompt_builder renders as neighbour posts (Q4.1). Mutated only in
        # step 6.
        self._last_posts: dict[str, Interaction] = {}

    async def run(self) -> None:
        """Entry point. Resumes from a checkpoint when one exists, else starts
        fresh at turn 1."""
        self._run_dir.mkdir(parents=True, exist_ok=True)

        resume_state = self._checkpoint_mgr.load(self._run.run_id)
        if resume_state is not None:
            start_turn = resume_state.last_completed_turn + 1
            self._agent_manager.restore_from_snapshot(resume_state.agent_snapshot)
            self._last_posts = self._rebuild_last_posts()
        else:
            self._agent_manager.initialize_population()
            self._run.status = "running"
            self._run.started_at_utc = _now_utc_iso()
            self._logger.write_run_config(self._run)
            start_turn = 1

        last_completed = start_turn - 1
        try:
            for turn in range(start_turn, self._run.K + 1):
                await self._run_turn(turn)
                last_completed = turn
                if turn % self._run.checkpoint_every_n_turns == 0:
                    self._save_checkpoint(turn)
        except BaseException:
            # Any exception reaching here is fatal by construction: the two
            # recoverable failure modes are absorbed per-agent below. Record
            # what was reached before re-raising, so a resume knows where to
            # pick up and the run does not look "running" forever.
            self._finalize("failed_partial" if last_completed else "failed_total")
            raise

        # The loop's modulo can skip the final turn (K=5 with
        # checkpoint_every_n_turns=2 checkpoints 2 and 4, never 5), which
        # would make a completed run resume from the wrong place.
        if last_completed and last_completed % self._run.checkpoint_every_n_turns != 0:
            self._save_checkpoint(last_completed)

        self._logger.write_agents_final(self._agent_manager.all_agents())
        self._finalize("completed")

    async def _run_turn(self, turn: int) -> None:
        """One turn, in the exact order Fix F requires."""
        all_agents = self._agent_manager.all_agents()

        # Steps 2 and 3 are sequential with respect to each other, not merely
        # before dispatch: they draw from two different seeded Random
        # instances, and a single Random is not safe under concurrent use
        # (full_design_doc.md:1528).
        neighbor_map = {
            agent.agent_id: self._interaction_engine.select_neighbors(
                agent, all_agents, turn, self._neighbor_rng
            )
            for agent in all_agents
        }
        meme_map = self._meme_pool.resolve_injections_for_turn(all_agents, turn)

        # The frozen snapshot of what everyone last said. Copied so step 6's
        # writes cannot be observed by a dispatch task still in flight.
        frozen_posts = dict(self._last_posts)

        results = await asyncio.gather(
            *(
                self._process_one_agent_turn(
                    agent,
                    neighbor_map[agent.agent_id],
                    meme_map[agent.agent_id],
                    frozen_posts,
                    turn,
                )
                for agent in all_agents
            ),
            return_exceptions=True,
        )

        # Step 6.
        for agent, result in zip(all_agents, results, strict=True):
            if isinstance(result, BaseException):
                # Sec 3.9's sample `continue`s here, which silently swallows
                # CostCeilingExceeded and contradicts Sec 3.11's statement
                # that it is fatal. Re-raise: StanceParseFailure and
                # FatalGatewayError are already absorbed below, so anything
                # arriving here is an unanticipated bug and must surface.
                raise result
            await self._logger.write_interaction(result)
            self._agent_manager.apply_interaction(agent.agent_id, result)
            self._last_posts[agent.agent_id] = result

    async def _process_one_agent_turn(
        self,
        agent: Agent,
        neighbors: list[Agent],
        meme: MemeContent | None,
        frozen_posts: dict[str, Interaction],
        turn: int,
    ) -> Interaction:
        stance_before = _current_stance_for_logging(agent, turn)

        if meme is not None:
            return self._build_meme_interaction(
                agent, neighbors, meme, turn, stance_before
            )

        # Neighbours who have not spoken yet (turn 1, or an agent whose every
        # turn so far failed) simply contribute nothing.
        neighbor_posts = [
            frozen_posts[n.agent_id] for n in neighbors if n.agent_id in frozen_posts
        ]

        emphasize_format = False
        for attempt in range(MAX_STANCE_ATTEMPTS):
            built = self._prompt_builder.build_discussion_prompt(
                agent, neighbor_posts, turn, emphasize_format=emphasize_format
            )
            try:
                response = await self._gateway.generate(
                    built.text, image=built.image, temperature=self._run.temperature
                )
                stance_after, reason = parse_stance(response.text)
                stance_after = clamp_and_validate_scale(
                    stance_after, self._run.stance_scale
                )
            except StanceParseFailure:
                # Fix I: re-ask with a louder format block. Nothing else about
                # the prompt changes, so a retry stays a response to the same
                # stimulus (Q4.6).
                emphasize_format = True
                continue
            except FatalGatewayError:
                return self._failed_interaction(agent, neighbors, turn, stance_before)

            return Interaction(
                interaction_id=str(uuid.uuid4()),
                run_id=self._run.run_id,
                turn=turn,
                speaker_agent_id=agent.agent_id,
                neighbor_agent_ids=[n.agent_id for n in neighbors],
                stance_before=stance_before,
                stance_after=stance_after,
                reason_text=reason,
                content_type="generated_text",
                meme_id=None,
                used_vision_fallback=built.used_vision_fallback,
                prompt_token_count=response.prompt_tokens,
                completion_token_count=response.completion_tokens,
                model_backend_id=self._run.model_backend_id,
                latency_ms=response.latency_ms,
                api_call_status="success" if attempt == 0 else "retried_success",
                timestamp_utc=_now_utc_iso(),
            )

        # Retry budget exhausted. Fix I: log and exclude, never fabricate a
        # placeholder stance. stance_after repeats stance_before, and because
        # no StanceRecord is appended, interaction_engine's _current_stance
        # keeps reading this agent's last known value on later turns.
        return self._failed_interaction(agent, neighbors, turn, stance_before)

    def _build_meme_interaction(
        self,
        agent: Agent,
        neighbors: list[Agent],
        meme: MemeContent,
        turn: int,
        stance_before: float,
    ) -> Interaction:
        """A meme turn costs no LLM call: the dataset label is the stance.

        used_vision_fallback is False here by definition. It describes what a
        *receiving* speaker saw of someone else's meme, and this agent's own
        posting turn never builds a prompt at all.
        """
        return Interaction(
            interaction_id=str(uuid.uuid4()),
            run_id=self._run.run_id,
            turn=turn,
            speaker_agent_id=agent.agent_id,
            neighbor_agent_ids=[n.agent_id for n in neighbors],
            stance_before=stance_before,
            stance_after=meme.stance_label,
            reason_text=meme.caption_text,
            content_type="meme",
            meme_id=meme.meme_id,
            used_vision_fallback=False,
            prompt_token_count=0,
            completion_token_count=0,
            model_backend_id=self._run.model_backend_id,
            latency_ms=0,
            api_call_status="success",
            timestamp_utc=_now_utc_iso(),
        )

    def _failed_interaction(
        self, agent: Agent, neighbors: list[Agent], turn: int, stance_before: float
    ) -> Interaction:
        return Interaction(
            interaction_id=str(uuid.uuid4()),
            run_id=self._run.run_id,
            turn=turn,
            speaker_agent_id=agent.agent_id,
            neighbor_agent_ids=[n.agent_id for n in neighbors],
            stance_before=stance_before,
            stance_after=stance_before,
            reason_text="",
            content_type="generated_text",
            meme_id=None,
            used_vision_fallback=False,
            prompt_token_count=0,
            completion_token_count=0,
            model_backend_id=self._run.model_backend_id,
            latency_ms=0,
            api_call_status="failed_logged_null",
            timestamp_utc=_now_utc_iso(),
        )

    # --- resume and lifecycle -------------------------------------------

    def _rebuild_last_posts(self) -> dict[str, Interaction]:
        """Reconstructs each agent's most recent post after a resume.

        CheckpointState carries only agent_snapshot, and StanceRecord has no
        content_type or meme_id, so agent state alone cannot say whether a
        past post was a meme. Without this, a resumed run would render every
        previous meme as though it had been ordinary text, quietly changing
        the stimulus mid-run. interactions.jsonl is append-only and complete
        through last_completed_turn, so it is the right source. Not a schema
        change; phase4.md Section 2 did not anticipate this.
        """
        path = self._run_dir / "interactions.jsonl"
        if not path.exists():
            return {}
        last: dict[str, Interaction] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            interaction = Interaction.model_validate_json(line)
            previous = last.get(interaction.speaker_agent_id)
            if previous is None or interaction.turn >= previous.turn:
                last[interaction.speaker_agent_id] = interaction
        return last

    def _save_checkpoint(self, turn: int) -> None:
        self._checkpoint_mgr.save(
            self._run.run_id, turn, self._agent_manager.snapshot()
        )

    def _finalize(self, status: str) -> None:
        """Re-persists run_config.json with the terminal lifecycle fields.

        Sec 4.5 requires this, but nothing in the codebase read or wrote
        status / started_at_utc / completed_at_utc / total_cost_usd before
        now. This is the one record that is mutated and rewritten rather than
        appended.
        """
        self._run.status = status
        self._run.completed_at_utc = _now_utc_iso()
        if self._cost_tracker is not None:
            self._run.total_cost_usd = self._cost_tracker.total_usd
        self._logger.write_run_config(self._run)


def build_orchestrator(
    run: ExperimentRun,
    gateway: ModelGateway,
    cost_tracker: CostTracker | None = None,
    runs_dir: Path = Path("runs"),
) -> SimulationOrchestrator:
    """Wires a full orchestrator from a validated ExperimentRun.

    Exists so the SeedManager routing lives in exactly one place. Each stream
    is deliberately offset from the others, and handing a consumer the wrong
    one produces a run that still completes and is still deterministic but is
    no longer the run the config describes, which is close to undetectable
    after the fact.
    """
    seeds = SeedManager(run.seed)
    meme_pool = MemePoolManager(run.meme_injection, seeds.meme_injection_rng)
    return SimulationOrchestrator(
        run=run,
        agent_manager=AgentManager(run, seeds.persona_assignment_rng),
        interaction_engine=AlphaSampling(run.alpha, run.N),
        meme_pool=meme_pool,
        gateway=gateway,
        logger=LoggingWriter(runs_dir / run.run_id),
        checkpoint_mgr=CheckpointManager(runs_dir),
        prompt_builder=PromptBuilder(
            run, gateway.supports_vision, meme_lookup=meme_pool.get_meme
        ),
        neighbor_rng=seeds.neighbor_sampling_rng,
        cost_tracker=cost_tracker,
        runs_dir=runs_dir,
    )


def _current_stance_for_logging(agent: Agent, turn: int) -> float:
    """Mirrors interaction_engine._current_stance exactly.

    Deliberately duplicated rather than imported: if these two ever disagree,
    stance_before in the log would not be the value neighbour sampling
    actually used, and every regression fitted on that column would be wrong.
    Keeping the logic visible here makes that coupling explicit.
    """
    if turn == 1 or not agent.stance_history:
        return agent.initial_stance
    return agent.stance_history[-1].stance_value


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
