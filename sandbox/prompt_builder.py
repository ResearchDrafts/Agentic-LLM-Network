"""Prompt construction (Tier 3).

This module has no counterpart in full_design_doc.md or sandbox_hld.md. Both
reference a PromptBuilder from the Orchestrator's sample code, but neither
specifies its template, its inputs, or how persona / language_condition /
memory / neighbour posts get woven into text. The design is resolved in
phase4.md Section 1 as Q4.1 through Q4.10; the load-bearing decisions are
restated here at their point of use rather than left in the planning doc.

Two constraints dominate everything below:

1. RQ1 compares polarization dynamics across language conditions, so the
   prompt must be identical across those conditions except for one directive
   line. Fix A (sandbox_hld.md:390) counts memory in turns rather than tokens
   precisely to stop one language silently receiving less context; a builder
   that made Hinglish prompts systematically longer would reintroduce that
   confound one level up.

2. The output must satisfy stance_parser's regex, and parse_stance strips
   EVERY "STANCE: <n>" occurrence from reason_text. A model that restates its
   stance mid-sentence therefore has that fragment silently deleted from the
   field RQ2's coherence and code-mix analyses consume, so the format
   instruction insists on exactly one STANCE line, first.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sandbox.models import Agent, ExperimentRun, Interaction, MemeContent
from sandbox.vision_fallback import build_meme_prompt

# Fix A's window. Memory is re-sliced here even though AgentManager already
# caps memory_window and Agent validates assignment, because neither guards
# in-place .append() (phase4.md D7). This is the cap that reaches the model,
# so it is the one that determines experimental results.
MEMORY_WINDOW_TURNS = 5

# phase4.md Q4.2. Exactly one of these lines varies across a language sweep;
# everything else in the prompt is byte-identical. Keep them short and
# parallel: any growth here is a direct, uncontrolled prompt-length
# difference between the conditions RQ1 compares.
LANGUAGE_DIRECTIVES = {
    "english": "Write your reply in English.",
    "hinglish": "Write your reply in Hinglish (Hindi-English code-mixed, Roman script).",
    "hindi": "Write your reply in Hindi (Devanagari script).",
}


@dataclass
class BuiltPrompt:
    """What the Orchestrator hands to ModelGateway.generate().

    used_vision_fallback is wider than sandbox_hld.md:285's original meaning
    (a text-only backend receiving the caption instead of the image). Here it
    is True whenever any meme in this prompt was rendered caption-only for any
    reason, including the second-and-later memes in a turn that a single image
    slot cannot carry. The field answers "did this agent see less than the
    full meme content it was shown", which is what any analysis using it
    actually needs to know. Signed off in phase4.md Q4.8.
    """

    text: str
    image: bytes | None
    used_vision_fallback: bool


class PromptBuilder:
    def __init__(
        self,
        run: ExperimentRun,
        supports_vision: bool,
        meme_lookup: Callable[[str], MemeContent],
    ):
        if run.language_condition not in LANGUAGE_DIRECTIVES:
            # Unreachable through ExperimentRun's own regex, kept as
            # defence in depth: silently falling back to English would make a
            # misconfigured run look like a valid English one in the results.
            raise ValueError(f"no language directive for {run.language_condition!r}")
        self._run = run
        self._supports_vision = supports_vision
        self._meme_lookup = meme_lookup
        self._low = min(run.stance_scale)
        self._high = max(run.stance_scale)

    def build_discussion_prompt(
        self,
        agent: Agent,
        neighbor_posts: list[Interaction],
        turn: int,
        *,
        emphasize_format: bool = False,
    ) -> BuiltPrompt:
        """Builds one speaker's prompt for one turn.

        neighbor_posts is list[Interaction], not list[Agent]: only Interaction
        carries content_type and meme_id, without which the meme path cannot
        be rendered at all. This resolves a direct contradiction between
        full_design_doc.md:754 and sandbox_hld.md:230 in favour of the HLD
        (phase4.md Q4.1).

        Callers pass the frozen turn-start snapshot. This method never reads
        any turn-t output, which is what makes Fix F hold.
        """
        image, fallback_used = self._resolve_meme_media(neighbor_posts)

        sections = [
            f"You are: {agent.persona}",
            f"You are taking part in an ongoing discussion about: {self._run.topic}",
            self._position_section(agent),
        ]
        memory = self._memory_section(agent)
        if memory:
            sections.append(memory)
        neighbours = self._neighbor_section(neighbor_posts, turn)
        if neighbours:
            sections.append(neighbours)
        sections.append(LANGUAGE_DIRECTIVES[self._run.language_condition])
        sections.append(self._format_section(emphasize_format))

        return BuiltPrompt(
            text="\n\n".join(sections),
            image=image,
            used_vision_fallback=fallback_used,
        )

    # --- sections --------------------------------------------------------

    def _position_section(self, agent: Agent) -> str:
        # Agent.current_stance() rather than interaction_engine's turn-aware
        # _current_stance: the two differ only when stance_history is
        # non-empty at turn 1, which Fix F's ordering makes impossible.
        return (
            f"Your current position is {_fmt(agent.current_stance())} on a scale "
            f"from {_fmt(self._low)} to {_fmt(self._high)}, where "
            f"{_fmt(self._low)} means {self._run.stance_low_label} and "
            f"{_fmt(self._high)} means {self._run.stance_high_label}."
        )

    def _memory_section(self, agent: Agent) -> str:
        """The agent's own recent reasoning, from stance_history.

        Per plan.md:43-44, memory_window is a redundant cache of these same
        interaction_ids, so it is not read here. The [-N:] slice is Fix A's
        last line of defence and must not be removed (phase4.md Q4.3).
        """
        recent = agent.stance_history[-MEMORY_WINDOW_TURNS:]
        if not recent:
            return ""
        lines = [
            f"  Turn {record.turn} (position {_fmt(record.stance_value)}): {record.reason_text}"
            for record in recent
        ]
        return "What you said recently:\n" + "\n".join(lines)

    def _neighbor_section(self, neighbor_posts: list[Interaction], turn: int) -> str:
        """Renders the sampled neighbours' most recent posts.

        Omitted entirely on turn 1, when nobody has posted yet (phase4.md
        Q4.9). Showing bare initial_stance numbers with no reasoning would be
        a stimulus with no analogue in any later turn.
        """
        if turn == 1 or not neighbor_posts:
            return ""

        lines = []
        rendered_image_for = self._first_meme_speaker(neighbor_posts)
        for post in neighbor_posts:
            header = f"  {post.speaker_agent_id} (position {_fmt(post.stance_after)}):"
            if post.content_type == "meme":
                meme = self._meme_lookup(post.meme_id)
                # Only the neighbour whose image is actually attached gets the
                # multimodal framing; every other meme is caption-only.
                as_image = (
                    self._supports_vision and post.speaker_agent_id == rendered_image_for
                )
                body, _ = build_meme_prompt(meme, receiver_supports_vision=as_image)
                lines.append(f"{header}\n{_indent(body)}")
            else:
                lines.append(f"{header} {post.reason_text}")
        return "What others in your feed posted most recently:\n" + "\n".join(lines)

    def _format_section(self, emphasize_format: bool) -> str:
        """The output contract stance_parser depends on.

        emphasize_format is set only on retries 2 and 3, after a
        StanceParseFailure. It must change nothing except this block:
        altering persona, topic, memory or neighbour content on a retry would
        mean the logged Interaction is a response to a different stimulus
        than a first-attempt one, silently mixing two conditions in one
        dataset (phase4.md Q4.6).
        """
        contract = (
            "Reply in exactly this format. The STANCE line must come first, "
            "and the number must appear on that line only:\n"
            f"STANCE: <a number from {_fmt(self._low)} to {_fmt(self._high)}>\n"
            "<your reasoning, two or three sentences>"
        )
        if not emphasize_format:
            return contract
        return (
            "IMPORTANT: your previous reply could not be read. Follow the "
            "format below exactly. Begin with the literal word STANCE, a "
            "colon, and a single number. Do not repeat the number anywhere "
            "else in your reply.\n\n" + contract
        )

    # --- meme media ------------------------------------------------------

    def _resolve_meme_media(
        self, neighbor_posts: list[Interaction]
    ) -> tuple[bytes | None, bool]:
        """Decides which single image (if any) is attached to this prompt.

        ModelGateway.generate() takes one `image`, but N neighbours may
        include several memes. At most one image is attached: the first
        meme-bearing neighbour in neighbour order, which AlphaSampling
        produced from a seeded RNG and is therefore reproducible. Every other
        meme degrades to its caption (phase4.md Q4.8).
        """
        meme_posts = [p for p in neighbor_posts if p.content_type == "meme"]
        if not meme_posts:
            return None, False

        if not self._supports_vision:
            # Text-only backend: every meme is caption-only, the original
            # sandbox_hld.md:285 case.
            return None, True

        first = self._meme_lookup(meme_posts[0].meme_id)
        _, image = build_meme_prompt(first, receiver_supports_vision=True)
        # True when a second meme had to degrade despite a vision backend.
        return image, len(meme_posts) > 1

    @staticmethod
    def _first_meme_speaker(neighbor_posts: list[Interaction]) -> str | None:
        for post in neighbor_posts:
            if post.content_type == "meme":
                return post.speaker_agent_id
        return None


def _fmt(value: float) -> str:
    """Renders 4.0 as "4" but leaves 6.5 alone.

    Stance values are whole numbers when drawn by AgentManager but may be
    fractional when a model returns one (clamp_and_validate_scale checks only
    the endpoints), so both forms have to read naturally in a prompt.
    """
    return str(int(value)) if float(value).is_integer() else str(value)


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line if line else line for line in text.split("\n"))
