"""Tests for prompt_builder.py.

Organised around phase4.md Section 1's numbered decisions. The RQ1 parity
test and the memory-cap test are the two that protect research validity
rather than merely asserting behaviour.
"""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from sandbox.meme_pool_manager import MemePoolManager
from sandbox.models import MemeInjectionConfig, StanceRecord
from sandbox.prompt_builder import (
    LANGUAGE_DIRECTIVES,
    MEMORY_WINDOW_TURNS,
    PromptBuilder,
)
from sandbox.stance_parser import clamp_and_validate_scale, parse_stance
from tests.factories import STANCE_SCALE, make_agent, make_interaction, make_run


@pytest.fixture
def meme_lookup():
    pool = MemePoolManager(
        MemeInjectionConfig(enabled=True, meme_pool_id="test_pool", injection_rate=1.0),
        random.Random(0),
    )
    return pool.get_meme


def _builder(meme_lookup, *, supports_vision=True, **run_overrides) -> PromptBuilder:
    return PromptBuilder(
        make_run(**run_overrides), supports_vision=supports_vision, meme_lookup=meme_lookup
    )


def _history(n: int) -> list[StanceRecord]:
    return [
        StanceRecord(
            turn=i + 1, stance_value=4.0, reason_text=f"reason {i}", interaction_id=f"i{i}"
        )
        for i in range(n)
    ]


def _text_post(agent_id: str, stance: float = 5.0, reason: str = "a neighbour view"):
    return make_interaction(agent_id, turn=2, stance_after=stance, reason_text=reason)


def _meme_post(agent_id: str, meme_id: str = "meme_test_001", stance: float = 6.5):
    return make_interaction(
        agent_id, turn=2, stance_after=stance, content_type="meme", meme_id=meme_id
    )


# --- Q4.2: language is the only thing that varies (the RQ1 guard) --------


def test_all_three_language_conditions_render(meme_lookup):
    for condition in LANGUAGE_DIRECTIVES:
        built = _builder(meme_lookup, language_condition=condition).build_discussion_prompt(
            make_agent(), [], turn=1
        )
        assert LANGUAGE_DIRECTIVES[condition] in built.text


def test_prompts_differ_across_languages_only_in_the_directive_line(meme_lookup):
    """The RQ1 validity guard.

    Fix A counts memory in turns rather than tokens so no language silently
    receives less context. That protection is worthless if the prompt itself
    differs between conditions, so every line except the directive must be
    byte-identical.
    """
    agent = make_agent(stance_history=_history(3))
    posts = [_text_post("agent_0011"), _text_post("agent_0077", stance=2.0)]

    rendered = {
        condition: _builder(meme_lookup, language_condition=condition)
        .build_discussion_prompt(agent, posts, turn=3)
        .text
        for condition in LANGUAGE_DIRECTIVES
    }

    stripped = {
        condition: [
            line for line in text.split("\n") if line != LANGUAGE_DIRECTIVES[condition]
        ]
        for condition, text in rendered.items()
    }
    baseline = stripped["english"]
    for condition, lines in stripped.items():
        assert lines == baseline, f"{condition} differs from english beyond its directive"


def test_language_directive_length_spread_stays_bounded():
    """The residual asymmetry, measured rather than assumed.

    "Hinglish" cannot be named in as few characters as "English", so exact
    parity is impossible and the prompts do differ in length by a fixed
    amount (42 characters at time of writing). This bounds that constant. If
    it fails, someone has grown a directive into a paragraph and turned a
    rounding error into a real length confound between RQ1's conditions.
    """
    lengths = {c: len(d) for c, d in LANGUAGE_DIRECTIVES.items()}
    spread = max(lengths.values()) - min(lengths.values())
    assert spread <= 60, lengths


def test_language_length_difference_is_negligible_on_a_realistic_prompt(meme_lookup):
    """The same 42 characters, expressed as a share of a full-size prompt.

    Deliberately built at the scale a real run produces (a full 5-turn memory
    window and N=5 neighbour posts) rather than a bare minimum, because the
    fixed offset looks alarming against a toy prompt and negligible against a
    real one, and it is the real one that determines whether RQ1's comparison
    is clean.
    """
    agent = make_agent(stance_history=_history(MEMORY_WINDOW_TURNS))
    posts = [
        _text_post(f"agent_{i:04d}", reason="A reasonably typical neighbour post.")
        for i in range(5)
    ]
    lengths = {
        condition: len(
            _builder(
                meme_lookup,
                language_condition=condition,
                topic="whether remote work should be the default",
            )
            .build_discussion_prompt(agent, posts, turn=6)
            .text
        )
        for condition in LANGUAGE_DIRECTIVES
    }
    spread = max(lengths.values()) - min(lengths.values())
    assert spread / min(lengths.values()) < 0.05, lengths


def test_unknown_language_condition_raises_rather_than_defaulting(meme_lookup):
    run = make_run()
    object.__setattr__(run, "__dict__", {**run.__dict__, "language_condition": "french"})
    with pytest.raises(ValueError, match="no language directive"):
        PromptBuilder(run, supports_vision=True, meme_lookup=meme_lookup)


# --- Q4.5: the output contract stance_parser depends on ------------------


def test_a_reply_following_the_template_round_trips_through_the_parser(meme_lookup):
    """Closes the loop: the format this prompt asks for is the format the
    parser accepts. Asserting the prompt's wording alone would not catch a
    drift between the two."""
    built = _builder(meme_lookup).build_discussion_prompt(make_agent(), [], turn=1)
    assert "STANCE: <a number from 1 to 7>" in built.text

    reply = "STANCE: 6\nThe commuting argument moved me."
    value, reason = parse_stance(reply)
    assert clamp_and_validate_scale(value, STANCE_SCALE) == 6.0
    assert reason == "The commuting argument moved me."


def test_prompt_instructs_the_stance_number_to_appear_only_once(meme_lookup):
    """parse_stance strips EVERY STANCE occurrence from reason_text, so a
    restated stance silently vanishes from RQ2's input."""
    built = _builder(meme_lookup).build_discussion_prompt(make_agent(), [], turn=1)
    assert "that line only" in built.text


def test_scale_bounds_come_from_the_configured_stance_scale(meme_lookup):
    built = _builder(meme_lookup, stance_scale=[0, 1, 2, 3]).build_discussion_prompt(
        make_agent(), [], turn=1
    )
    assert "from 0 to 3" in built.text


def test_stance_anchor_labels_are_rendered(meme_lookup):
    built = _builder(
        meme_lookup,
        stance_low_label="remote work should never be default",
        stance_high_label="remote work should always be default",
    ).build_discussion_prompt(make_agent(), [], turn=1)
    assert "remote work should never be default" in built.text
    assert "remote work should always be default" in built.text


# --- Q4.3: Fix A's memory window -----------------------------------------


def test_memory_renders_at_most_five_turns(meme_lookup):
    agent = make_agent(stance_history=_history(12))
    built = _builder(meme_lookup).build_discussion_prompt(agent, [], turn=13)
    assert built.text.count("  Turn ") == MEMORY_WINDOW_TURNS


def test_memory_slice_holds_even_when_the_cap_was_bypassed(meme_lookup):
    """Fix A's last line of defence.

    validate_assignment cannot see an in-place append (phase4.md D7), so an
    agent can carry more history than the cap allows. The prompt is what
    reaches the model, so the cap must hold here regardless.
    """
    agent = make_agent(stance_history=_history(3))
    for extra in _history(20):
        agent.stance_history.append(extra)
    assert len(agent.stance_history) > MEMORY_WINDOW_TURNS

    built = _builder(meme_lookup).build_discussion_prompt(agent, [], turn=24)
    assert built.text.count("  Turn ") == MEMORY_WINDOW_TURNS


def test_memory_renders_fewer_entries_when_history_is_short(meme_lookup):
    agent = make_agent(stance_history=_history(2))
    built = _builder(meme_lookup).build_discussion_prompt(agent, [], turn=3)
    assert built.text.count("  Turn ") == 2


def test_memory_section_omitted_entirely_when_history_is_empty(meme_lookup):
    built = _builder(meme_lookup).build_discussion_prompt(make_agent(), [], turn=1)
    assert "What you said recently" not in built.text


# --- Q4.9: turn 1 ---------------------------------------------------------


def test_turn_one_omits_the_neighbour_block(meme_lookup):
    built = _builder(meme_lookup).build_discussion_prompt(
        make_agent(), [_text_post("agent_0011")], turn=1
    )
    assert "What others in your feed" not in built.text


def test_later_turns_render_neighbour_posts(meme_lookup):
    built = _builder(meme_lookup).build_discussion_prompt(
        make_agent(), [_text_post("agent_0011", reason="commuting is wasted life")], turn=2
    )
    assert "What others in your feed" in built.text
    assert "commuting is wasted life" in built.text


# --- Q4.7 / Q4.8: memes, vision, and the single image slot ---------------


def test_meme_neighbour_on_a_vision_backend_attaches_the_image(meme_lookup):
    built = _builder(meme_lookup, supports_vision=True).build_discussion_prompt(
        make_agent(), [_meme_post("agent_0077")], turn=2
    )
    assert built.image is not None
    assert built.used_vision_fallback is False
    assert "When will people learn??" in built.text


def test_meme_neighbour_on_a_text_only_backend_falls_back_to_caption(meme_lookup):
    built = _builder(meme_lookup, supports_vision=False).build_discussion_prompt(
        make_agent(), [_meme_post("agent_0077")], turn=2
    )
    assert built.image is None
    assert built.used_vision_fallback is True
    assert "When will people learn??" in built.text


def test_text_only_backend_never_reads_an_image_from_disk(meme_lookup):
    """vision_fallback documents that the filesystem is untouched on the
    caption branch. A text-only run must not depend on meme images existing."""
    with patch("sandbox.vision_fallback._load_image", side_effect=AssertionError("read disk")):
        built = _builder(meme_lookup, supports_vision=False).build_discussion_prompt(
            make_agent(), [_meme_post("agent_0077")], turn=2
        )
    assert built.image is None


def test_two_meme_neighbours_attach_exactly_one_image_and_flag_the_downgrade(meme_lookup):
    """ModelGateway.generate takes a single image, so the second meme degrades
    to its caption even on a vision backend. used_vision_fallback reports
    that, per Q4.8's widened definition."""
    built = _builder(meme_lookup, supports_vision=True).build_discussion_prompt(
        make_agent(),
        [_meme_post("agent_0011", "meme_test_001"), _meme_post("agent_0077", "meme_test_002")],
        turn=2,
    )
    assert built.image is not None
    assert built.used_vision_fallback is True
    assert "When will people learn??" in built.text
    assert "This is fine." in built.text


def test_no_memes_means_no_image_and_no_fallback(meme_lookup):
    built = _builder(meme_lookup).build_discussion_prompt(
        make_agent(), [_text_post("agent_0011")], turn=2
    )
    assert built.image is None
    assert built.used_vision_fallback is False


def test_meme_framing_comes_from_vision_fallback_not_a_local_copy(meme_lookup):
    """vision_fallback claims to be the only place constructing meme-derived
    prompts. If prompt_builder grew its own copy, this drifts silently."""
    from sandbox.vision_fallback import _INSTRUCTION

    built = _builder(meme_lookup).build_discussion_prompt(
        make_agent(), [_meme_post("agent_0077")], turn=2
    )
    assert _INSTRUCTION.split(".")[0] in built.text


def test_unknown_meme_id_raises(meme_lookup):
    with pytest.raises(KeyError):
        _builder(meme_lookup).build_discussion_prompt(
            make_agent(), [_meme_post("agent_0077", "meme_does_not_exist")], turn=2
        )


# --- Q4.6: emphasize_format ----------------------------------------------


def test_emphasize_format_changes_only_the_format_block(meme_lookup):
    """A retry must be a response to the same stimulus. If emphasize_format
    altered persona, topic, memory or neighbour content, first attempts and
    retries would be two different conditions in one dataset."""
    builder = _builder(meme_lookup)
    agent = make_agent(stance_history=_history(3))
    posts = [_text_post("agent_0011")]

    plain = builder.build_discussion_prompt(agent, posts, turn=3).text
    emphasised = builder.build_discussion_prompt(
        agent, posts, turn=3, emphasize_format=True
    ).text

    assert emphasised != plain
    # Split on the language directive, the last thing before the format
    # block. Everything ahead of it is the experimental stimulus and must be
    # identical; only what follows may change.
    directive = LANGUAGE_DIRECTIVES["english"]
    assert plain.split(directive)[0] == emphasised.split(directive)[0]
    assert "could not be read" in emphasised
    assert "could not be read" not in plain


def test_emphasize_format_still_satisfies_the_parser_contract(meme_lookup):
    built = _builder(meme_lookup).build_discussion_prompt(
        make_agent(), [], turn=1, emphasize_format=True
    )
    assert "STANCE: <a number from 1 to 7>" in built.text


# --- purity ---------------------------------------------------------------


def test_builder_does_not_mutate_its_arguments(meme_lookup):
    agent = make_agent(stance_history=_history(7))
    posts = [_text_post("agent_0011"), _meme_post("agent_0077")]
    before_history = list(agent.stance_history)
    before_memory = list(agent.memory_window)
    before_posts = list(posts)

    _builder(meme_lookup).build_discussion_prompt(agent, posts, turn=8)

    assert agent.stance_history == before_history
    assert agent.memory_window == before_memory
    assert posts == before_posts


def test_stance_values_render_without_trailing_zeroes(meme_lookup):
    """4.0 should read as "4"; 6.5 must stay "6.5"."""
    built = _builder(meme_lookup).build_discussion_prompt(
        make_agent(initial_stance=4.0), [_text_post("agent_0011", stance=6.5)], turn=2
    )
    assert "position is 4 on a scale" in built.text
    assert "(position 6.5)" in built.text
