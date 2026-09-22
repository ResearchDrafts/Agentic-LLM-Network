from pathlib import Path

import pytest

from sandbox.models import MemeContent
from sandbox.vision_fallback import build_meme_prompt

_REAL_IMAGE_PATH = "data/memes/images/test_001.jpg"


def _make_meme(image_path: str = _REAL_IMAGE_PATH) -> MemeContent:
    return MemeContent(
        meme_id="meme_test_001",
        image_path=image_path,
        caption_text="When will people learn??",
        stance_label=6.5,
        offensiveness_label=1.0,
        source_dataset="test_fixture_v1",
    )


def test_vision_capable_receiver_gets_prompt_and_matching_image_bytes():
    meme = _make_meme()

    prompt_text, image_bytes = build_meme_prompt(meme, receiver_supports_vision=True)

    assert image_bytes == Path(_REAL_IMAGE_PATH).read_bytes()
    assert meme.caption_text in prompt_text


def test_text_only_receiver_gets_no_image_and_caption_woven_into_prompt():
    meme = _make_meme()

    prompt_text, image_bytes = build_meme_prompt(meme, receiver_supports_vision=False)

    assert image_bytes is None
    assert meme.caption_text in prompt_text


def test_missing_image_file_with_vision_receiver_raises_file_not_found():
    meme = _make_meme(image_path="data/memes/images/does_not_exist.jpg")

    with pytest.raises(FileNotFoundError):
        build_meme_prompt(meme, receiver_supports_vision=True)


def test_text_only_path_never_touches_filesystem_for_a_nonexistent_image():
    meme = _make_meme(image_path="data/memes/images/does_not_exist.jpg")

    # Must succeed even though the image path is bogus -- the caption-only
    # branch must never call _load_image.
    prompt_text, image_bytes = build_meme_prompt(meme, receiver_supports_vision=False)

    assert image_bytes is None
    assert meme.caption_text in prompt_text
