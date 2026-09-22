"""Caption-only prompt substitution for text-only backends (full_design_doc.md Sec 3.6).

Substitutes a caption-only prompt when a meme must be shown to a
text-only-backend receiver. This is the ONLY place in the codebase that
constructs a meme-derived prompt: the instruction framing is identical
regardless of vision support, only the image payload differs.
"""

from __future__ import annotations

from pathlib import Path

from sandbox.models import MemeContent

_INSTRUCTION = (
    "You are shown the following content from another participant "
    "in the discussion. Consider it as their contribution and respond "
    "according to your persona and current view."
)


def build_meme_prompt(meme: MemeContent, receiver_supports_vision: bool) -> tuple[str, bytes | None]:
    """
    Returns (prompt_text, image_bytes_or_None). If receiver_supports_vision
    is False, image is always None and the caption is woven into the
    prompt text -- the filesystem is never touched in that branch, since a
    text-only receiver never needs the image bytes at all.
    """
    if receiver_supports_vision:
        image_bytes = _load_image(meme.image_path)
        prompt_text = f"{_INSTRUCTION}\n\n[Image shown separately]\nCaption: {meme.caption_text}"
        return prompt_text, image_bytes
    else:
        prompt_text = f"{_INSTRUCTION}\n\nContent (text-only, as no image is available to you): {meme.caption_text}"
        return prompt_text, None


def _load_image(image_path: str) -> bytes:
    """Raises FileNotFoundError (uncaught) on a missing file -- treated as
    a data-integrity problem with the meme pool, not something to silently
    degrade around."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"meme image missing on disk: {image_path}")
    return path.read_bytes()
