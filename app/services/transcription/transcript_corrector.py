"""Stage 2 — LLM-based phonetic correction & alignment.

Receives the raw Whisper transcript (which may contain phonetic misspellings of
brand names, cocktail terms, and guest names) and corrects it using three
contextual sources:

  1. Strategy context  — the live strategy.json definitions (brands, formats,
                          positioning) loaded at runtime, so new content is
                          handled without any code changes.
  2. Visual clues      — GPT-4o's description of the actual bottle labels,
                          on-screen text, and bar tools visible in the video
                          frames.  These are the ground truth for spelling.
  3. Domain expertise  — a system prompt encoding beverage-industry conventions.

This approach is intentionally static-list-free: as long as a brand appears in
strategy.json or is visible on screen, it will be corrected automatically.
"""

import logging

from app.services.openai.openai_client import chat_text
from app.services.strategy.strategy_knowledge import get_strategy_prompt_context

logger = logging.getLogger(__name__)

_CORRECTION_SYSTEM_PROMPT = """\
You are an expert audio transcription editor specialising in the premium \
beverage, hospitality, and mixology industries.

Your task is to take a raw, phonetically imperfect speech-to-text transcript \
and return a corrected, publication-ready version.

Rules:
1. Correct phonetic spelling errors for brand names, guest names, \
   cocktail ingredients, venues, and industry terms using the provided \
   "Strategy Context" and "Visual Clues" sections below.
   Examples: "remy count row" → "Rémy Cointreau", "fords jin" → "Fords Gin", \
   "tom ask" → "Tom Aske".
2. Do NOT hallucinate content. If a word is genuinely unintelligible and there \
   are no context clues, leave it as-is or mark it [inaudible].
3. Preserve the speaker's natural tone, register, and sentence structure — \
   do NOT summarise, paraphrase, or re-order speech.
4. Use "Visual Clues" (bottle labels, on-screen text, bar tools) as the \
   highest-authority spelling source; they override phonetic guesses.
5. Return ONLY the corrected transcript text — no preamble, no explanation, \
   no meta-commentary.\
"""


def align_and_correct_transcript(
    raw_transcript: str,
    visual_description: str,
    strategy_context: str | None = None,
) -> tuple[str, dict]:
    """Correct phonetic errors in a raw Whisper transcript.

    Args:
        raw_transcript:    Output of Stage 1 (transcribe_raw_audio).
        visual_description: GPT-4o frame analysis describing visible labels,
                            bottles, tools, and on-screen text.
        strategy_context:  Formatted strategy string from get_strategy_prompt_context().
                           If None it is fetched automatically.

    Returns:
        (corrected_transcript, usage_dict) where usage_dict has keys:
            prompt_tokens, completion_tokens, cached_tokens.
        Falls back to (raw_transcript, zero_usage) if the LLM call fails.
    """
    _ZERO_USAGE: dict = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    if not raw_transcript.strip():
        return "", _ZERO_USAGE

    if strategy_context is None:
        strategy_context = get_strategy_prompt_context()

    user_payload = f"""\
[RAW TRANSCRIPT TO CORRECT]
{raw_transcript}

[STRATEGY CONTEXT — brands, formats, positioning defined in strategy.json]
{strategy_context}

[VISUAL CLUES — items visible in video frames; highest-authority spelling source]
{visual_description if visual_description.strip() else "(no visual description available)"}
"""

    try:
        corrected, usage = chat_text(system=_CORRECTION_SYSTEM_PROMPT, user=user_payload)
        corrected = corrected.strip()
        if not corrected:
            logger.warning("Correction layer returned empty string — falling back to raw transcript.")
            return raw_transcript, _ZERO_USAGE
        logger.info(
            "Transcript corrected: %d chars raw → %d chars corrected.",
            len(raw_transcript),
            len(corrected),
        )
        return corrected, usage
    except Exception as exc:
        # Stage 2 failure is non-fatal: degrade gracefully to the raw transcript.
        logger.warning(
            "Transcript correction failed (%s) — using raw Whisper output.", str(exc)
        )
        return raw_transcript, _ZERO_USAGE
