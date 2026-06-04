import logging
import re

from app.models.schemas import (
    Caption,
    CarouselSlides,
    Classification,
    ContentAnalysis,
    GeneratedAssets,
    PublishingRecommendation,
    ReelScript,
    StoryIdea,
)
from app.services.openai.openai_client import chat_json
from app.services.strategy.strategy_knowledge import get_strategy_prompt_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for asset generation
# Kept as a module constant so the recipe constraint is defined once and can
# be audited / updated without touching function logic.
# ---------------------------------------------------------------------------
_GENERATION_SYSTEM_PROMPT = """\
You are the PM Drinks Content Generator for Pritesh Mody (@pritesh.mody), \
a UK cocktail authority.

You repurpose high-fidelity, corrected transcripts into structured Instagram assets.

STRICT RECIPE RULES — read before generating anything:
1. ONLY populate the 'recipe' field if an explicit cocktail recipe with \
   measurements (ml, oz, cl, dashes, parts) is present verbatim in the \
   Transcript or Visual Description provided in the user message.
2. If the speaker is telling an anecdote, reviewing a venue, or discussing \
   collaborations without naming ingredient quantities, set 'recipe' to an \
   empty string. Do NOT fabricate or infer proportions from brand context alone.
3. When 'recipe' is empty, focus carousel, captions, and stories entirely on \
   the storytelling, hospitality insight, or celebrity angle.
4. Never invent a cocktail name that does not appear in the source material.\
"""


def _divergence_block(analysis: ContentAnalysis) -> str:
    if not analysis.audio_visual_divergence:
        return ""
    return f"""
AUDIO / VISUAL DIVERGENCE DETECTED — follow strictly:
- Audio (dialogue): {analysis.audio_focus_summary}
- On-screen (visual): {analysis.visual_focus_summary}
- Repurposing guidance: {analysis.repurposing_guidance}
- Use spoken dialogue for story, hooks, names, and scheduling context where relevant.
- Use on-screen visuals ONLY for recipe, pour technique, and ingredients actually shown.
- NEVER invent a cocktail name, recipe, or measures not present in the transcript or visual description.
- If recipe intent is true but measures are not evidenced, set caption.recipe to an empty string and note gaps in caption.story instead."""


def generate_assets(
    analysis: ContentAnalysis,
    classification: Classification,
    filename: str,
) -> tuple[GeneratedAssets, dict]:
    strategy = get_strategy_prompt_context()
    recipe_rule = (
        "Recipe MUST be in caption only when intent.recipe is true AND ingredients/measures "
        "are explicitly stated in the visual description or transcript — never fabricate."
    )
    if analysis.audio_visual_divergence:
        recipe_rule = (
            "Audio and visuals diverge: caption.recipe must come ONLY from the visual description "
            "(or transcript if the speaker states the recipe). Leave caption.recipe empty if not evidenced."
        )

    context = f"""Filename: {filename}
Pillar: {classification.pillar}
Format: {classification.format}
Summary: {analysis.summary}
Topics: {', '.join(analysis.topics)}
Transcript: {analysis.transcript[:8000]}
Visual: {analysis.visual_description[:2000]}
Entities: {analysis.entities.model_dump_json()}
Recipe intent: {analysis.intent.recipe}
{_divergence_block(analysis)}"""

    prompt = f"""Generate Instagram content assets for Pritesh Mody (@pritesh.mody).

{context}

Rules:
- Hook in first 1.5 seconds for reels; lead with celebrity name for Drinks With Fun People
- {recipe_rule}
- 3-5 hashtags only, always include #priteshmody
- One specific binary/short-answer question CTA in caption — never generic (e.g. "Gin or Vodka Martini?" not "Tell me in the comments")
- Tag every named person/brand from entities that appear in the chosen narrative
- Carousel is TEXT ONLY (5 slides), slide 1 must work sound-off
- DM-SHARE DOCTRINE (highest-weighted Reels signal): The reel.cta MUST be a hyper-specific share-bait sentence that answers all three: WHO would send this / TO WHOM / WHY RIGHT NOW. Formula: "Send this to [specific persona] who [specific behaviour]." Examples: "Send this to your bartender mate who over-dilutes Martinis.", "Tag the friend who always orders the house Chardonnay when they could do better.", "Forward this to whoever just bought a home cocktail kit and has no idea where to start." Generic CTAs like 'share this with a friend' are strictly forbidden.

Return JSON:
{{
  "reel": {{"hook": "", "body": "", "cta": ""}},
  "caption": {{"hook": "", "story": "", "recipe": "", "cta": "", "hashtags": [], "tags": []}},
  "carousel": {{"slide_1": "", "slide_2": "", "slide_3": "", "slide_4": "", "slide_5": ""}},
  "stories": [
    {{"type": "poll|question_sticker|behind_the_scenes", "content": ""}}
  ]
}}"""

    # Compose system prompt: recipe-constraint block first, then live strategy.
    system = f"{_GENERATION_SYSTEM_PROMPT}\n\n{strategy}"

    data, usage = chat_json(system=system, user=prompt)
    assets = GeneratedAssets(
        reel=ReelScript.model_validate(data.get("reel", {})),
        caption=Caption.model_validate(data.get("caption", {})),
        carousel=CarouselSlides.model_validate(data.get("carousel", {})),
        stories=[StoryIdea.model_validate(s) for s in data.get("stories", [])],
    )
    return _sanitize_recipe_if_ungrounded(assets, analysis), usage


# Regex patterns that prove a real numeric drink specification exists in the source.
_MEASUREMENT_PATTERNS = [
    r"\d+\s*ml",
    r"\d+\s*oz",
    r"\d+\s*cl",
    r"dash(es)?",
    r"part(s)?",
    r"drop(s)?",
]

# Core spirits / mixers / modifiers that must appear in the source to count as
# ingredient evidence.  Deliberately broad so accented speech is still caught.
_CORE_INGREDIENTS = [
    "gin", "vodka", "tequila", "mezcal", "rum", "whiskey", "whisky", "bourbon",
    "cognac", "brandy", "vermouth", "aperol", "campari", "cointreau", "triple sec",
    "lime", "lemon", "orange", "grapefruit", "sugar", "syrup", "honey", "agave",
    "bitters", "tonic", "soda", "prosecco", "champagne", "beer", "cider",
    "elderflower", "cucumber", "mint", "basil", "ginger", "jalapeño",
]


def _sanitize_recipe_if_ungrounded(assets: GeneratedAssets, analysis: ContentAnalysis) -> GeneratedAssets:
    """Strip any recipe the LLM generated that is not grounded in source material.

    Two-gate check:
    1. Hard gate — if the upstream intent classifier says no recipe, blank it.
    2. Evidence gate — require at least one numeric measurement pattern OR a
       recognised core ingredient to appear in the combined transcript + visual
       description.  This prevents the model from fabricating a plausible-
       sounding recipe when none was spoken or shown (e.g. the Giorgio Welsh
       transcript incident).
    """
    recipe_text = assets.caption.recipe.strip() if assets.caption.recipe else ""

    # Gate 1: intent classifier says this content has no recipe.
    if not analysis.intent.recipe:
        if recipe_text:
            logger.warning(
                "Sanitiser stripped fabricated recipe (intent.recipe=False): %s…",
                recipe_text[:60],
            )
        assets.caption.recipe = ""
        return assets

    # No recipe was generated — nothing to validate.
    if not recipe_text:
        return assets

    # Build the evidence corpus from all available source material.
    if analysis.audio_visual_divergence:
        # For divergent content, recipe evidence must come from visuals only.
        evidence = f"{analysis.visual_description}\n{analysis.visual_focus_summary}".lower()
    else:
        evidence = (
            f"{analysis.transcript}\n"
            f"{analysis.visual_description}\n"
            f"{analysis.visual_focus_summary}"
        ).lower()

    # Gate 2a: numeric measurement present in source?
    has_measurements = any(re.search(pat, evidence) for pat in _MEASUREMENT_PATTERNS)

    # Gate 2b: at least one recognised base ingredient present in source?
    has_ingredient = (
        len(analysis.entities.ingredients) > 0
        or any(ing in evidence for ing in _CORE_INGREDIENTS)
    )

    if not (has_measurements or has_ingredient):
        logger.warning(
            "Sanitiser stripped ungrounded recipe (no measurement or ingredient evidence): %s…",
            recipe_text[:60],
        )
        assets.caption.recipe = ""
        _scrub_recipe_from_secondary_fields(assets)

    return assets


def _scrub_recipe_from_secondary_fields(assets: GeneratedAssets) -> None:
    """After a recipe is cleared, remove any numeric measure tokens that leaked
    into carousel slides or story content during generation.

    This catches edge cases where the LLM front-loaded ingredient quantities
    into a carousel slide (e.g. slide_2 = "50ml Gin, 25ml elderflower…") even
    though the recipe field itself has been blanked.
    """
    _measure_re = re.compile(
        r"\b\d+\s*(?:ml|oz|cl)\b|\b(?:dash(?:es)?|part(?:s)?|drop(?:s)?)\b",
        re.IGNORECASE,
    )

    # Sweep carousel slides
    for slot in ("slide_1", "slide_2", "slide_3", "slide_4", "slide_5"):
        slide_text = getattr(assets.carousel, slot, "") or ""
        if _measure_re.search(slide_text):
            logger.warning(
                "Scrubbing leaked recipe measures from carousel %s: %s…", slot, slide_text[:60]
            )
            setattr(assets.carousel, slot, "")

    # Sweep story content
    for story in assets.stories:
        if _measure_re.search(story.content or ""):
            logger.warning(
                "Scrubbing leaked recipe measures from story content: %s…", story.content[:60]
            )
            story.content = ""


# ---------------------------------------------------------------------------
# Weekly Rhythm — deterministic day/time slots per pillar
# Source: PM Drinks Instagram Strategy v2
# ---------------------------------------------------------------------------
_WEEKLY_RHYTHM: dict[str, dict[str, str]] = {
    "CRAFT":     {"day": "Monday",   "time": "7:00 PM",  "alt_day": "Saturday", "alt_time": "11:00 AM"},
    "AUTHORITY": {"day": "Tuesday",  "time": "7:00 PM",  "alt_day": "Thursday", "alt_time": "7:00 PM"},
    "PROOF":     {"day": "Thursday", "time": "8:00 PM",  "alt_day": "Thursday", "alt_time": "8:00 PM"},
}


def _weekly_rhythm_slot(pillar: str) -> tuple[str, str]:
    """Return the (recommended_day, recommended_time) for a given pillar.

    Falls back to Thursday 7 PM for any unrecognised pillar so the pipeline
    never produces an empty scheduling recommendation.
    """
    slot = _WEEKLY_RHYTHM.get(pillar.upper(), {"day": "Thursday", "time": "7:00 PM"})
    return slot["day"], slot["time"]


def generate_publishing_recommendation(
    analysis: ContentAnalysis,
    classification: Classification,
) -> tuple[PublishingRecommendation, dict]:
    strategy = get_strategy_prompt_context()
    divergence = ""
    if analysis.audio_visual_divergence:
        divergence = f"\nNote: audio and visuals diverge. {analysis.repurposing_guidance}"

    # ── Deterministic scheduling: lock day/time from Weekly Rhythm table ──
    # The LLM is NOT allowed to choose the posting day — the strategy defines
    # it.  We inject the locked slot as a hard constraint in the prompt so the
    # model fills in reason/series/follow-up coherently around a fixed anchor.
    locked_day, locked_time = _weekly_rhythm_slot(classification.pillar)

    prompt = f"""Recommend publishing strategy for this Instagram content.

Pillar: {classification.pillar}
Format: {classification.format}
Summary: {analysis.summary}
People mentioned: {', '.join(analysis.entities.people)}
{divergence}

SCHEDULING CONSTRAINT (do not override):
- recommended_day MUST be "{locked_day}"
- recommended_time MUST be "{locked_time}"
These are fixed by the PM Drinks Weekly Rhythm. Your job is to write the reason,
series label, and follow-up slots that make this slot make sense — not to choose a different day.

Return JSON:
{{
  "platform": "Instagram",
  "content_type": "Reel|Carousel|Story",
  "recommended_day": "{locked_day}",
  "recommended_time": "{locked_time}",
  "series": "e.g. Drinks With Fun People · #02 or Menu Behind The Menu · #01 or empty",
  "reason": "2-3 sentences tied to pillar and format",
  "confidence": 0-100,
  "clearance_flag": "[CLEAR: guest team] if celebrity, else empty",
  "follow_up_slots": [
    {{"day": "", "time": "", "note": ""}}
  ]
}}"""

    data, usage = chat_json(
        system=f"You recommend Instagram publishing schedules for a cocktail creator.\n\n{strategy}",
        user=prompt,
    )
    # Enforce locked values server-side — LLM cannot override the Weekly Rhythm
    data["recommended_day"] = locked_day
    data["recommended_time"] = locked_time
    return PublishingRecommendation.model_validate(data), usage
