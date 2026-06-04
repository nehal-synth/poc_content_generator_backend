from pathlib import Path

from app.models.schemas import ContentAnalysis, ContentIntent, Entities
from app.services.openai.openai_client import chat_json
from app.services.strategy.strategy_knowledge import get_strategy_prompt_context


class InsufficientContentError(Exception):
    """Raised when there is no transcript or visual description to analyze."""


def analyze_content(
    source_text: str,
    filename: str,
    visual_description: str = "",
) -> tuple[ContentAnalysis, dict]:
    if not source_text.strip() and not visual_description.strip():
        raise InsufficientContentError(
            "No transcript or visual description available. "
            "Upload content with speech, companion notes, or clearer video footage."
        )

    has_transcript = bool(source_text.strip())
    has_visual = bool(visual_description.strip())

    strategy = get_strategy_prompt_context()
    divergence_instructions = ""
    if has_transcript and has_visual:
        divergence_instructions = """
IMPORTANT — dual-channel video (audio + picture may disagree):
- The transcript is what was SAID (may be crew chat, scheduling, anecdotes, any language).
- The visual description is what was SHOWN on screen (cocktail build, bar action, B-roll, etc.).
- Determine if audio and visuals tell the SAME story or DIFFERENT stories (e.g. background conversation while cocktail B-roll plays).
- If they diverge, set audio_visual_divergence to true and fill audio_focus_summary vs visual_focus_summary separately.
- Set intent.recipe true ONLY if a recipe is evidenced in the visual description or transcript (ingredients/measures/steps). Do not infer recipe from brand context alone.
- repurposing_guidance: one short paragraph telling the copywriter what to use from audio vs visuals (e.g. "Use dialogue for story/hook and names; use on-screen action only for recipe and pour technique; do not invent ingredients.").
- summary: explain both channels and how they relate for Instagram repurposing."""
    elif has_visual and not has_transcript:
        divergence_instructions = """
Only visual description is available (little or no speech). Base entities, topics, intent, and summary on visuals."""

    prompt = f"""Analyze this content for Pritesh Mody's Instagram content repurposing platform.

Filename: {filename}
Transcript / document text (spoken audio or uploaded text):
{source_text[:12000] if has_transcript else "(none)"}

Visual description (on-screen action from video frames):
{visual_description[:4000] if has_visual else "(none)"}
{divergence_instructions}

Return JSON with this exact structure:
{{
  "summary": "2-3 paragraphs",
  "audio_focus_summary": "what the transcript/dialogue is about (empty if no transcript)",
  "visual_focus_summary": "what happens on screen (empty if no visual)",
  "audio_visual_divergence": true/false,
  "repurposing_guidance": "how to use audio vs visuals when generating posts",
  "entities": {{
    "people": [],
    "brands": [],
    "restaurants": [],
    "hotels": [],
    "cocktails": [],
    "ingredients": [],
    "locations": []
  }},
  "topics": ["topic1", "topic2"],
  "intent": {{
    "recipe": true/false,
    "storytelling": true/false,
    "education": true/false,
    "case_study": true/false,
    "opinion": true/false
  }}
}}"""

    data, usage = chat_json(
        system=(
            "You are a content analyst for a premium cocktail consultancy. "
            "You separate what was said from what was shown when they differ.\n\n"
            f"{strategy}"
        ),
        user=prompt,
    )
    return ContentAnalysis(
        transcript=source_text,
        visual_description=visual_description,
        summary=data.get("summary", ""),
        audio_focus_summary=data.get("audio_focus_summary", ""),
        visual_focus_summary=data.get("visual_focus_summary", ""),
        audio_visual_divergence=bool(data.get("audio_visual_divergence", False)),
        repurposing_guidance=data.get("repurposing_guidance", ""),
        entities=Entities.model_validate(data.get("entities", {})),
        topics=data.get("topics", []),
        intent=ContentIntent.model_validate(data.get("intent", {})),
    ), usage


def describe_video_frames(frame_paths: list[Path]) -> tuple[str, dict]:
    if not frame_paths:
        raise InsufficientContentError(
            "Could not extract video frames for visual analysis. "
            "Check that ffmpeg is installed and the video file is valid."
        )

    prompt = """Describe ONLY what is visible in these video key frames (on-screen action).
Do not guess what people are saying — you cannot hear the audio.
Focus on: cocktail builds, pours, ingredients and bottles visible, bar tools, garnish, setting, on-screen text/branding, and who appears on camera.
If the frames show drink-making while audio might be unrelated conversation, describe the drink-making clearly.
Write 2-4 paragraphs suitable for content repurposing.

Return JSON: {"description": "your paragraphs here"}"""

    data, usage = chat_json(
        system="You describe visible cocktail and hospitality footage. You never invent dialogue.",
        user=prompt,
        image_paths=[str(p) for p in frame_paths],
    )
    if isinstance(data, dict) and "description" in data:
        description = data["description"]
    else:
        description = str(data)
    if not str(description).strip():
        raise InsufficientContentError(
            "Visual analysis returned no description. Try a longer video or add companion notes."
        )
    return str(description).strip(), usage
