from app.models.schemas import Classification, ContentAnalysis, FormatScore
from app.services.openai.openai_client import chat_json
from app.services.strategy.strategy_knowledge import get_format_catalog, get_pillar_ids, get_strategy_prompt_context


def classify_content(analysis: ContentAnalysis, filename: str) -> tuple[Classification, dict]:
    strategy = get_strategy_prompt_context()
    formats = get_format_catalog()
    pillars = get_pillar_ids()
    format_list = "\n".join(f"- {f['id']}: {f['description']}" for f in formats)

    divergence_note = ""
    if analysis.audio_visual_divergence:
        divergence_note = f"""
Audio vs visuals DIVERGE (treat as one repurposing package):
- Audio focus: {analysis.audio_focus_summary}
- Visual focus: {analysis.visual_focus_summary}
- Guidance: {analysis.repurposing_guidance}
Classify based on the COMBINED repurposing intent (usually lead with the stronger publishable angle)."""

    prompt = f"""Classify this content for Instagram strategy.

Filename: {filename}
Summary: {analysis.summary}
Topics: {', '.join(analysis.topics)}
Transcript excerpt: {analysis.transcript[:4000]}
Visual excerpt: {analysis.visual_description[:2000]}
{divergence_note}

Available pillars: {', '.join(pillars)}
Available formats:
{format_list}

Return JSON:
{{
  "pillar": "PROOF|AUTHORITY|CRAFT",
  "pillar_confidence": 0-100,
  "pillar_scores": {{"PROOF": 0-100, "AUTHORITY": 0-100, "CRAFT": 0-100}},
  "format": "exact format name from list",
  "format_confidence": 0-100,
  "format_scores": [
    {{"format": "name", "confidence": 0-100, "description": "short reason"}}
  ]
}}
Include all 10 formats in format_scores sorted by confidence descending."""

    data, usage = chat_json(
        system=f"You classify cocktail content into strategy pillars and formats.\n\n{strategy}",
        user=prompt,
    )
    format_scores = [FormatScore.model_validate(s) for s in data.get("format_scores", [])]
    return Classification(
        pillar=data.get("pillar", "CRAFT"),
        pillar_confidence=int(data.get("pillar_confidence", 70)),
        pillar_scores={k: int(v) for k, v in data.get("pillar_scores", {}).items()},
        format=data.get("format", "Behind The Build"),
        format_confidence=int(data.get("format_confidence", 70)),
        format_scores=format_scores,
    ), usage
