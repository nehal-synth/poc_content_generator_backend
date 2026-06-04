from pathlib import Path

from app.config import settings
from app.services.openai.openai_client import require_openai_client


class TranscriptionError(Exception):
    pass


def transcribe_raw_audio(audio_path: Path) -> tuple[str, float]:
    """Stage 1 — Raw phonetic extraction.

    Produces an English-only transcript with zero temperature to prevent
    accent-driven language drift (e.g. Welsh, Hindi).  The output is
    intentionally unpolished: brand names and proper nouns may be phonetically
    misspelled.  A separate correction layer (transcript_corrector.py) is
    responsible for semantic alignment using visual and strategy context.

    Returns:
        (transcript_text, audio_duration_seconds) — duration is the value
        reported by the Whisper API when verbose_json format is used.  Falls
        back to 0.0 if the API does not return duration metadata.
    """
    client = require_openai_client()

    with audio_path.open("rb") as audio_file:
        result = client.audio.transcriptions.create(
            model=settings.whisper_model,
            file=audio_file,
            response_format="verbose_json",  # returns duration + text
            language="en",   # Structural constraint: forces English phonetics globally
            temperature=0.0, # Deterministic decoding — no hallucination, no sampling drift
        )

    # verbose_json returns an object with .text and .duration
    text = (result.text or "").strip() if hasattr(result, "text") else str(result).strip()
    duration_seconds: float = float(getattr(result, "duration", 0.0) or 0.0)

    if not text:
        raise TranscriptionError("Whisper returned an empty transcript for this audio.")

    return text, duration_seconds
