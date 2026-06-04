import json
from typing import Any

from openai import OpenAI

from app.config import settings


class OpenAINotConfiguredError(Exception):
    pass


def require_openai_client() -> OpenAI:
    if not settings.openai_api_key.strip():
        raise OpenAINotConfiguredError(
            "OPENAI_API_KEY is not set. Add it to backend/.env to enable AI processing."
        )
    return OpenAI(api_key=settings.openai_api_key)


def _extract_usage(response) -> dict[str, int]:
    """Extract token usage from an OpenAI chat completion response.

    Returns a dict with keys:
        prompt_tokens     – total input tokens billed (including cached)
        completion_tokens – output tokens
        cached_tokens     – subset of prompt_tokens served from the prompt cache
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    cached_tokens = getattr(
        getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0
    ) or 0

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
    }


def chat_json(
    system: str,
    user: str,
    image_paths: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Call the chat completions API and return a JSON-parsed dict plus usage metadata.

    Returns:
        (data_dict, usage_dict) where usage_dict has keys:
            prompt_tokens, completion_tokens, cached_tokens
    """
    client = require_openai_client()

    if image_paths:
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for path in image_paths:
            import base64

            data = base64.b64encode(open(path, "rb").read()).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{data}", "detail": "low"},
                }
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    raw = response.choices[0].message.content or "{}"
    return json.loads(raw), _extract_usage(response)


def chat_text(system: str, user: str) -> tuple[str, dict[str, int]]:
    """Call the chat completions API and return plain text plus usage metadata.

    Returns:
        (text, usage_dict) where usage_dict has keys:
            prompt_tokens, completion_tokens, cached_tokens
    """
    client = require_openai_client()
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content or "", _extract_usage(response)
