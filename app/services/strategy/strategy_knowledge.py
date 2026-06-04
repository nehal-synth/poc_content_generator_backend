import json
from functools import lru_cache
from pathlib import Path
from typing import Any


STRATEGY_PATH = Path(__file__).resolve().parents[2] / "data" / "strategy.json"


@lru_cache
def load_strategy() -> dict[str, Any]:
    with STRATEGY_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def get_strategy_prompt_context() -> str:
    strategy = load_strategy()
    lines = [
        f"Positioning: {strategy['positioning']}",
        "",
        "Content Pillars:",
    ]
    for pillar in strategy["pillars"]:
        lines.append(
            f"- {pillar['id']} (~{pillar['share']}%): {pillar['description']} "
            f"Examples: {', '.join(pillar['examples'])}"
        )
    lines.append("")
    lines.append("Content Formats:")
    for fmt in strategy["formats"]:
        lines.append(
            f"- {fmt['id']} ({fmt['pillar']}, {fmt['media']}, {fmt['length']}): "
            f"{fmt['description']} Hook: {fmt['hook_template']}"
        )
    lines.append("")
    lines.append("Caption rules:")
    for key, value in strategy["caption_rules"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Reel rules:")
    for key, value in strategy["reel_rules"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def get_format_catalog() -> list[dict[str, Any]]:
    return load_strategy()["formats"]


def get_pillar_ids() -> list[str]:
    return [p["id"] for p in load_strategy()["pillars"]]
