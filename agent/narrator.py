"""LLM-powered narrative generation for lead-lag events."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_narrative_prompt(
    events: list,
    macro_context: Optional[Dict[str, float]] = None,
) -> str:
    """Build a prompt for LLM narrative generation.

    Args:
        events: List of StructuralBreakEvent instances.
        macro_context: Optional dict of macro indicator values (e.g. VIX level).

    Returns:
        Formatted prompt string.
    """
    lines = [
        "You are a quantitative macro analyst. Analyze the following lead-lag "
        "relationship changes detected in financial markets and provide a concise "
        "daily commentary explaining their significance and potential trading implications.",
        "",
        "## Detected Events",
    ]

    for event in events:
        lines.append(f"- [{event.event_type.upper()}] {event.description}")

    if macro_context:
        lines.append("")
        lines.append("## Macro Context")
        for key, val in macro_context.items():
            lines.append(f"- {key}: {val:.4f}")

    lines.extend([
        "",
        "## Instructions",
        "1. Explain what each event means economically.",
        "2. Identify which events are most significant for risk/trading.",
        "3. Suggest any portfolio adjustments warranted by these signals.",
        "4. Keep the response concise (under 300 words).",
    ])

    return "\n".join(lines)


def call_llm(
    prompt: str,
    model: str = "gpt-4",
    api_key: Optional[str] = None,
) -> str:
    """Call the OpenAI API to generate a narrative.

    Gracefully handles missing API keys by returning a placeholder message.

    Args:
        prompt: The prompt to send to the LLM.
        model: OpenAI model name (e.g., 'gpt-4', 'gpt-3.5-turbo').
        api_key: OpenAI API key. Falls back to settings/env if None.

    Returns:
        Generated narrative string.
    """
    if api_key is None:
        try:
            from config.settings import Settings  # noqa: PLC0415

            api_key = Settings().openai_api_key
        except Exception:
            pass

    if not api_key:
        logger.warning("No OpenAI API key configured; returning placeholder narrative")
        return (
            "[LLM narrative unavailable: OPENAI_API_KEY not set. "
            "Configure your API key to enable automated commentary.]"
        )

    try:
        from openai import OpenAI  # noqa: PLC0415

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a quantitative macro analyst specializing in cross-asset relationships.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return f"[LLM call failed: {exc}]"


def format_daily_report(
    events: list,
    narratives: Optional[str] = None,
    date: Optional[datetime] = None,
) -> str:
    """Format a markdown daily report from events and narratives.

    Args:
        events: List of StructuralBreakEvent instances.
        narratives: Optional LLM-generated narrative string.
        date: Report date. Defaults to now.

    Returns:
        Markdown-formatted report string.
    """
    if date is None:
        date = datetime.utcnow()

    lines = [
        f"# Lead-Lag Monitor Daily Report",
        f"**Date:** {date.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        "",
        "## Detected Events",
    ]

    if not events:
        lines.append("_No structural breaks detected today._")
    else:
        spikes = [e for e in events if e.event_type == "te_spike"]
        decays = [e for e in events if e.event_type == "te_decay"]
        regime_changes = [e for e in events if e.event_type == "regime_change"]

        if spikes:
            lines.append("\n### TE Spikes (New/Strengthening Relationships)")
            for e in spikes:
                lines.append(f"- {e.description}")

        if decays:
            lines.append("\n### TE Decays (Weakening Relationships)")
            for e in decays:
                lines.append(f"- {e.description}")

        if regime_changes:
            lines.append("\n### Regime Transitions")
            for e in regime_changes:
                lines.append(f"- {e.description}")

    if narratives:
        lines.extend(["", "---", "", "## Analyst Commentary", "", narratives])

    return "\n".join(lines)
