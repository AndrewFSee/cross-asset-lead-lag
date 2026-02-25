"""Agent module for monitoring, narrative generation, and alerting."""

from agent.alerts import format_alert, send_email_alert, send_slack_alert
from agent.monitor import LeadLagMonitor, StructuralBreakEvent
from agent.narrator import call_llm, format_daily_report, generate_narrative_prompt

__all__ = [
    "LeadLagMonitor",
    "StructuralBreakEvent",
    "generate_narrative_prompt",
    "call_llm",
    "format_daily_report",
    "send_slack_alert",
    "send_email_alert",
    "format_alert",
]
