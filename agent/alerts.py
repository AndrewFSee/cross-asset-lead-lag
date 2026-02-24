"""Alerting system for lead-lag structural break events."""

from __future__ import annotations

import json
import logging
import smtplib
import urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def format_alert(events: list) -> str:
    """Format a list of events into a concise alert message.

    Args:
        events: List of StructuralBreakEvent instances.

    Returns:
        Formatted alert string.
    """
    if not events:
        return "Lead-Lag Monitor: No structural breaks detected."

    lines = [f"🚨 Lead-Lag Monitor Alert: {len(events)} event(s) detected"]
    for event in events:
        emoji = {"te_spike": "📈", "te_decay": "📉", "regime_change": "🔄"}.get(
            event.event_type, "⚠️"
        )
        lines.append(f"{emoji} {event.description}")

    return "\n".join(lines)


def send_slack_alert(
    webhook_url: str,
    message: str,
) -> bool:
    """Send an alert message to a Slack channel via webhook.

    Args:
        webhook_url: Slack incoming webhook URL.
        message: Message text to send.

    Returns:
        True if sent successfully, False otherwise.
    """
    if not webhook_url:
        logger.warning("No Slack webhook URL configured; skipping alert")
        return False

    payload = json.dumps({"text": message}).encode("utf-8")
    try:
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                logger.info("Slack alert sent successfully")
                return True
            else:
                logger.warning("Slack alert failed with status %d", resp.status)
                return False
    except Exception as exc:
        logger.error("Failed to send Slack alert: %s", exc)
        return False


def send_email_alert(
    smtp_config: Dict[str, object],
    subject: str,
    body: str,
    to_addresses: Optional[List[str]] = None,
) -> bool:
    """Send an alert via email using SMTP.

    Args:
        smtp_config: Dict with keys: host, port, user, password, from_addr.
        subject: Email subject line.
        body: Email body text (plain text or HTML).
        to_addresses: List of recipient email addresses.

    Returns:
        True if sent successfully, False otherwise.
    """
    if not smtp_config:
        logger.warning("No SMTP config provided; skipping email alert")
        return False

    if to_addresses is None:
        to_addresses = [str(smtp_config.get("from_addr", ""))]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = str(smtp_config.get("from_addr", ""))
    msg["To"] = ", ".join(to_addresses)
    msg.attach(MIMEText(body, "plain"))

    try:
        host = str(smtp_config.get("host", "localhost"))
        port = int(smtp_config.get("port", 587))
        user = str(smtp_config.get("user", ""))
        password = str(smtp_config.get("password", ""))

        with smtplib.SMTP(host, port, timeout=10) as server:
            server.starttls()
            if user and password:
                server.login(user, password)
            server.sendmail(msg["From"], to_addresses, msg.as_string())

        logger.info("Email alert sent to %s", to_addresses)
        return True
    except Exception as exc:
        logger.error("Failed to send email alert: %s", exc)
        return False
