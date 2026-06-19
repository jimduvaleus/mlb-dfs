"""Email delivery for app-generated notifications, via a Gmail App Password.

Credentials are read from environment variables (populated from a gitignored
.env at the repo root via python-dotenv) so they never pass through
AppConfig / the /api/config endpoint or the React ConfigForm.
"""
import logging
import smtplib
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_SMTP_HOST = "smtp.gmail.com"
_SMTP_PORT = 587


def send_notification_email(summary: str, body: str) -> None:
    """Send a plain-text email for an app-generated notification.

    Logs and returns on missing credentials or any SMTP failure rather than
    raising — email delivery must never break the caller's request.
    """
    sender = os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    recipient = os.environ.get("NOTIFICATION_EMAIL_TO")
    if not (sender and password and recipient):
        logger.warning("Skipping notification email: GMAIL_ADDRESS/GMAIL_APP_PASSWORD/NOTIFICATION_EMAIL_TO not set")
        return

    msg = EmailMessage()
    msg["Subject"] = summary
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(body)

    try:
        with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=15) as smtp:
            smtp.starttls()
            smtp.login(sender, password)
            smtp.send_message(msg)
    except Exception:
        logger.exception("Failed to send notification email: %s", summary)
