"""Gmail helpers for the voice assistant: read recent mail and create drafts.

The assistant only reads mail and writes drafts — it never sends. Authorization
is handled by google_auth_helper (run `python google_auth_helper.py` once).
"""
import base64
from email.message import EmailMessage

from googleapiclient.discovery import build

from google_auth_helper import get_credentials

_service = None


def get_service():
    """Returns an authorized Gmail API client."""
    global _service
    if _service is None:
        _service = build("gmail", "v1", credentials=get_credentials())
    return _service


def list_recent_emails(max_results=5, only_unread=True):
    """Returns sender/subject/snippet for the most recent (optionally unread) mail."""
    service = get_service()
    query = "is:unread" if only_unread else ""
    listing = (
        service.users()
        .messages()
        .list(userId="me", q=query, maxResults=max_results)
        .execute()
    )
    emails = []
    for ref in listing.get("messages", []):
        msg = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=ref["id"],
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            )
            .execute()
        )
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        emails.append(
            {
                "from": headers.get("From"),
                "subject": headers.get("Subject"),
                "date": headers.get("Date"),
                "snippet": msg.get("snippet"),
            }
        )
    return emails


def create_draft(to, subject, body):
    """Creates a draft email (never sent) and returns its draft id."""
    service = get_service()
    message = EmailMessage()
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)
    encoded = base64.urlsafe_b64encode(message.as_bytes()).decode()
    draft = (
        service.users()
        .drafts()
        .create(userId="me", body={"message": {"raw": encoded}})
        .execute()
    )
    return draft["id"]
