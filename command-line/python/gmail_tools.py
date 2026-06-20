"""Gmail helpers for the voice assistant: read recent mail and create drafts.

Run `python gmail_tools.py` once to authorize (opens a browser) and create
token.json. The assistant only reads mail and writes drafts — it never sends.
"""
import base64
import os
from email.message import EmailMessage

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# readonly = read mail; compose = create drafts (the code never calls send).
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]
CREDENTIALS_FILE = os.environ.get("GMAIL_CREDENTIALS", "credentials.json")
TOKEN_FILE = os.environ.get("GMAIL_TOKEN", "token.json")

_service = None


def get_service():
    """Returns an authorized Gmail API client, running the consent flow if needed."""
    global _service
    if _service is not None:
        return _service
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    _service = build("gmail", "v1", credentials=creds)
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


if __name__ == "__main__":
    get_service()
    print(f"Gmail authorized. Token saved to {TOKEN_FILE}.")
