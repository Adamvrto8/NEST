"""Shared Google OAuth for the assistant's Gmail and Calendar tools.

Run `python google_auth_helper.py` once to authorize (opens a browser) and
create token.json covering every scope below. On a headless Raspberry Pi, run
this on a machine with a browser and copy the resulting token.json over.
"""
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# One token covers all of these. gmail.compose creates drafts (never sends);
# calendar.events reads and creates calendar events.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/calendar.events",
]
CREDENTIALS_FILE = os.environ.get("GMAIL_CREDENTIALS", "credentials.json")
TOKEN_FILE = os.environ.get("GMAIL_TOKEN", "token.json")


def get_credentials():
    """Returns authorized credentials, running the consent flow if needed."""
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
    return creds


if __name__ == "__main__":
    get_credentials()
    print(f"Authorized. Token saved to {TOKEN_FILE}.")
