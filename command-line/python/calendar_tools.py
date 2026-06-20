"""Google Calendar helpers for the voice assistant: list and create events.

Authorization is handled by google_auth_helper (run `python google_auth_helper.py`
once). The Google Calendar API must be enabled in the Cloud project.
"""
import datetime
import os

from googleapiclient.discovery import build

from google_auth_helper import get_credentials

# Time zone used when creating timed events. Defaults to the assistant's locale.
CALENDAR_TIMEZONE = os.environ.get("CALENDAR_TIMEZONE", "Europe/Bratislava")

_service = None


def get_service():
    """Returns an authorized Calendar API client."""
    global _service
    if _service is None:
        _service = build("calendar", "v3", credentials=get_credentials())
    return _service


def _event_time(value):
    """Builds a Calendar time field: all-day for a date, timed for a datetime."""
    if len(value) == 10:  # YYYY-MM-DD
        return {"date": value}
    return {"dateTime": value, "timeZone": CALENDAR_TIMEZONE}


def list_upcoming_events(max_results=10, time_min=None, time_max=None):
    """Returns upcoming events from the primary calendar, soonest first."""
    service = get_service()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    params = {
        "calendarId": "primary",
        "timeMin": time_min or now,
        "maxResults": max_results,
        "singleEvents": True,
        "orderBy": "startTime",
    }
    if time_max:
        params["timeMax"] = time_max
    items = service.events().list(**params).execute().get("items", [])
    events = []
    for item in items:
        events.append(
            {
                "summary": item.get("summary", "(no title)"),
                "start": item["start"].get("dateTime", item["start"].get("date")),
                "end": item["end"].get("dateTime", item["end"].get("date")),
                "location": item.get("location"),
            }
        )
    return events


def create_event(summary, start, end, description=None, location=None):
    """Creates an event on the primary calendar and returns its event id."""
    service = get_service()
    event = {
        "summary": summary,
        "start": _event_time(start),
        "end": _event_time(end),
    }
    if description:
        event["description"] = description
    if location:
        event["location"] = location
    created = service.events().insert(calendarId="primary", body=event).execute()
    return created.get("id")
