"""Builds a personalized daily brief: weather, news, today's calendar and mail.

Triggered when the user asks for a "prehľad". Each section fails independently so
one outage doesn't sink the whole brief. Configure location, news source, and the
user's name with the BRIEF_* environment variables below.
"""
import datetime
import os

import feedparser
import requests

import calendar_tools
import gmail_tools

# Personalization (defaults tuned for Lubo: Bratislava weather, SME news).
USER_NAME = os.environ.get("BRIEF_USER_NAME", "Lubo")
LATITUDE = float(os.environ.get("BRIEF_LATITUDE", "48.1486"))
LONGITUDE = float(os.environ.get("BRIEF_LONGITUDE", "17.1077"))
LOCATION_NAME = os.environ.get("BRIEF_LOCATION_NAME", "Bratislava")
NEWS_FEED_URL = os.environ.get("BRIEF_NEWS_FEED", "https://www.sme.sk/rss")
NEWS_COUNT = int(os.environ.get("BRIEF_NEWS_COUNT", "5"))
MAIL_COUNT = int(os.environ.get("BRIEF_MAIL_COUNT", "5"))

# Some feeds block the default feedparser agent; identify as a normal client.
NEWS_USER_AGENT = "Mozilla/5.0 (compatible; SmartSpeaker/1.0)"

# WMO weather codes (Open-Meteo) → Slovak descriptions.
WEATHER_CODES = {
    0: "jasno",
    1: "prevažne jasno",
    2: "polooblačno",
    3: "zamračené",
    45: "hmla",
    48: "mrznúca hmla",
    51: "slabé mrholenie",
    53: "mierne mrholenie",
    55: "husté mrholenie",
    56: "slabé mrznúce mrholenie",
    57: "husté mrznúce mrholenie",
    61: "slabý dážď",
    63: "mierny dážď",
    65: "silný dážď",
    66: "slabý mrznúci dážď",
    67: "silný mrznúci dážď",
    71: "slabé sneženie",
    73: "mierne sneženie",
    75: "silné sneženie",
    77: "snehové zrná",
    80: "slabé dažďové prehánky",
    81: "mierne dažďové prehánky",
    82: "prudké dažďové prehánky",
    85: "slabé snehové prehánky",
    86: "silné snehové prehánky",
    95: "búrka",
    96: "búrka so slabým krupobitím",
    99: "búrka so silným krupobitím",
}


def get_weather():
    """Returns current and today's forecast from Open-Meteo (no API key)."""
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "current": "temperature_2m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,weather_code",
            "timezone": "auto",
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    current, daily = data["current"], data["daily"]
    return {
        "location": LOCATION_NAME,
        "current_temp_c": current["temperature_2m"],
        "condition": WEATHER_CODES.get(current["weather_code"], "neznáme"),
        "today_high_c": daily["temperature_2m_max"][0],
        "today_low_c": daily["temperature_2m_min"][0],
    }


def get_news():
    """Returns the top headlines from the configured RSS feed."""
    resp = requests.get(
        NEWS_FEED_URL, headers={"User-Agent": NEWS_USER_AGENT}, timeout=10
    )
    resp.raise_for_status()
    feed = feedparser.parse(resp.content)
    return [entry.get("title", "") for entry in feed.entries[:NEWS_COUNT]]


def get_today_events():
    """Returns calendar events occurring today."""
    start = (
        datetime.datetime.now()
        .astimezone()
        .replace(hour=0, minute=0, second=0, microsecond=0)
    )
    end = start + datetime.timedelta(days=1)
    return calendar_tools.list_upcoming_events(
        max_results=20, time_min=start.isoformat(), time_max=end.isoformat()
    )


def get_unread_mail():
    """Returns recent unread emails."""
    return gmail_tools.list_recent_emails(max_results=MAIL_COUNT, only_unread=True)


def build_brief():
    """Gathers every section; a failing section returns an error note, not a crash."""
    brief = {}
    if USER_NAME:
        brief["user_name"] = USER_NAME
    sections = {
        "weather": get_weather,
        "news": get_news,
        "calendar": get_today_events,
        "mail": get_unread_mail,
    }
    for key, fetch in sections.items():
        try:
            brief[key] = fetch()
        except Exception as exc:
            brief[key] = {"error": str(exc)}
    return brief
