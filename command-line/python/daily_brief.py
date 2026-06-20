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

# Personalization (defaults: Bratislava, Slovak Google News).
USER_NAME = os.environ.get("BRIEF_USER_NAME", "")
LATITUDE = float(os.environ.get("BRIEF_LATITUDE", "48.1486"))
LONGITUDE = float(os.environ.get("BRIEF_LONGITUDE", "17.1077"))
NEWS_FEED_URL = os.environ.get(
    "BRIEF_NEWS_FEED", "https://news.google.com/rss?hl=sk&gl=SK&ceid=SK:sk"
)
NEWS_COUNT = int(os.environ.get("BRIEF_NEWS_COUNT", "5"))
MAIL_COUNT = int(os.environ.get("BRIEF_MAIL_COUNT", "5"))

# WMO weather codes (Open-Meteo) → short descriptions.
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
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
        "current_temp_c": current["temperature_2m"],
        "condition": WEATHER_CODES.get(current["weather_code"], "unknown"),
        "today_high_c": daily["temperature_2m_max"][0],
        "today_low_c": daily["temperature_2m_min"][0],
    }


def get_news():
    """Returns the top headlines from the configured RSS feed."""
    feed = feedparser.parse(NEWS_FEED_URL)
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
