"""Free, turn-based variant of the smart speaker.

Pipeline: wake word (Vosk) -> record until you pause (energy VAD) -> Whisper
(Slovak STT) -> Gemini free text tier (brain + tools) -> edge-tts (Slovak TTS).
Online, but $0 on the Gemini free tier. Reuses the same tool backends as main.py.

Run with the venv python:  python main_local.py
"""
import asyncio
import datetime
import json
import math
import os
import struct
import threading
import time
from collections import deque

import numpy as np
import pyaudio
from google import genai
from google.genai import types
from vosk import Model, KaldiRecognizer, SetLogLevel
from faster_whisper import WhisperModel
import edge_tts
import miniaudio
from dotenv import load_dotenv

load_dotenv()  # read GEMINI_API_KEY (and BRIEF_*/CALENDAR_* overrides) from .env

import gmail_tools
import calendar_tools
import daily_brief

SetLogLevel(-1)

# --- audio ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 1024
pya = pyaudio.PyAudio()

# --- wake word (same as the Live version) ---
VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "model")
WAKE_PHRASES = ["hey gin", "hey jean", "hey gene"]

# --- speech to text ---
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
WHISPER_LANGUAGE = "sk"

# --- text to speech ---
TTS_VOICE = os.environ.get("TTS_VOICE", "sk-SK-LukasNeural")
TTS_FILE = "tts_out.mp3"

# --- brain (free text tier) ---
# Tried in order; falls over to the next on rate limits / 503 overload, so one
# model's daily cap or high-demand spike doesn't drop the turn.
FREE_MODELS = [
    m.strip()
    for m in os.environ.get(
        "FREE_MODEL", "gemini-2.5-flash,gemini-2.5-flash-lite,gemini-2.0-flash"
    ).split(",")
    if m.strip()
]
SYSTEM_PROMPT = (
    "Si nápomocný a priateľský domáci hlasový asistent. Vždy odpovedaj po slovensky, "
    "stručne, jasne a prirodzene. Nikdy neodpovedaj po anglicky. Hovor zdvorilo a "
    "jednoducho, vhodne pre staršieho používateľa. Keď používateľ požiada o prehľad, "
    "zavolaj nástroj get_daily_brief a zhrň počasie, správy, kalendár a emaily vrúcne "
    "a stručne. Na aktuálne počasie použi get_weather, na správy get_news. Odpovedaj iba "
    "obyčajným textom, ktorý sa dá prečítať nahlas — žiadne odrážky ani formátovanie."
)

# --- turn-taking ---
SILENCE_RMS = int(os.environ.get("SILENCE_RMS", "500"))  # below this = silence
SILENCE_SECONDS = 1.2   # this much silence ends an utterance
MAX_UTTERANCE = 12      # hard cap on one utterance
NO_SPEECH_TIMEOUT = 6   # give up if nothing is said after waking
CONVERSATION_TIMEOUT = 15  # return to wake word after this idle

SLOVAK_WEEKDAYS = [
    "pondelok", "utorok", "streda", "štvrtok", "piatok", "sobota", "nedeľa",
]

TOOL_DECLARATIONS = [
    {
        "name": "get_daily_brief",
        "description": "Get the user's personalized daily brief in one call: today's weather, top news headlines, today's calendar events, and recent unread emails. Use when the user asks for a 'prehľad', a summary, or a morning briefing.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_current_time",
        "description": "Get the current time, date, and weekday. Use when the user asks what time or day it is, or to resolve relative dates.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_weather",
        "description": "Get the current weather and today's forecast for the user's location. Use when the user asks about the weather.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_news",
        "description": "Get the latest news headlines. Use when the user asks about the news.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "set_timer",
        "description": "Set a countdown timer that beeps when it finishes. Use for kitchen timers or reminders after a delay.",
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {"type": "integer", "description": "Timer duration in seconds."},
                "label": {"type": "string", "description": "Optional description of the timer."},
            },
            "required": ["seconds"],
        },
    },
    {
        "name": "read_emails",
        "description": "Read the user's recent Gmail messages. Use when the user asks to check or read their email.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_results": {"type": "integer", "description": "How many emails to fetch (default 5)."},
                "only_unread": {"type": "boolean", "description": "If true, only unread emails (default true)."},
            },
        },
    },
    {
        "name": "draft_email",
        "description": "Create a draft email for the user to review and send later. This never sends automatically.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string", "description": "Email subject line."},
                "body": {"type": "string", "description": "Body text of the email."},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "list_calendar_events",
        "description": "List upcoming events from the user's Google Calendar. Use when the user asks about their schedule.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_results": {"type": "integer", "description": "Maximum number of events (default 10)."},
                "time_min": {"type": "string", "description": "Optional ISO 8601 start of range."},
                "time_max": {"type": "string", "description": "Optional ISO 8601 end of range."},
            },
        },
    },
    {
        "name": "create_calendar_event",
        "description": "Create an event in the user's Google Calendar. Resolve relative dates with get_current_time first.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Title of the event."},
                "start": {"type": "string", "description": "Start as ISO 8601 ('YYYY-MM-DDTHH:MM:SS' or 'YYYY-MM-DD')."},
                "end": {"type": "string", "description": "End as ISO 8601, same format as start."},
                "location": {"type": "string", "description": "Optional location."},
                "description": {"type": "string", "description": "Optional notes."},
            },
            "required": ["summary", "start", "end"],
        },
    },
]

def make_config():
    """Builds the request config, injecting today's date so the model rarely has
    to call get_current_time (saving a request) for date/calendar questions."""
    now = datetime.datetime.now()
    dated = (
        f"{SYSTEM_PROMPT} Dnešný dátum je {now.strftime('%Y-%m-%d')} "
        f"({SLOVAK_WEEKDAYS[now.weekday()]}), aktuálny čas je {now.strftime('%H:%M')}."
    )
    return types.GenerateContentConfig(
        system_instruction=dated,
        tools=[{"function_declarations": TOOL_DECLARATIONS}],
    )


def play_beep(freq=880, duration=0.25):
    """Plays a short tone for wake/timer feedback."""
    n = int(SAMPLE_RATE * duration)
    fade = int(SAMPLE_RATE * 0.01)
    pad = int(SAMPLE_RATE * 0.25)  # trailing silence so Bluetooth output isn't cut off
    samples = bytearray()
    for i in range(n):
        env = min(1.0, i / fade, (n - i) / fade)
        value = int(0.45 * 32767 * env * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
        samples += struct.pack("<h", value)
    samples += b"\x00\x00" * pad
    stream = pya.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, output=True)
    try:
        stream.write(bytes(samples))
        time.sleep(0.15)  # let the buffer drain before closing (Bluetooth latency)
    finally:
        stream.close()


def dispatch(name, args):
    """Runs a tool the model asked for and returns a JSON-serializable dict."""
    if name == "get_daily_brief":
        try:
            return daily_brief.build_brief()
        except Exception as exc:
            return {"status": "error", "message": f"Brief unavailable: {exc}"}
    if name == "get_current_time":
        now = datetime.datetime.now()
        return {
            "time": now.strftime("%H:%M"),
            "date": now.strftime("%Y-%m-%d"),
            "weekday": SLOVAK_WEEKDAYS[now.weekday()],
        }
    if name == "get_weather":
        try:
            return daily_brief.get_weather()
        except Exception as exc:
            return {"status": "error", "message": f"Weather unavailable: {exc}"}
    if name == "get_news":
        try:
            return {"headlines": daily_brief.get_news()}
        except Exception as exc:
            return {"status": "error", "message": f"News unavailable: {exc}"}
    if name == "set_timer":
        seconds = int(args.get("seconds", 0))
        if seconds <= 0:
            return {"status": "error", "message": "Duration must be positive."}
        threading.Timer(seconds, lambda: play_beep(880, 0.4)).start()
        return {"status": "ok", "seconds": seconds, "label": args.get("label")}
    if name == "read_emails":
        try:
            emails = gmail_tools.list_recent_emails(
                int(args.get("max_results", 5)), bool(args.get("only_unread", True))
            )
            return {"count": len(emails), "emails": emails}
        except Exception as exc:
            return {"status": "error", "message": f"Gmail unavailable: {exc}"}
    if name == "draft_email":
        to, subject, body = args.get("to"), args.get("subject"), args.get("body")
        if not (to and subject and body):
            return {"status": "error", "message": "Missing recipient, subject, or body."}
        try:
            return {"status": "ok", "draft_id": gmail_tools.create_draft(to, subject, body)}
        except Exception as exc:
            return {"status": "error", "message": f"Gmail unavailable: {exc}"}
    if name == "list_calendar_events":
        try:
            events = calendar_tools.list_upcoming_events(
                int(args.get("max_results", 10)), args.get("time_min"), args.get("time_max")
            )
            return {"count": len(events), "events": events}
        except Exception as exc:
            return {"status": "error", "message": f"Calendar unavailable: {exc}"}
    if name == "create_calendar_event":
        summary, start, end = args.get("summary"), args.get("start"), args.get("end")
        if not (summary and start and end):
            return {"status": "error", "message": "Missing summary, start, or end."}
        try:
            event_id = calendar_tools.create_event(
                summary, start, end, args.get("description"), args.get("location")
            )
            return {"status": "ok", "event_id": event_id}
        except Exception as exc:
            return {"status": "error", "message": f"Calendar unavailable: {exc}"}
    return {"status": "error", "message": f"Unknown function: {name}"}


def load_wake_model():
    if not os.path.isdir(VOSK_MODEL_PATH):
        raise RuntimeError(
            f"Vosk model not found at '{VOSK_MODEL_PATH}'. Download one from "
            "https://alphacephei.com/vosk/models, unzip it, and set VOSK_MODEL_PATH."
        )
    return Model(VOSK_MODEL_PATH)


def wait_for_wake_word(model):
    """Blocks on the mic locally until a wake phrase is heard."""
    grammar = json.dumps(WAKE_PHRASES + ["[unk]"])
    recognizer = KaldiRecognizer(model, SAMPLE_RATE, grammar)
    stream = pya.open(
        format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True,
        frames_per_buffer=CHUNK,
    )
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                text = json.loads(recognizer.Result()).get("text", "")
            else:
                text = json.loads(recognizer.PartialResult()).get("partial", "")
            if any(phrase in text for phrase in WAKE_PHRASES):
                return
    finally:
        stream.close()


def record_utterance():
    """Records the mic until a pause; returns float32 audio at 16 kHz, or None."""
    stream = pya.open(
        format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True,
        frames_per_buffer=CHUNK,
    )
    preroll = deque(maxlen=8)  # ~0.5 s of lead-in so the first word isn't clipped
    frames = []
    started = False
    silence_start = None
    start_time = time.time()
    peak = 0.0
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16)
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
            peak = max(peak, rms)
            now = time.time()
            if rms > SILENCE_RMS:
                if not started:
                    frames.extend(preroll)
                started = True
                silence_start = None
                frames.append(data)
            elif started:
                frames.append(data)
                if silence_start is None:
                    silence_start = now
                elif now - silence_start >= SILENCE_SECONDS:
                    break
            else:
                preroll.append(data)
            if started and now - start_time >= MAX_UTTERANCE:
                break
            if not started and now - start_time >= NO_SPEECH_TIMEOUT:
                print(f"  (no speech; mic peak level {peak:.0f}, threshold {SILENCE_RMS})")
                return None
    finally:
        stream.close()
    if not frames:
        return None
    audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    peak_amp = float(np.max(np.abs(audio)))
    if peak_amp > 0:
        audio = audio * min(4.0, 0.95 / peak_amp)  # normalize quiet mics, cap the gain
    return audio


def transcribe(whisper, audio):
    """Transcribes Slovak speech to text."""
    segments, _ = whisper.transcribe(
        audio,
        language=WHISPER_LANGUAGE,
        beam_size=5,
        vad_filter=True,
        initial_prompt="Rozhovor po slovensky o počasí, správach, kalendári a emailoch.",
    )
    return "".join(segment.text for segment in segments).strip()


_exhausted = set()  # models whose daily cap is hit; skipped for the rest of the run


def generate(client, contents, config, per_model_retries=2):
    """Calls the model with retry + fallback across FREE_MODELS, so a model's
    daily cap or 503 overload fails over instead of dropping the turn."""
    last_exc = None
    models = [m for m in FREE_MODELS if m not in _exhausted] or FREE_MODELS
    for model in models:
        delay = 2
        for attempt in range(per_model_retries):
            try:
                return client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except Exception as exc:
                last_exc = exc
                text = str(exc)
                if "RESOURCE_EXHAUSTED" in text or "429" in text:
                    _exhausted.add(model)  # daily cap won't reset today; stop trying it
                    print(f"  ({model}: daily cap reached, skipping it from now on...)")
                    break
                if any(c in text for c in ("500", "503", "UNAVAILABLE")) and attempt < per_model_retries - 1:
                    print(f"  ({model}: throttled, retry in {delay}s...)")
                    time.sleep(delay)
                    delay *= 2
                    continue
                print(f"  ({model}: failed [{text[:50]}], trying next model...)")
                break
    raise last_exc


def ask(client, contents, config):
    """Sends the conversation to Gemini, resolving any tool calls, returns reply text."""
    while True:
        response = generate(client, contents, config)
        contents.append(response.candidates[0].content)
        calls = response.function_calls
        if not calls:
            return (response.text or "").strip(), contents
        tool_parts = []
        for call in calls:
            args = dict(call.args or {})
            result = dispatch(call.name, args)
            print(f"  [tool] {call.name}({args}) -> {str(result)[:160]}")
            tool_parts.append(
                types.Part.from_function_response(name=call.name, response=result)
            )
        contents.append(types.Content(role="user", parts=tool_parts))


def speak(text):
    """Synthesizes Slovak speech with edge-tts and plays it."""
    if not text:
        return
    asyncio.run(edge_tts.Communicate(text, TTS_VOICE).save(TTS_FILE))
    decoded = miniaudio.mp3_read_file_s16(TTS_FILE)
    stream = pya.open(
        format=pyaudio.paInt16, channels=decoded.nchannels,
        rate=decoded.sample_rate, output=True,
    )
    try:
        stream.write(decoded.samples.tobytes())
    finally:
        stream.close()


def main():
    wake_model = load_wake_model()
    print(f"Loading Whisper ({WHISPER_MODEL_SIZE})...")
    whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    client = genai.Client()
    print(f'Smart speaker ready (free mode). Waiting for wake word ("{WAKE_PHRASES[0]}")...')
    try:
        while True:
            try:
                wait_for_wake_word(wake_model)
                print("\nWake word detected.")
                play_beep()
                contents = []
                config = make_config()
                last_active = time.time()
                while time.time() - last_active <= CONVERSATION_TIMEOUT:
                    print("Listening — speak now...")
                    audio = record_utterance()
                    if audio is None:
                        break
                    text = transcribe(whisper, audio)
                    if not text:
                        print("  (empty transcription)")
                        continue
                    print(f"\033[3mYou: {text}\033[0m")
                    contents.append(
                        types.Content(role="user", parts=[types.Part(text=text)])
                    )
                    reply, contents = ask(client, contents, config)
                    print(f"Gin: {reply}")
                    speak(reply)
                    last_active = time.time()
                print("Returning to sleep.\n")
            except Exception as exc:
                print(f"\nError, recovering: {exc}")
                time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        pya.terminate()
        print("\nShut down.")


if __name__ == "__main__":
    main()
