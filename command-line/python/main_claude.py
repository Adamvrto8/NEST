"""Turn-based smart speaker with a Claude brain (paid, reliable).

Pipeline: wake word (Vosk) -> record until you pause -> Whisper (Slovak STT) ->
Claude Messages API with tool use + web search -> edge-tts (Slovak TTS). Needs
ANTHROPIC_API_KEY. Reuses the same tool backends as the other variants.

Run with the venv python:  python main_claude.py
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

import anthropic
import numpy as np
import pyaudio
from vosk import Model, KaldiRecognizer, SetLogLevel
from faster_whisper import WhisperModel
import edge_tts
import miniaudio

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

# --- wake word ---
VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "model")
WAKE_PHRASES = ["hey gin", "hey jean", "hey gene"]

# --- speech to text ---
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
WHISPER_LANGUAGE = "sk"

# --- text to speech ---
TTS_VOICE = os.environ.get("TTS_VOICE", "sk-SK-LukasNeural")
TTS_FILE = "tts_out.mp3"

# --- brain (Claude) ---
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5")
client = None  # created in main() from ANTHROPIC_API_KEY
SYSTEM_PROMPT = (
    "Si nápomocný a priateľský domáci hlasový asistent. Vždy odpovedaj po slovensky, "
    "stručne, jasne a prirodzene. Nikdy neodpovedaj po anglicky. Hovor zdvorilo a "
    "jednoducho, vhodne pre staršieho používateľa. Keď používateľ požiada o prehľad, "
    "zavolaj nástroj get_daily_brief a zhrň počasie, správy, kalendár a emaily vrúcne "
    "a stručne. Na aktuálne počasie použi get_weather, na správy get_news. Iné aktuálne "
    "informácie (napríklad šport) vyhľadaj na internete. Odpovedaj iba obyčajným textom, "
    "ktorý sa dá prečítať nahlas — žiadne odrážky ani formátovanie."
)

# --- turn-taking ---
SILENCE_RMS = int(os.environ.get("SILENCE_RMS", "500"))
SILENCE_SECONDS = 1.2
MAX_UTTERANCE = 12
NO_SPEECH_TIMEOUT = 6
CONVERSATION_TIMEOUT = 15

SLOVAK_WEEKDAYS = [
    "pondelok", "utorok", "streda", "štvrtok", "piatok", "sobota", "nedeľa",
]

# Custom (client-side) tools — same schemas as the other variants, in Claude's
# "input_schema" shape. The web_search server tool is added in TOOLS below.
TOOL_DECLARATIONS = [
    {
        "name": "get_daily_brief",
        "description": "Get the user's personalized daily brief in one call: today's weather, top news headlines, today's calendar events, and recent unread emails. Use when the user asks for a 'prehľad', a summary, or a morning briefing.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_current_time",
        "description": "Get the current time, date, and weekday. Use when the user asks what time or day it is.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_weather",
        "description": "Get the current weather and today's forecast for the user's location. Use when the user asks about the weather.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_news",
        "description": "Get the latest news headlines. Use when the user asks about the news.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_timer",
        "description": "Set a countdown timer that beeps when it finishes. Use for kitchen timers or reminders after a delay.",
        "input_schema": {
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
        "input_schema": {
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
        "input_schema": {
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
        "input_schema": {
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
        "input_schema": {
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

# Claude's built-in server-side web search (basic variant, works on Haiku).
TOOLS = TOOL_DECLARATIONS + [
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
]


def system_prompt():
    """System prompt with today's date injected so date/calendar questions
    don't need an extra get_current_time round-trip."""
    now = datetime.datetime.now()
    return (
        f"{SYSTEM_PROMPT} Dnešný dátum je {now.strftime('%Y-%m-%d')} "
        f"({SLOVAK_WEEKDAYS[now.weekday()]}), aktuálny čas je {now.strftime('%H:%M')}."
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
        time.sleep(0.15)
    finally:
        stream.close()


def dispatch(name, args):
    """Runs a client-side tool and returns a JSON-serializable dict."""
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
    preroll = deque(maxlen=8)
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
        audio = audio * min(4.0, 0.95 / peak_amp)
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


def respond(messages):
    """Sends the conversation to Claude, resolving tool calls, returns reply text."""
    while True:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=system_prompt(),
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = dispatch(block.name, dict(block.input))
                    print(f"  [tool] {block.name}({dict(block.input)}) -> {str(result)[:160]}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
            messages.append({"role": "user", "content": tool_results})
            continue
        if response.stop_reason == "pause_turn":
            continue  # server-side web search paused; resend to resume
        break
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text, messages


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
    global client
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    wake_model = load_wake_model()
    print(f"Loading Whisper ({WHISPER_MODEL_SIZE})...")
    whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    print(f'Smart speaker ready (Claude: {CLAUDE_MODEL}). Waiting for wake word ("{WAKE_PHRASES[0]}")...')
    try:
        while True:
            try:
                wait_for_wake_word(wake_model)
                print("\nWake word detected.")
                play_beep()
                messages = []
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
                    messages.append({"role": "user", "content": text})
                    reply, messages = respond(messages)
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
