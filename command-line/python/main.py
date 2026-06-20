import asyncio
import datetime
import json
import math
import os
import struct
from google import genai
from google.genai import types
import pyaudio
from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(-1)  # silence Vosk/Kaldi startup logging

client = genai.Client()

# --- pyaudio config ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# --- Wake word config ---
# Wake-word detection runs fully offline with Vosk — no account, key, or payment.
# Download a small English model (~40 MB), unzip it, and point VOSK_MODEL_PATH at
# the folder: https://alphacephei.com/vosk/models (e.g. vosk-model-small-en-us-0.15).
# WAKE_PHRASES are the spoken phrases that activate the assistant; add homophones
# (e.g. "hey jean") if Vosk mishears your pronunciation of "gin".
VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "model")
WAKE_PHRASES = ["hey gin", "hey jean", "hey gene"]

# Return to wake-word listening after this many seconds without speech.
CONVERSATION_TIMEOUT = 15

pya = pyaudio.PyAudio()

# --- Live API config ---
# Custom functions the model can call. Add new connectors (Gmail, etc.) here and
# implement them in dispatch_function().
TOOL_DECLARATIONS = [
    {
        "name": "get_current_time",
        "description": "Get the current local date and time. Use when the user asks what time or what day it is.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "set_timer",
        "description": "Set a countdown timer or alarm that beeps when it finishes. Use for kitchen timers or reminders after a delay.",
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "Timer duration in seconds.",
                },
                "label": {
                    "type": "string",
                    "description": "Optional short description of what the timer is for.",
                },
            },
            "required": ["seconds"],
        },
    },
]

MODEL = "gemini-3.1-flash-live-preview"
CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": "Si nápomocný a priateľský domáci hlasový asistent. Vždy odpovedaj po slovensky, stručne, jasne a prirodzene. Nikdy neodpovedaj po anglicky, ani keď používateľ použije cudzie slovo. Hovor zdvorilo a používaj jednoduché vety vhodné pre staršieho používateľa. Ak sa ťa používateľ spýta na aktuálne informácie ako počasie, správy, čas alebo udalosti, vyhľadaj odpoveď na internete.",
    "tools": [
        {"google_search": {}},
        {"function_declarations": TOOL_DECLARATIONS},
    ],
    "output_audio_transcription": {},
    "input_audio_transcription": {},
}

audio_queue_output = asyncio.Queue()
audio_queue_mic = asyncio.Queue(maxsize=5)
last_activity = 0.0
active_timers = set()


def _now():
    return asyncio.get_running_loop().time()


def _drain(queue):
    while not queue.empty():
        queue.get_nowait()


def play_beep(freq=880, duration=0.15):
    """Plays a short tone so the user knows the wake word registered."""
    n = int(RECEIVE_SAMPLE_RATE * duration)
    fade = int(RECEIVE_SAMPLE_RATE * 0.01)
    samples = bytearray()
    for i in range(n):
        env = min(1.0, i / fade, (n - i) / fade)  # fade in/out to avoid clicks
        value = int(0.3 * 32767 * env * math.sin(2 * math.pi * freq * i / RECEIVE_SAMPLE_RATE))
        samples += struct.pack("<h", value)
    stream = pya.open(
        format=FORMAT, channels=CHANNELS, rate=RECEIVE_SAMPLE_RATE, output=True
    )
    try:
        stream.write(bytes(samples))
    finally:
        stream.close()


async def _run_timer(seconds, label):
    """Sleeps for the timer, then beeps an alarm even if back in wake-word mode."""
    await asyncio.sleep(seconds)
    suffix = f" ({label})" if label else ""
    print(f"\n⏰ Časovač{suffix} dozvonil.")
    for _ in range(3):
        await asyncio.to_thread(play_beep, 1046, 0.2)
        await asyncio.sleep(0.15)


def start_timer(seconds, label=None):
    """Schedules a timer as a standalone task that outlives the conversation."""
    task = asyncio.create_task(_run_timer(seconds, label))
    active_timers.add(task)
    task.add_done_callback(active_timers.discard)


async def dispatch_function(name, args):
    """Runs a tool the model asked for and returns a JSON-serializable result."""
    if name == "get_current_time":
        now = datetime.datetime.now()
        return {
            "time": now.strftime("%H:%M"),
            "date": now.strftime("%Y-%m-%d"),
            "weekday": now.strftime("%A"),
        }
    if name == "set_timer":
        try:
            seconds = int(args.get("seconds"))
        except (TypeError, ValueError):
            return {"status": "error", "message": "Invalid duration."}
        if seconds <= 0:
            return {"status": "error", "message": "Duration must be positive."}
        start_timer(seconds, args.get("label"))
        return {"status": "ok", "seconds": seconds, "label": args.get("label")}
    return {"status": "error", "message": f"Unknown function: {name}"}


async def handle_tool_call(session, tool_call):
    """Executes the model's function calls and sends their results back."""
    responses = []
    for fc in tool_call.function_calls:
        result = await dispatch_function(fc.name, fc.args or {})
        responses.append(
            types.FunctionResponse(id=fc.id, name=fc.name, response=result)
        )
    await session.send_tool_response(function_responses=responses)


def load_wake_model():
    """Loads the offline Vosk model used for local wake-word detection."""
    if not os.path.isdir(VOSK_MODEL_PATH):
        raise RuntimeError(
            f"Vosk model not found at '{VOSK_MODEL_PATH}'. Download one from "
            "https://alphacephei.com/vosk/models, unzip it, and set VOSK_MODEL_PATH."
        )
    return Model(VOSK_MODEL_PATH)


def wait_for_wake_word(model):
    """Blocks on the mic locally until a wake phrase is heard (no network use)."""
    # Restrict recognition to the wake phrases (+ "[unk]") for speed and accuracy.
    grammar = json.dumps(WAKE_PHRASES + ["[unk]"])
    recognizer = KaldiRecognizer(model, SEND_SAMPLE_RATE, grammar)
    stream = pya.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SEND_SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )
    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                text = json.loads(recognizer.Result()).get("text", "")
            else:
                text = json.loads(recognizer.PartialResult()).get("partial", "")
            if any(phrase in text for phrase in WAKE_PHRASES):
                return
    finally:
        stream.close()


async def listen_audio():
    """Listens for audio and puts it into the mic audio queue."""
    mic_info = pya.get_default_input_device_info()
    stream = await asyncio.to_thread(
        pya.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=SEND_SAMPLE_RATE,
        input=True,
        input_device_index=mic_info["index"],
        frames_per_buffer=CHUNK_SIZE,
    )
    try:
        kwargs = {"exception_on_overflow": False} if __debug__ else {}
        while True:
            data = await asyncio.to_thread(stream.read, CHUNK_SIZE, **kwargs)
            await audio_queue_mic.put({"data": data, "mime_type": "audio/pcm"})
    finally:
        stream.close()


async def send_realtime(session):
    """Sends audio from the mic audio queue to the GenAI session."""
    while True:
        msg = await audio_queue_mic.get()
        await session.send_realtime_input(audio=msg)


async def receive_audio(session):
    """Receives responses from GenAI and puts audio data into the speaker audio queue."""
    global last_activity
    last_was_input = False
    while True:
        turn = session.receive()
        async for response in turn:
            if response.tool_call:
                last_activity = _now()
                await handle_tool_call(session, response.tool_call)
                continue
            sc = response.server_content
            if not sc:
                continue
            last_activity = _now()
            if sc.model_turn:
                for part in sc.model_turn.parts:
                    if part.inline_data and isinstance(part.inline_data.data, bytes):
                        audio_queue_output.put_nowait(part.inline_data.data)
            if sc.output_transcription:
                if last_was_input:
                    print()
                    last_was_input = False
                t = sc.output_transcription.text
                print(t, end="", flush=True)
                if t.rstrip()[-1:] in '.!?':
                    print()
            if sc.input_transcription:
                if not last_was_input:
                    print()
                    last_was_input = True
                t = sc.input_transcription.text
                print(f"\033[3m{t}\033[0m", end="", flush=True)
                if t.rstrip()[-1:] in '.!?':
                    print()

        # Empty the queue on interruption to stop playback
        while not audio_queue_output.empty():
            audio_queue_output.get_nowait()


async def play_audio():
    """Plays audio from the speaker audio queue."""
    global last_activity
    stream = await asyncio.to_thread(
        pya.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=RECEIVE_SAMPLE_RATE,
        output=True,
    )
    try:
        while True:
            bytestream = await audio_queue_output.get()
            last_activity = _now()
            await asyncio.to_thread(stream.write, bytestream)
    finally:
        stream.close()


async def converse(session):
    """Runs one conversation, returning to wake-word mode after a silent pause."""
    global last_activity
    _drain(audio_queue_mic)
    _drain(audio_queue_output)
    last_activity = _now()
    tasks = [
        asyncio.create_task(send_realtime(session)),
        asyncio.create_task(listen_audio()),
        asyncio.create_task(receive_audio(session)),
        asyncio.create_task(play_audio()),
    ]
    try:
        while _now() - last_activity <= CONVERSATION_TIMEOUT:
            await asyncio.sleep(0.5)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def run():
    """Waits for the wake word, then opens a Gemini conversation on demand."""
    wake_model = load_wake_model()
    print(f'Smart speaker ready. Waiting for wake word ("{WAKE_PHRASES[0]}")...')
    try:
        while True:
            await asyncio.to_thread(wait_for_wake_word, wake_model)
            await asyncio.to_thread(play_beep)
            print("\nWake word detected — connecting to Gemini...")
            async with client.aio.live.connect(
                model=MODEL, config=CONFIG
            ) as live_session:
                print("Connected. Start speaking!")
                await converse(live_session)
            print("Returning to sleep.\n")
    except asyncio.CancelledError:
        pass
    finally:
        pya.terminate()
        print("\nShut down.")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Interrupted by user.")
