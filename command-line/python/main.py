import asyncio
import os
import struct
from google import genai
import pyaudio
import pvporcupine

client = genai.Client()

# --- pyaudio config ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# --- Wake word config ---
# "Hey Gin" is a custom phrase, so it needs a free model file (.ppn) generated
# at https://console.picovoice.ai (choose "Raspberry Pi" as the platform). Point
# WAKE_KEYWORD_PATH at that file. Leave it unset to fall back to the built-in
# keyword in WAKE_BUILTIN so you can test before generating the custom file.
PV_ACCESS_KEY = os.environ.get("PV_ACCESS_KEY")
WAKE_KEYWORD_PATH = os.environ.get("WAKE_KEYWORD_PATH")
WAKE_BUILTIN = "jarvis"

# Return to wake-word listening after this many seconds without speech.
CONVERSATION_TIMEOUT = 15

pya = pyaudio.PyAudio()

# --- Live API config ---
MODEL = "gemini-3.1-flash-live-preview"
CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": "Si nápomocný a priateľský domáci hlasový asistent. Vždy odpovedaj po slovensky, stručne, jasne a prirodzene. Nikdy neodpovedaj po anglicky, ani keď používateľ použije cudzie slovo. Hovor zdvorilo a používaj jednoduché vety vhodné pre staršieho používateľa ",
    "output_audio_transcription": {},
    "input_audio_transcription": {},
}

audio_queue_output = asyncio.Queue()
audio_queue_mic = asyncio.Queue(maxsize=5)
last_activity = 0.0


def _now():
    return asyncio.get_running_loop().time()


def _drain(queue):
    while not queue.empty():
        queue.get_nowait()


def create_porcupine():
    """Creates the Porcupine wake-word detector from the configured keyword."""
    if not PV_ACCESS_KEY:
        raise RuntimeError(
            "Set PV_ACCESS_KEY — get a free key at https://console.picovoice.ai"
        )
    if WAKE_KEYWORD_PATH:
        return pvporcupine.create(
            access_key=PV_ACCESS_KEY, keyword_paths=[WAKE_KEYWORD_PATH]
        )
    return pvporcupine.create(access_key=PV_ACCESS_KEY, keywords=[WAKE_BUILTIN])


def wait_for_wake_word(porcupine):
    """Blocks on the mic locally until the wake word is heard (no network use)."""
    stream = pya.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=porcupine.sample_rate,
        input=True,
        frames_per_buffer=porcupine.frame_length,
    )
    try:
        while True:
            data = stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("%dh" % porcupine.frame_length, data)
            if porcupine.process(pcm) >= 0:
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
    porcupine = create_porcupine()
    label = WAKE_KEYWORD_PATH or f'"{WAKE_BUILTIN}"'
    print(f"Smart speaker ready. Waiting for wake word ({label})...")
    try:
        while True:
            await asyncio.to_thread(wait_for_wake_word, porcupine)
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
        porcupine.delete()
        pya.terminate()
        print("\nShut down.")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Interrupted by user.")
