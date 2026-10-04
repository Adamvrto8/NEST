# NEST — Slovak voice smart speaker

A hands-free, Slovak-language voice assistant for a Raspberry Pi 4 (or any computer with a mic and a speaker). It idles until it hears a wake word, listens, and answers out loud in Slovak.

It can search the web, read and draft Gmail, read and create Google Calendar events, set timers, and give a personalized morning brief with weather, news, calendar and unread mail.

## How it works

```
mic → wake word (Vosk, offline) → speech in → model with tools → speech out → speaker
                                                    │
                          timers · Gmail · Calendar · daily brief · web search
```

The assistant comes in three interchangeable variants that share the wake word, the tools and the Slovak voice, but use a different model:

| Variant | File | Branch | Model | Speech in → out | Cost |
| --- | --- | --- | --- | --- | --- |
| Live | `main.py` | `main` | Gemini Live (real-time audio) | Gemini native | Paid, needs Google Cloud billing |
| Free | `main_local.py` | `free-pipeline` | Gemini free text tier | Whisper (local) → edge-tts | Free |
| Claude | `main_claude.py` | `claude-pipeline` | Claude (Anthropic) | Whisper (local) → edge-tts | Paid, cheap on Haiku |

## What I built

Everything for the speaker lives in [`command-line/python`](command-line/python):

- Offline wake word with Vosk, including a wake beep and homophone handling
- A function-calling layer with timer, time and date tools
- Gmail tools (read and draft, never send) and Google Calendar tools (read and create) behind one OAuth helper
- A daily brief skill that combines weather (Open-Meteo), news headlines (RSS), today's calendar and unread mail, all returned in Slovak
- Three model back ends behind the same tools: Gemini Live, Whisper + Gemini text, Whisper + Claude
- A systemd unit and automatic recovery from network and session errors, so it can run 24/7 on a Pi

## Run it

Setup, API keys, the wake-word model and the Raspberry Pi service are described step by step in the [speaker README](command-line/python/README.md).

```bash
cd command-line/python
pip install -r requirements.txt
cp .env.example .env      # add your GEMINI_API_KEY
python main.py
```

## Credits

The project started from Google's Gemini Live API example code. The other folders in this repository (`gemini-live-ephemeral-tokens-websocket`, `gemini-live-genai-python-sdk`, `gemini-live-translate-livekit`, `command-line/node`) are those upstream samples, kept unchanged.
