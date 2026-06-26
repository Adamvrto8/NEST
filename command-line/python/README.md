# NEST — Slovak voice smart speaker

A hands-free, **Slovak-language** voice assistant for a Raspberry Pi 4 (or any
computer with a mic + speaker). It idles until it hears a wake word, listens,
and answers out loud in Slovak. It can search the web, read and draft Gmail,
read and create Google Calendar events, set timers, and give a personalized
morning brief.

> **Use headphones / a speaker without an open mic.** With the system default
> audio devices there's usually no echo cancellation, so the assistant can hear
> itself. Headphones avoid that.

## The three variants

The same speaker comes in **three interchangeable variants** — one per git
branch — that share the same wake word, tools, and Slovak voice but use a
different "brain":

| Variant | File | Branch | Brain | Speech in → out | Cost |
| --- | --- | --- | --- | --- | --- |
| **Live** | `main.py` | `main` | Gemini Live (real-time audio) | Gemini native | **Paid** — needs Google Cloud billing |
| **Free** | `main_local.py` | `free-pipeline` | Gemini free **text** tier | Whisper (local) → edge-tts | **Free** — $0, a little slower |
| **Claude** | `main_claude.py` | `claude-pipeline` | Claude (Anthropic) | Whisper (local) → edge-tts | **Paid** — Anthropic API (cheap on Haiku) |

> The **`claude-pipeline`** branch contains **all three** files, so you can try
> each without switching branches. `main` has only `main.py`; `free-pipeline`
> has `main.py` + `main_local.py`. To use a variant whose file isn't on your
> current branch, `git switch` to its branch from the table above.

Everything the speaker says or prints is **Slovak**, including the data the tools
return (weather conditions, weekday names, etc.).

### Which one should I use?

- **Want it working for free, no billing?** → **Free** (`main_local.py`). Runs on
  the Gemini free tier; turn-based (speak, pause, it replies).
- **Want the most natural, real-time voice and willing to enable billing?** →
  **Live** (`main.py`).
- **Want a reliable paid brain that isn't Gemini?** → **Claude**
  (`main_claude.py`); very cheap on Haiku.

## Setup

### 1. Python environment

```powershell
# Windows (PowerShell) — development machine
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / Raspberry Pi
python3 -m venv .venv
source .venv/bin/activate
```

On Linux/macOS you also need PortAudio for the mic (`sudo apt install portaudio19-dev`
on the Pi, `brew install portaudio` on macOS).

### 2. Dependencies (install for the variant you want)

```bash
# Live   (main.py)
pip install -r requirements.txt

# Free   (main_local.py)  — adds local Whisper STT + edge-tts
pip install -r requirements.txt -r requirements-local.txt

# Claude (main_claude.py) — adds the Anthropic SDK on top of the Free deps
pip install -r requirements.txt -r requirements-local.txt -r requirements-claude.txt
```

> **Windows tip:** if `python` points at the wrong interpreter, call the venv
> directly: `.\.venv\Scripts\python.exe …` instead of `python …`.

### 3. API keys — `.env`

Copy the template and paste your keys once; you won't have to set them in the
terminal on every run.

```bash
cp .env.example .env
```

Then edit `.env`:

```ini
GEMINI_API_KEY=...      # free at https://aistudio.google.com/apikey   (Live + Free)
ANTHROPIC_API_KEY=...   # https://console.anthropic.com/settings/keys  (Claude only)
```

`.env` is git-ignored, so your keys are never committed. The scripts load it
automatically on startup. A real shell environment variable still takes
precedence over `.env`.

### 4. Wake-word model

The assistant listens for the wake word **fully offline** with
[Vosk](https://alphacephei.com/vosk/) — no account, key, or payment.

1. Download a small English model (~40 MB) from
   [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models), e.g.
   `vosk-model-small-en-us-0.15`.
2. Unzip it and either rename the folder to `model` (the default) or point
   `VOSK_MODEL_PATH` at it.

The wake phrase is **"Hey Gin"** — edit `WAKE_PHRASES` in whichever `main*.py`
you run to change it or add homophones (e.g. `"hey jean"`) if Vosk mishears you.

```bash
export VOSK_MODEL_PATH="model"   # optional; defaults to ./model
```

## Run each variant

With the venv active and the wake-word model in place:

```bash
python main.py          # Live   — real-time Gemini audio (needs billing)
python main_local.py    # Free   — Whisper + Gemini free text + edge-tts ($0)
python main_claude.py   # Claude — Whisper + Claude + edge-tts (needs ANTHROPIC_API_KEY)
```

On Windows without the venv activated, use the explicit path, e.g.
`.\.venv\Scripts\python.exe main_local.py`.

You should see **"Smart speaker ready. Waiting for wake word…"** Say the wake
word, then talk. Press `Ctrl+C` to quit. All variants recover from network or
session errors on their own and return to wake-word listening.

## Tools

All variants share the same tools:

- **Timers** and **time / date** — local, no setup.
- **Gmail (read & draft)** — reads recent mail and creates drafts; it never sends.
- **Calendar (read & create)** — lists upcoming events and adds appointments.
- **Daily brief** — say *"chcem prehľad"* for a personalized morning briefing:
  today's weather (Open-Meteo), top news headlines (RSS), today's calendar, and
  unread mail — all in Slovak.
- **Web search** — availability depends on the brain:
  - **Live** (`main.py`) — built-in Google Search grounding.
  - **Claude** (`main_claude.py`) — Anthropic web-search tool.
  - **Free** (`main_local.py`) — no open web search on the free tier, but it
    still gets news/weather through the daily-brief tools above.

### Daily brief settings

Configure with environment variables (all optional; can also go in `.env`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `BRIEF_USER_NAME` | Lubo | Name the assistant greets in the brief |
| `BRIEF_LATITUDE` / `BRIEF_LONGITUDE` | Bratislava | Location for weather (Open-Meteo, no key) |
| `BRIEF_LOCATION_NAME` | Bratislava | Place name spoken in the brief |
| `BRIEF_NEWS_FEED` | SME (sme.sk) | RSS feed URL for headlines — point at any site's RSS |
| `BRIEF_NEWS_COUNT` / `BRIEF_MAIL_COUNT` | 5 | How many headlines / unread emails to include |

Weather and news need no setup; calendar and mail use the Google auth below.

### Google setup (optional, for Gmail & Calendar)

1. In [Google Cloud](https://console.cloud.google.com/), create a project and
   enable the **Gmail API** and the **Google Calendar API**. (These APIs are free
   and do **not** require billing.)
2. Configure the OAuth consent screen as **External**, add your account under
   **Test users**, and (optionally) **Publish** the app so the token doesn't
   expire every 7 days.
3. Create an **OAuth client** of type **Desktop app**, download its JSON, and
   save it as `credentials.json` next to `main.py`.
4. Authorize once (opens a browser, writes `token.json` for all scopes):

   ```bash
   python google_auth_helper.py
   ```

   On a headless Raspberry Pi, run this on a machine with a browser and copy the
   resulting `token.json` over. Set `CALENDAR_TIMEZONE` (e.g.
   `Europe/Bratislava`) if you're not in the default zone.

## Run on boot (Raspberry Pi)

Use the included `smart-speaker.service` so the app starts on boot and restarts
if it crashes. Edit `User`, the paths, and the `ExecStart` line to point at the
variant you want (it defaults to `main.py` — change it to `main_local.py` or
`main_claude.py` as needed), then:

```bash
# Keys can live in the project .env (loaded automatically) OR in this file:
echo 'GEMINI_API_KEY=your-key' | sudo tee /etc/default/smart-speaker
sudo cp smart-speaker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now smart-speaker
journalctl -u smart-speaker -f   # follow logs
```

---

## Bonus: real-time stream translation (`translate.py`)

Unrelated to the speaker — a leftover upstream sample that translates any remote
audio stream URL in real time (uses the Gemini Live API).

```bash
python translate.py --target sk
```

- `--url`: audio stream URL to translate (defaults to a sample WAV).
- `--target`: target language code (`sk`, `es`, `fr`, …). Defaults to `es`.
- `--original-volume`: background volume of the original speaker (`0.0`–`1.0`,
  default `0.08`). Set to `0.0` to mute it.
