# EchoLingo

[简体中文](README.md) | English

EchoLingo is a real-time lecture interpreter. It turns what the lecturer says into side-by-side captions in the original language and your language, sentence by sentence, and after class an AI model of your choice can turn the transcript into study notes. By default, all recognition and translation run on your own computer. The published download is for the Mac; Windows and Linux versions are in testing.

[![Platform](https://img.shields.io/badge/platform-macOS%2014%2B%20%C2%B7%20Apple%20Silicon-lightgrey)](#requirements)
[![Version](https://img.shields.io/badge/version-0.3.0%20Beta-orange)](https://github.com/Asphr726/EchoLingo/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

![EchoLingo live bilingual captions](docs/screenshots/live.png)

> **v0.3.0 Beta**: the published download is for Apple Silicon Macs running macOS 14 or later. Windows and Linux builds are in testing and not published yet; see [Windows / Linux](#windows--linux-in-testing-not-yet-published). The app's interface is in English.

[Highlights](#highlights) · [Requirements](#requirements) · [Install](#install) · [Windows / Linux](#windows--linux-in-testing-not-yet-published) · [Quick start](#quick-start) · [Guide](#guide) · [Privacy](#privacy) · [FAQ](#faq) · [Roadmap](#roadmap) · [Build from source](#build-from-source) · [License and acknowledgements](#license-and-acknowledgements)

## Highlights

- **Local-first and private.** Speech recognition (Qwen3-ASR) and translation (Hy-MT2) run on your computer by default. No account is needed, and once the models are downloaded everything works offline. Audio or text leaves the computer only if you turn on a cloud provider, and every kind of upload needs your explicit consent first. API keys are stored in the system secure store (macOS Keychain, Windows Credential Manager, or Secret Service on Linux).
- **Designed for far-field lecture audio.** The default Lecture / far-field audio profile runs noise suppression and automatic gain control (AGC) on your computer. Speech detection only labels the audio and never throws any of it away. A sentence is finalized only after a pause confirms that it has ended, which keeps sentences from being cut in half.
- **Side-by-side bilingual transcript and a floating caption window.** The main window shows the original on the left and the translation on the right, sentence by sentence. The floating caption window stays on top of your slides, PDF or note-taking app, with adjustable font size, opacity and content.
- **Lecture context.** Before class, add the topic, names and terms (write `term = translation` for a fixed translation), or import them from the slides. Recognition and translation both use this context, so names and terms come out right more often. Terms you use every week go into a standing glossary.
- **AI notes.** The AI notes feature turns a messy spoken transcript into structured study notes, written in your translation language, with formulas rendered from LaTeX. You can attach course materials (PDF, PPTX, DOCX, Markdown, LaTeX, plain text). The notes are saved with the session, and each section carries a time range that jumps back to that part of the transcript. Notes are written by a chat model you choose: a cloud provider with your own key, or a local server such as Ollama.
- **Automatic session titles.** Once the AI assistant is set up, it names the session when you press **Stop**. Titles you rename yourself are never replaced.
- **Pluggable cloud providers.** When you want them, cloud recognition and cloud translation are available. You choose each one separately, so for example you can combine local recognition with cloud translation.
  - Recognition: Qwen Cloud (Alibaba Model Studio), OpenAI Realtime, Deepgram, AssemblyAI (English only), Gladia.
  - Translation: Qwen-MT, OpenAI-compatible chat models (OpenAI, DeepSeek, Google Gemini, Groq, OpenRouter, SiliconFlow, and custom endpoints such as Ollama, LM Studio or vLLM), DeepL, Google Cloud Translation, Azure AI Translator.
  - Qwen Cloud's Beijing region gives new accounts free quota (90 days per model).
- **Searchable history with export.** Search by title or transcript text, rename and delete sessions, and export them as Markdown, TXT, JSON, SRT or VTT subtitles, or export the AI notes on their own.
- **Languages:** English, Chinese, Japanese and Korean, translated between any two of them.

## Requirements

| Item | macOS (published) | Windows / Linux (in testing) |
| --- | --- | --- |
| System | macOS 14 Sonoma or later | Windows 10 21H2 or later, or Windows 11 (x64); Ubuntu 22.04 or later, or Debian 12 or later (x64) |
| Chip | Apple Silicon (M1 or later); Intel Macs are not supported | An x64 processor with AVX2; optionally an NVIDIA graphics card for the GPU acceleration pack |
| Memory | 16 GB or more recommended | 16 GB or more recommended |
| Disk | The local models take about 4.5 GB; keep at least 6 GB free | The same; the optional GPU acceleration pack is another 2–3 GB download and takes more space once unpacked |
| Permissions | Microphone; to capture sound playing on the Mac (online classes, videos), also Screen & System Audio Recording | Windows: desktop apps must be allowed to use the microphone; Linux: none |
| Network | Only for downloading models, update checks, cloud providers and AI notes | The same |

On Windows and Linux, recognition and translation run on the CPU unless the GPU acceleration pack is installed. On slower processors the captions may fall behind the lecture; the [GPU acceleration pack](#windows--linux-in-testing-not-yet-published) or a cloud recognizer solves that.

## Install

The steps below are for macOS. For Windows and Linux, see [Windows / Linux](#windows--linux-in-testing-not-yet-published).

1. Download `EchoLingo_0.3.0_aarch64.dmg` from [GitHub Releases](https://github.com/Asphr726/EchoLingo/releases).
2. Open the dmg and drag **EchoLingo** into the Applications folder.
3. Open it for the first time. The Beta is not notarized by Apple yet, so macOS blocks it at first:
   - **macOS 14**: in Applications, Control-click (or right-click) EchoLingo, choose **Open**, then choose **Open** again in the dialog.
   - **macOS 15 and later**: double-click the app once and choose **Done** when the warning appears. Then open **System Settings → Privacy & Security**, scroll to the bottom, click **Open Anyway** next to EchoLingo and confirm with your password.
   - If macOS still says the app is damaged or cannot be opened, run this command in Terminal and open the app again:

     ```bash
     xattr -dr com.apple.quarantine /Applications/EchoLingo.app
     ```

4. Download the local models. After the first-run setup, open **Settings → Models** and click **Download** next to each model in **Local model manager**:

   | Model | Purpose | Size |
   | --- | --- | --- |
   | Qwen3-ASR 0.6B | Speech recognition | 1.7 GB |
   | Hy-MT2 1.8B (Q4_K_M) | Translation | 1.1 GB |
   | Qwen3 ForcedAligner 0.6B | Word-accurate timestamps after class (optional, recommended) | 1.7 GB |

   Each row shows its download progress, and EchoLingo checks the files automatically when a download finishes. You can **Verify** or **Delete** a downloaded model at any time. If a download fails or is interrupted, click **Download** again; if the files of a downloaded model are incomplete, the row shows **Retry**. Models come from Hugging Face and are downloaded only once.

   > **Users in mainland China**: if the download is slow or keeps failing, try a Hugging Face mirror. Run
   > `launchctl setenv HF_ENDPOINT https://hf-mirror.com` in Terminal, quit EchoLingo completely with ⌘Q, reopen it and click **Download** again.
   > The setting lasts until you log out or restart the Mac. To go back to the official source, run `launchctl unsetenv HF_ENDPOINT` and reopen the app.

## Windows / Linux (in testing, not yet published)

> **Not published yet.** Windows and Linux builds exist, but they have not been tested on real Windows or Linux computers yet, so they are not on the Releases page. The steps below are for testers who received a build; the app and its settings work as described in the rest of this guide unless noted here.

**Windows 10 / 11 (x64)**

1. Run `EchoLingo_0.3.0_x64-setup.exe`. It installs for your user account only and needs no administrator rights; if the Microsoft Edge WebView2 runtime is missing, the installer adds it.
2. The installer is not code-signed yet, so Windows SmartScreen may say "Windows protected your PC". Click **More info**, then **Run anyway**.
3. Open EchoLingo from the Start menu and download the local models in **Settings → Models**, as in step 4 of [Install](#install).

**Linux (Ubuntu 22.04+ / Debian 12+, x64)**

1. In the folder you downloaded it to, install the package with `sudo apt install ./EchoLingo_0.3.0_amd64.deb`.
2. Open EchoLingo from the application menu, or run `echolingo-desktop` in a terminal. Only a .deb package is available for now; an AppImage is on the roadmap.
3. Open EchoLingo from the applications menu and download the local models in **Settings → Models**.

**NVIDIA GPU acceleration pack (optional).** Without it, recognition and translation run on the CPU. If the computer has an NVIDIA GeForce RTX 20-series or GTX 16-series card or newer with a recent driver, open **Settings → Models → GPU acceleration** and click **Download**. The pack (about 2–3 GB) is downloaded in parts, every part is checked, and a self-test runs before the pack is switched on; **Use GPU acceleration** turns it off again at any time. If the GPU runtime fails to start, EchoLingo falls back to the CPU and the card says why. AMD and Intel graphics cards are not supported yet. On Linux, GPU translation also needs the Vulkan loader (`sudo apt install libvulkan1`); without it translation stays on the CPU.

**Differences from the Mac version**

- **System audio**: on Windows, **System audio** captures whatever the default output device (speakers or headphones) plays; there is no picker. On Linux, system audio capture is not available yet, so use a microphone.
- **Keys** are stored in the Windows Credential Manager, or on Linux in a Secret Service provider such as GNOME Keyring or KWallet. If none can be reached, **Settings → Cloud providers** says so, and keys can still come from environment variables such as `DEEPL_API_KEY`.
- **Hugging Face mirror**: on Windows, run `setx HF_ENDPOINT https://hf-mirror.com` in Command Prompt or PowerShell, then quit and reopen EchoLingo (`reg delete HKCU\Environment /v HF_ENDPOINT /f` goes back to the official source). On Linux, add `export HF_ENDPOINT=https://hf-mirror.com` to `~/.profile` and log in again, or start EchoLingo from a terminal where the variable is set.
- **Floating caption**: the opacity setting is available on macOS only.

## Quick start

1. **First-run setup.** On first launch, a short wizard walks you through seven steps: Welcome, Languages, Audio (source and profile), Inference mode, Permissions, Audio test and Ready. You can change every choice later in Settings.
2. **Download the models.** Follow step 4 of [Install](#install) in **Settings → Models**.
3. **Start the lecture.** On the **Live** screen, check the source language (the language the lecturer speaks) and the target language (the language you want to read), then press **Start**. The first time, loading the models takes 1–2 minutes. After that, EchoLingo loads the models in the background when it launches, so Start is immediate.
4. **Open the floating caption.** Click **Floating caption** and drag the small caption window next to the slides.
5. **End the lecture.** Press **Stop**. The session is saved to **History** automatically and, if the AI assistant is set up with automatic titles on, gets a title right away.

![First-run setup](docs/screenshots/onboarding.png)

## Guide

### Live

![Lecture context panel](docs/screenshots/lecture-context.png)

- **Source language / Target language**: any two of English, Chinese, Japanese and Korean.
- **Audio source**: **Microphone** for a lecture in the room, or **System audio** for sound playing on the computer (online classes, recorded videos, meetings). With System audio, macOS asks you to choose the display or app to capture each time a session starts; Windows captures the default output device; Linux does not offer System audio yet.
- **Audio profile**: keep **Lecture / far-field** (the default) for classrooms. Choose **Conversation** for close-up, face-to-face speech. **Raw diagnostics** turns off all audio processing and is meant for troubleshooting only.
- **Input device**: the microphone to use.
- **Inference mode**:
  - **Local** uses only the models on your computer.
  - **Cloud** uses cloud providers.
  - **Auto** uses the local models by default. When you press Start, it uses the preferred cloud provider set in **Settings → Cloud providers** instead only if the local models cannot keep up in real time on this computer and you have allowed the matching upload. Auto never turns on an upload permission for you.
- The **Inference** panel on the right lets you pick the recognition provider (ASR provider) and the translation provider separately. **Audio / VAD** shows the input level and whether speech is detected. **Latency** shows how long each step takes. The card at the bottom says whether this session's audio and text leave the computer; in Auto mode or on a cloud route it also has the **Allow audio upload** and **Allow transcript upload** switches.
- **Start / Pause / Resume / Stop** control the session.
- In the transcript, the original is on the left and the translation on the right. The bottom row marked **LIVE** is the sentence still being recognized; its lighter words may still change. When a sentence is final it stays fixed, and its translation follows. Scrolling up pauses auto-scroll and keeps your place while new lines arrive; **Back to live** returns to the newest line.

**Floating caption**: **Floating caption** on the Live screen (or **Open caption** in **Settings → General**) opens a small always-on-top window that you can move and resize. It shows the last 1–3 sentences and the one being spoken. In **Settings → Appearance** you can show the original and the translation, the original only or the translation only, and set the font size (18–72 px), the opacity (macOS only) and the number of sentences.

![Floating caption window](docs/screenshots/caption.png)

**Lecture context**: before class, open this panel and write one entry per line: the topic, names and terms. A line like `pre-attentive vision = 前注意视觉` becomes a glossary pair for this lecture. The limit is 2,000 characters.

- **Import from slides…** pulls terms out of course materials (PDF, PPTX, DOCX and more). Without an AI assistant, the text is only extracted on your computer. If the assistant is set up but not yet allowed to receive text, EchoLingo first asks whether it may read the slide text to pick out terms and translations; choose **Extract on this computer** to keep it local. Once the assistant is allowed, slide text goes to it directly, and the hint next to the button says so.
- While a session runs, the context is read-only.
- Put terms you need in every lecture into **Settings → Translation → Glossary**: one entry per line, lines starting with `#` are ignored, and the limit is 4,000 characters. Changes apply from the next session.
- With local models, the context never leaves your computer. With cloud providers, it is sent along with the audio or text only when the matching upload is allowed.

### History and AI notes

![AI notes in History](docs/screenshots/history-notes.png)

- **Search and organize**: the search box finds sessions by title and by transcript text. A small sparkle (✦) before a title means the assistant wrote it. At the top right, the pencil renames the session and the bin deletes it (after a confirmation, together with its AI notes). Sessions that still have a default title (such as `en → zh lecture`) also get a sparkle button, **Generate title**, to name them on demand.
- **Transcript tab**: the bilingual transcript with timestamps. **Export** saves MD, TXT, JSON, SRT or VTT; if the session has AI notes, the Markdown export puts them at the top. After class, the local alignment model refines the timestamps to word level in the background, and **Timing** then shows `word-aligned`.
- **AI notes tab**:
  1. Set up the assistant in **Settings → AI assistant** first (see below).
  2. Click **Create notes**. EchoLingo asks whether to add course materials (**Add files…**), or you can go ahead with **Create without files**. You can attach up to 5 files of up to 25 MB each, in PDF, PowerPoint (.pptx), Word (.docx), Markdown, LaTeX, CSV or plain text. The text is extracted on your computer and only the extracted text is sent. Scanned PDFs without a text layer are reported and skipped.
  3. The notes stream in as they are written, in the session's translation language: a title, an overview, sections, formulas and key terms. Sessions longer than about 20 minutes are written in 12–15 minute parts that continue one document.
  4. Each section heading has a time range such as `08:40–21:15`. Click it to jump to that part of the transcript, then **Back to notes** to return. The **Contents** list on the left jumps between sections.
  5. The toolbar has **Copy** (copies the Markdown), **Export .md** and **Regenerate** (the current notes are replaced only once the new ones are saved). Notes are saved with the session, so reopening it shows them without calling the model again.

### Settings

| Section | What it holds |
| --- | --- |
| General | Open the floating caption; in-app updates (Updates) |
| Audio | Default audio profile |
| Languages | Default source and target languages |
| Inference | Default inference mode (Auto / Local / Cloud); preload local models at launch |
| Models | Default recognition and translation providers; download, verify and delete local models; the GPU acceleration pack (Windows and Linux) |
| Cloud providers | Preferred cloud recognizer and translator for Auto; one key card per provider |
| AI assistant | The model that writes notes and titles, its consent switch, automatic titles |
| Privacy | The two upload switches: Audio upload and Transcript upload |
| Translation | The standing glossary |
| Appearance | How the floating caption looks |
| Advanced | Diagnostics and the logs folder |

**Cloud providers**: each provider has a card. Paste the key and click **Save** to store it in the system secure store (macOS Keychain, Windows Credential Manager, or Secret Service on Linux), **Clear** to remove it, and **Test** to check the setup.

- Recognition tests only complete the authenticated handshake; they never upload audio.
- Translation tests send one fixed English sentence ("Welcome to the lecture.") and need Transcript upload. If it is not allowed yet, a translation-only provider asks for your consent before testing; a provider with both recognition and translation (such as Qwen Cloud or OpenAI) tests recognition only, and **Include translation in the test…** below the result adds translation.
- The Qwen Cloud card has a **Region** setting, either **Singapore (international)** or **Beijing (mainland China)**. It must match the console where you created the key. The Beijing region has the free quota. **Workspace ID** is optional; leave it empty.
- How to sign up, free tiers and error codes for every provider are in the [cloud provider setup guide](docs/cloud-setup.md).

![Cloud providers settings](docs/screenshots/settings-cloud.png)

**AI assistant**: notes and automatic titles are written by a chat model you choose and billed to your own API key. Recognition and translation are not affected.

- **Provider**: Qwen (Alibaba Model Studio), OpenAI, DeepSeek, Google Gemini, Groq, OpenRouter, SiliconFlow, or a custom OpenAI-compatible endpoint (for example Ollama on your computer). Save that provider's key under **Cloud providers** first; a custom endpoint needs its **Base URL** there instead, and its key is optional.
- **Model**: leave it empty to use the default (for example `qwen-plus` or `gpt-4o-mini`). For long lectures, choose a model with a large context window.
- **Test**: sends one short fixed prompt and no transcript.
- **Consent**: "Send transcripts and attached files to this model for notes and titles" must be on before any transcript or attachment text can be sent. Consent covers the selected provider only; switching providers turns it off.
- **Session titles**: with "Name sessions automatically when they end" on (the default) and consent given, sessions of at least 30 words get a title in their target language when you press Stop.

![AI assistant settings](docs/screenshots/settings-assistant.png)

**Consent dialogs**: a small lock on a button (for example **Start** on a cloud route, **Create notes**, **Generate title**, or **Test** for a translation provider) means that step needs your consent. Clicking it opens a dialog that explains what will be sent, to whom, and what stays on your computer. The dialog opens with the focus on the button that uploads nothing (**Not now**, or **Extract on this computer** when importing slides); nothing changes until you click a button that starts with **Allow**. Your answer is saved as the default and can be turned off at any time in **Settings → Privacy** or **Settings → AI assistant**. If something is still missing (for example a key), the dialog has a button that opens the right settings page.

![Consent dialog before any upload](docs/screenshots/consent.png)

### In-app updates

- By default EchoLingo checks for a new version once at every launch. You can turn this off in **Settings → General → Updates**.
- On the Mac, updates install from inside the app: EchoLingo downloads the new version, verifies its signature, replaces the app in place and restarts. Your sessions, settings, models and keys are kept.
- **Coming from 0.2.0**: version 0.2.0 has no in-app updates, so install once by hand: download the 0.3.0 dmg, quit EchoLingo with ⌘Q and drag the new version into Applications to replace the old one. Later versions update from inside the app.
- After an update, macOS may ask again for the microphone or Screen & System Audio Recording permission; see [No captions, or no sound gets through](#no-captions-or-no-sound-gets-through).
- For now only the Mac version updates from inside the app; download new Windows and Linux test builds by hand.

## Privacy

**What always stays on your computer**

- Audio capture, noise suppression, gain control, speech detection and resampling.
- All recognition and translation when you use the local models.
- Your session history: transcripts, translations, AI notes and settings.
- The temporary audio used to refine timestamps after class. It is deleted as soon as alignment succeeds, is cancelled or fails; files left behind by a crash are removed the next time a session starts once they are older than 24 hours. History does not keep full recordings.

**What is sent, and only after you allow it**

| Feature | What is sent | Sent to | Switch that must be on |
| --- | --- | --- | --- |
| Cloud recognition | The processed session audio, streamed while the session runs; most providers also receive the lecture context or its terms as a recognition hint | The recognition provider you chose | Settings → Privacy → Audio upload |
| Cloud translation | The recognized source sentences; depending on the provider, also a little of the preceding text (Qwen-MT also gets earlier translations), the lecture topic and the glossary pairs | The translation provider you chose | Settings → Privacy → Transcript upload |
| AI notes and titles | The session transcript, its lecture context and text extracted from attached files | The provider chosen in AI assistant | Settings → AI assistant → Consent |
| Import from slides (AI picks terms) | Text extracted from the slides | Same as above | Same as above |
| Connection tests | Recognition: handshake only; translation: one fixed English sentence; AI assistant: one fixed prompt | The provider being tested | Transcript upload for translation tests; otherwise none |

Audio is never sent for AI notes or titles.

**Update checks**: at launch EchoLingo reads a small version file (`latest.json`) from GitHub, and updates are downloaded from GitHub too. Update checks only contact GitHub and upload nothing; turn them off in **Settings → General → Updates**.

**How to revoke**: turn off Audio upload or Transcript upload in **Settings → Privacy**; turn off Consent or set the provider to **Off** in **Settings → AI assistant**; click **Clear** on a card in **Settings → Cloud providers** to delete a key from the secure store. Data already sent to a third-party provider is governed by that provider's own privacy policy.

**Where your data lives**:
- macOS: `~/Library/Application Support/app.echolingo.desktop/`
- Windows: `%LOCALAPPDATA%\app.echolingo.desktop\`
- Linux: `~/.local/share/app.echolingo.desktop/`

| Path | Contents |
| --- | --- |
| `history.sqlite` | Sessions, transcripts, translations, AI notes |
| `preferences.json` | Settings (no keys) |
| `models/` | Local models |
| `runtimes/gpu-pack/` | The GPU acceleration pack (Windows and Linux, once installed) |
| `alignment-spool/` | Temporary audio for timestamp alignment |
| `logs/` | Logs for troubleshooting |

Keys are stored in the system secure store (macOS Keychain, Windows Credential Manager, or Secret Service on Linux) under the service name `app.echolingo.desktop` and are never written to the settings file or the logs. To uninstall completely, first **Clear** each key under **Cloud providers**, then uninstall the app (on Windows in **Settings → Apps → Installed apps**) and delete the folder above.

## FAQ

### The model download is slow or fails

In mainland China, try the Hugging Face mirror. On macOS, run `launchctl setenv HF_ENDPOINT https://hf-mirror.com`, quit EchoLingo with ⌘Q, reopen it and click **Download** again; for Windows and Linux, see [Windows / Linux](#windows--linux-in-testing-not-yet-published). If downloads do not go through your proxy, set `HTTPS_PROXY` the same way, for example `launchctl setenv HTTPS_PROXY http://127.0.0.1:7890` on macOS or `setx HTTPS_PROXY http://127.0.0.1:7890` on Windows (use your own proxy address). After a failed or interrupted download, click **Download** again.

### Start says a required local model is not ready

The local models have not been downloaded yet. Download them in **Settings → Models**.

### The first Start takes a long time

Loading the models the first time takes 1–2 minutes. With "Preload local models when EchoLingo launches" on in **Settings → Inference** (the default), they are loaded in the background whenever the app starts.

### No captions, or no sound gets through

- On macOS, open **System Settings → Privacy & Security → Microphone** and make sure EchoLingo is on.
- To capture sound playing on the Mac, also turn EchoLingo on under **System Settings → Privacy & Security → Screen & System Audio Recording** (called **Screen Recording** on macOS 14), then reopen the app.
- After updating the Beta, macOS may ask for permission again. If the switch is on but there is still no sound, remove EchoLingo from the list and grant access again.
- On Windows, open **Settings → Privacy & security → Microphone** and make sure microphone access and **Let desktop apps access your microphone** are on.
- On Linux, check the input device and its level in the system sound settings.
- Check that the level bar in the **Audio / VAD** panel on the Live screen moves.

### Captions fall further and further behind (Windows / Linux)

Without the GPU acceleration pack, recognition runs on the CPU, and slower processors may not keep up in real time. Install the [GPU acceleration pack](#windows--linux-in-testing-not-yet-published) if the computer has a supported NVIDIA card, or set a cloud recognizer under **Settings → Cloud providers**: with the route on Auto, it is used when the local model misses its latency target, once you allow audio upload.

### The floating caption does not stay on top (Linux)

On Wayland desktops, an app cannot always keep its own window above the others. Start EchoLingo with the X11 backend, for example `GDK_BACKEND=x11 echolingo-desktop`, or log in to an X11 session.

### Keys cannot be saved (Linux)

EchoLingo saves keys through the Secret Service. Install or unlock GNOME Keyring or KWallet and reopen EchoLingo. Until then, set the provider's environment variable (for example `DEEPL_API_KEY`) before starting the app; each card in **Settings → Cloud providers** names its variable.

### The Qwen Cloud test fails

- HTTP 401 usually means the **Region** does not match the key. A key created in the [Model Studio console for mainland China](https://bailian.console.aliyun.com) needs **Beijing**; a key from the [international console](https://modelstudio.console.alibabacloud.com) needs **Singapore**.
- An HTTP 403 about the workspace: clear **Workspace ID** to use the key's default workspace.
- HTTP 403 `AllocationQuota.FreeTierOnly`: the free quota is used up.
- From mainland China, Qwen Cloud Beijing, DeepSeek and SiliconFlow are reachable directly; OpenAI, Deepgram, AssemblyAI, Gladia, DeepL, Google and Azure usually need a proxy.

### AI notes show an error

| Message | Meaning | What to do |
| --- | --- | --- |
| The AI assistant is not set up yet. | No assistant provider or key yet | Choose a provider in **Settings → AI assistant** and save its key under **Cloud providers** |
| Sending transcripts to the AI assistant is turned off. | Sending transcripts is not allowed | Turn on Consent in **Settings → AI assistant** |
| The provider rejected the API key. | The key was rejected | Check the key under **Cloud providers** (for Qwen, also the Region) |
| The provider’s rate limit or quota was reached. | The provider's rate limit or quota was reached | Wait a minute and try again, or pick another provider |
| This session is too long for the selected model. | The session is too long for the model's context window | Choose a model with a larger context, or attach fewer files |
| EchoLingo could not reach the provider. | The provider cannot be reached | Check the network connection and proxy |

Existing notes are kept when a new attempt fails. The error message usually has a button that opens the right settings page, and **Try again** to retry.

### How can I improve accuracy?

- Fill in the **lecture context** before class: the topic, names and technical terms. The more specific, the better. **Import from slides…** helps.
- Put recurring terms and their fixed translations in **Settings → Translation → Glossary**.
- Point the microphone at the lecturer or the room speakers, and keep it away from air vents and laptop fans; an external microphone helps. Keep the audio profile on **Lecture / far-field**.
- Make sure the source language is right. For online classes and videos use **System audio** rather than letting the microphone pick up the speakers.

### Can I use it completely offline?

Yes. Once the models are downloaded, local recognition and translation need no network. AI notes need a connection unless the assistant points to a custom endpoint running on your computer (such as Ollama).

## Roadmap

- **Windows and Linux**: published once they have been tested on real computers.
- GPU acceleration on AMD and Intel graphics cards.
- System audio capture on Linux, and an AppImage package for more distributions.
- Code signing on Windows and Apple notarization, so the first launch needs no manual steps.
- ARM64 builds for Windows and Linux.
- More languages.

## Build from source

You need Node.js 22 LTS, Rust 1.88 or later (stable) and Conda (Miniconda or Miniforge), plus:

- **macOS**: an Apple Silicon Mac with macOS 14 or later and the Xcode Command Line Tools.
- **Windows**: Windows 10 or 11 (x64) with the Visual Studio 2022 Build Tools (**Desktop development with C++**) and the WebView2 runtime.
- **Linux**: Ubuntu 22.04 or later (x64) with `sudo apt install build-essential curl file libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev libxdo-dev libssl-dev libasound2-dev libdbus-1-dev pkg-config`.

```bash
git clone https://github.com/Asphr726/EchoLingo.git
cd EchoLingo

# Python environment and the speech recognition runtime
conda env create -f environment.yml
conda activate echolingo-spike1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # Windows and Linux only: CPU build of PyTorch
python -m pip install -e ".[qwen]"

# Translation runtime (the llama.cpp build for this platform)
python scripts/fetch_llama_runtime.py

# Desktop app
npm install
python scripts/build_sidecar.py --clean          # packages the local inference service (and downloads the pinned Silero VAD model)
npm run desktop:dev                              # run in development mode
npm run desktop:build                            # Windows and Linux: installer and packages
APPLE_SIGNING_IDENTITY=- npm run desktop:build   # macOS: local ad-hoc signed build
```

The build ends up in `target/release/bundle/`: `dmg/EchoLingo_0.3.0_aarch64.dmg` on macOS, `nsis/EchoLingo_0.3.0_x64-setup.exe` on Windows, and `deb/EchoLingo_0.3.0_amd64.deb` on Linux. `APPLE_SIGNING_IDENTITY=-` produces an ad-hoc signature that is only suitable for your own Mac; public distribution needs a Developer ID certificate and Apple notarization.

Run the tests:

```bash
pytest                                      # Python (with the conda environment active)
cargo test --workspace                      # Rust
npm run typecheck && npm run test:desktop   # user interface
```

### Releasing (maintainers)

In-app updates only accept update packages signed with the EchoLingo update signing key. By convention the private key lives in `~/.tauri/echolingo-updater.key` (created with `npx tauri signer generate -w ~/.tauri/echolingo-updater.key`; the public key is `plugins.updater.pubkey` in `apps/desktop/src-tauri/tauri.conf.json`). Keep an offline backup of this file: without it, installed copies can never update from inside the app again and have to be reinstalled by hand. Local builds do not need the key; CI reads its contents from the repository secret `TAURI_SIGNING_PRIVATE_KEY` (plus `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` if the key has a password), for example `gh secret set TAURI_SIGNING_PRIVATE_KEY < ~/.tauri/echolingo-updater.key`.

1. Bump the version, write `.github/release-notes/vX.Y.Z.md` and push the tag `vX.Y.Z`.
2. CI builds every platform and creates two drafts: `vX.Y.Z` (the Mac dmg, the update package `.app.tar.gz` and its `.sig`, `SHA256SUMS.txt`, and a `latest.json` to review) and `windows-linux-vX.Y.Z` (the Windows / Linux installers and GPU packs; it stays a draft).
3. Once it checks out, publish `vX.Y.Z` as a pre-release.
4. Run `python scripts/update_manifest.py --tag vX.Y.Z --publish` (needs a signed-in GitHub CLI, `gh`). It checks the signing key and the download links, then replaces `latest.json` on the `updater` pre-release; installed copies see the new version at their next check.

`python scripts/smoke_update.py` rehearses a complete in-app update on your Mac with a throwaway key.

## License and acknowledgements

EchoLingo is open source under the [Apache License 2.0](LICENSE). The local models are not bundled with the app; you download them from Hugging Face inside the app, and they are distributed under their own licenses. The full list of third-party components and their licenses is in [docs/third-party-components.md](docs/third-party-components.md).

EchoLingo builds on these projects:

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) (Qwen team, Alibaba): speech recognition and forced alignment models
- [Hy-MT2](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) (Tencent Hunyuan): the local translation model
- [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit): the streaming recognition framework
- [llama.cpp](https://github.com/ggml-org/llama.cpp): the runtime for the local translation model
- [Silero VAD](https://github.com/snakers4/silero-vad): speech detection
- [Tauri](https://tauri.app): the desktop app framework
