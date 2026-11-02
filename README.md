# Hebrish Server

Self-hosted inference for [Hebrish](https://hebrish.app) models: a web UI and an HTTP API that
transcribe audio (English with Hebrew words, romanized: *makolet*, *arnona*), lay it out in
paragraphs and optionally clean it up with S1-mini, like the Hebrish desktop app. **Sabich** is
the default model; Jachnun and Sabich-Parakeet-110M can be chosen, and other Hebrish models can
be added (see "Adding models"). Runs on the CPU, an NVIDIA GPU (CUDA) or an AMD GPU (ROCm).

**Status: proof of concept (0.1.0, 2026-10-30).** No authentication: publish it on localhost,
a LAN or a tailnet address only, never on the internet. Jobs live in memory and are lost on
restart. One request is decoded at a time.

This repository is a starting point to deploy and adapt: it holds no host-specific settings.
Put your own (bind address, networks, GPU) in a compose override file next to `compose.yaml`
(see "Run").

## What it does

| | |
|---|---|
| Web UI (`/`) | Drop or choose an audio/video file (anything ffmpeg reads), or record in the browser; the transcript fills in piece by piece; job list with speed |
| Paragraphs | *None* or *Time-based* (one per ~30 s of audio, split at a pause), as in the app. The app's *Paragraph model* mode is not here yet |
| Clean-up | S1-mini through an Ollama server (prompt, chunking and dictionary guard copied from the app) |
| `POST /v1/audio/transcriptions` | OpenAI-compatible, synchronous. Fields: `file`, `model` (ignored: the server's model answers), `response_format` (`json`, `text`, `verbose_json`), plus `paragraphs` (`none`/`time`) and `cleanup` (`true`/`false`) |
| `POST /api/jobs`, `GET /api/jobs`, `GET /api/jobs/{id}` | Asynchronous jobs (what the UI uses); poll for `progress` and partial `text` |
| `POST /api/cleanup` | `{"text": ...}`: clean-up alone |
| `GET /api/health` | Engine, model id, version and revision, device (GPU name, CUDA or ROCm), whether clean-up is reachable |
| `/docs` | FastAPI's generated API docs |

```bash
curl -F file=@note.m4a -F response_format=text http://127.0.0.1:8870/v1/audio/transcriptions
```

From another machine, replace `127.0.0.1` with `<server-address>`. Recording in the browser
needs a secure context: it works on `localhost` or over HTTPS, not on a plain `http://` LAN
address (the browser hides the microphone there). Upload a file instead, or put the server
behind HTTPS (a reverse proxy, or `tailscale serve` on a tailnet).

## Models

`HEBRISH_MODEL` (compose variable `MODEL`) picks the model for the `sherpa` engine. Each is the
sherpa-onnx int8 build the Hebrish apps download, fetched from its CDN manifest on first start,
sha256-checked, and updated to the model's current version on a restart.

| `HEBRISH_MODEL` | Model | Manifest |
|---|---|---|
| `sabich` (default) | Sabich, code-switching and tech vocabulary; the model the apps recommend | `cdn.hebrish.app/models/sabich/latest.json` |
| `sabich-parakeet-110m` | Sabich-Parakeet-110M, a smaller, faster Sabich (`feature_dim` 80) | `cdn.hebrish.app/models/sabich-110m/latest.json` |
| `jachnun` | Jachnun | `cdn.hebrish.app/models/jachnun/latest.json` |

The `nemo` engine serves Sabich only (the full-precision `.nemo` checkpoint, below). Model
descriptions and results: [hebrish.app/models](https://hebrish.app/models).

### Adding models

A model that the apps run with engine `sherpa_offline_transducer` (Hebrish-ASR/Hebrish-App
`shared/catalogue.json`) needs one entry in `MODELS` in `hebrish_server/config.py`: a key, its
`manifest_url` and its `feature_dim`. `HEBRISH_SHERPA_MANIFEST` and `HEBRISH_FEATURE_DIM` override
both for a one-off test without editing the code.

Not supported yet, because each needs its own engine in `hebrish_server/engines/`:

- **Shakshuka** (streaming, `sherpa_online_transducer`): an `OnlineRecognizer` engine.
- Sabich variants on other bases (Qwen3-ASR, Canary-Qwen, asr4all): their own runtimes.
- **Meorav** (text-to-speech): a different API altogether.

An engine is a class with `decode(x) -> str` for one prepared 16 kHz piece and an `info` dict;
`engines.load()` picks it. Cutting, padding and paragraphs are shared (`pipeline.py`).

## Engines and hardware

| Engine | Model files | Hardware | Image |
|---|---|---|---|
| `sherpa` (default) | The chosen model's sherpa-onnx int8 build (above) | CPU | `docker/Dockerfile.cpu`, profile `cpu` |
| `nemo` | `Hebrish/Sabich` `Sabich.nemo` (gated, needs `HF_TOKEN`), pinned to revision `c818843b0e` = **Sabich v2.3** (run `r13-all`, 2026-10-30) | NVIDIA (torch cu128), AMD (torch rocm7.1), or CPU | `docker/Dockerfile.torch`, profiles `nvidia`, `rocm`, `nemo-cpu` |

The sherpa engine gives the same text as the apps: the audio preparation and cutting
(`hebrish_server/segment.py`) and Time-based paragraphs are copied verbatim from the desktop app
(Hebrish-ASR/Hebrish-App `desktop/hebrish/`, commit `ca269f5`). The NeMo engine runs the
full-precision checkpoint through the same cutting, so its text can differ slightly from the
int8 build. Both decode greedily.

The torch images install torch 2.11.0 from the PyTorch index for the hardware, and NeMo 2.7.3.
The wheels carry their own CUDA or ROCm libraries, so the host needs only the GPU driver
(NVIDIA: plus the NVIDIA Container Toolkit).

### ROCm

The `rocm` profile passes `/dev/kfd` and `/dev/dri`. If torch reports no GPU on a card that is
not in the torch build's gfx list, set `HSA_OVERRIDE_GFX_VERSION` in the service environment
(for example `11.0.1` for RDNA3 cards such as gfx1101; not yet tried with this image).

## Run

```bash
# CPU, sherpa-onnx (published on 127.0.0.1:8870 by default)
docker compose --profile cpu up -d --build
# another model
MODEL=sabich-parakeet-110m docker compose --profile cpu up -d --build
# NVIDIA or AMD GPU, NeMo (needs HF_TOKEN with access to Hebrish/Sabich)
HF_TOKEN=... docker compose --profile nvidia up -d --build
HF_TOKEN=... docker compose --profile rocm up -d --build
curl -s http://127.0.0.1:8870/api/health
```

Compose variables: `BIND` (host address, default `127.0.0.1`), `PORT` (8870), `MODEL`
(`sabich`), `THREADS` (6), `CLEANUP_URL` (Ollama, default the host's port 11434),
`CLEANUP_MODEL` (`s1-mini`), `HF_TOKEN`. Models are stored in `./models`. Without Docker:
`uv venv && uv pip install -e .` then `python -m hebrish_server` (environment variables in
`hebrish_server/config.py`).

To publish on a LAN or tailnet address, set `BIND` to that address, or keep your host's ports and
networks in an override file and start with
`docker compose -f compose.yaml -f compose.override.yaml --profile cpu up -d`
(`compose.override.yaml` is git-ignored, so a deployment can keep a clone of this repository
clean).

## Verified

2026-10-30, on an x86 CPU server (Intel i5-12600K, no GPU), Sabich v2.3: a smoke test, not a
benchmark, on four code-switching clips from public test set v2
(`Hebrish/English-Hebrew-Codeswitching-ASR-Benchmark` revision `48ec85a3`) and a 107.7 s file,
as WAV and M4A. The sherpa engine (`cpu` image) ran at about 18-23x realtime with 6 threads;
NeMo on the CPU torch build (`nemo-cpu`) at about 8-9.5x. Both engines gave the same text; three
clips matched the reference exactly. Time-based paragraphs and clean-up (Ollama `s1-mini`)
worked; a non-audio upload returns HTTP 422 with ffmpeg's message.

2026-10-31, from source (`python -m hebrish_server`, no Docker) on a laptop CPU: `sabich`
(Sabich v2.3) and `sabich-parakeet-110m` (v1.0, downloaded from its manifest) both transcribed a
40 s voice note through `/v1/audio/transcriptions`.

Not verified: any GPU (the `nvidia` and `rocm` images have not been built and run), and
`jachnun` in this server (the apps' build, served through the same code path; its manifest was
checked on 2026-10-31).

## Limits

- The Hebrish models are trained on one speaker's voice; other voices work, with more errors.
  More voices would help: recordings can be contributed through
  [hebrish.app/get-involved](https://hebrish.app/get-involved).
- S1-mini clean-up can drop a Hebrew word that is not in the dictionary guard (the server passes
  no personal dictionary).

## Licence

The code: MIT. The models are under their own licences and the Hebrish terms
([hebrish.app/terms](https://hebrish.app/terms)); Sabich v2.x is CC-BY-NC-4.0. Hebrish is
Daniel Rosehill's one-person project; feature requests and bug reports go through
[hebrish.app/feedback](https://hebrish.app/feedback).
