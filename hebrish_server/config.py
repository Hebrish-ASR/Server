"""Settings, all from environment variables (the compose file sets them)."""

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


ENGINE = _env("HEBRISH_ENGINE", "sherpa")            # sherpa | nemo
DEVICE = _env("HEBRISH_DEVICE", "auto")              # auto | cpu | cuda (cuda also means ROCm under a ROCm torch)
THREADS = int(_env("HEBRISH_THREADS", "6"))          # sherpa-onnx CPU threads: about the number of performance cores
MODELS_DIR = Path(_env("HEBRISH_MODELS_DIR", "./models"))
HOST = _env("HEBRISH_HOST", "127.0.0.1")
PORT = int(_env("HEBRISH_PORT", "8870"))

# Models the sherpa engine can serve: the sherpa-onnx int8 builds the Hebrish apps download (CDN
# manifest, sha256-checked; the same entries as Hebrish-ASR/Hebrish-App shared/catalogue.json,
# engine sherpa_offline_transducer). The manifest follows each model's current version. To add a
# model, add its catalogue id, manifest URL and feature_dim here (README, "Adding models").
MODELS = {
    "sabich": {"manifest": "https://cdn.hebrish.app/models/parakeet-sabich/latest.json", "feature_dim": 128},
    "sabich-parakeet-110m": {"manifest": "https://cdn.hebrish.app/models/parakeet-sabich-110m/latest.json",
                             "feature_dim": 80},
    "jachnun": {"manifest": "https://cdn.hebrish.app/models/parakeet-codeswitch/latest.json", "feature_dim": 128},
}
MODEL = _env("HEBRISH_MODEL", "sabich")               # a key of MODELS (sherpa engine)
if MODEL not in MODELS:
    raise ValueError(f"HEBRISH_MODEL must be one of {', '.join(MODELS)}, not {MODEL!r}")
SHERPA_MANIFEST = _env("HEBRISH_SHERPA_MANIFEST", MODELS[MODEL]["manifest"])     # overrides the model's URL
FEATURE_DIM = int(_env("HEBRISH_FEATURE_DIM", str(MODELS[MODEL]["feature_dim"])))
# NeMo engine: Sabich only, pinned (gated; needs HF_TOKEN). c818843b0e = Sabich v2.3, run r13-all, 2026-10-30
NEMO_REPO = _env("HEBRISH_NEMO_REPO", "Hebrish/Sabich")
NEMO_FILE = _env("HEBRISH_NEMO_FILE", "Sabich.nemo")
NEMO_REVISION = _env("HEBRISH_NEMO_REVISION", "c818843b0e")
NEMO_LABEL = _env("HEBRISH_NEMO_LABEL", "Sabich v2.3")

# Clean-up: an Ollama server with the S1-mini model (`ollama pull` it, or point CLEANUP_URL at a server that has it)
CLEANUP_URL = _env("HEBRISH_CLEANUP_URL", "http://127.0.0.1:11434")
CLEANUP_MODEL = _env("HEBRISH_CLEANUP_MODEL", "s1-mini")

PARAGRAPH_S = 30          # Time-based paragraphs: one per ~30 s of audio, as in the desktop app
MAX_JOBS_KEPT = 100       # finished jobs kept in memory (there is no database)
