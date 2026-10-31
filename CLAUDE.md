# CLAUDE.md

Hebrish Server (Hebrish-ASR/Server): FastAPI web UI + HTTP API running Hebrish models on CPU
(sherpa-onnx int8) or GPU (NeMo on CUDA/ROCm torch). Read `README.md` first. Proof of concept.

- **Public repository, and a boilerplate.** Write for strangers: no host names, LAN or tailnet
  addresses, personal paths, account ids or credentials (name secrets by their variable, e.g.
  `HF_TOKEN`). Deployment-specific settings belong in a git-ignored `compose.override.yaml` or a
  private deployment repo, never here.
- `hebrish_server/segment.py`, the `Pieces` class in `pipeline.py` and the top of `cleanup.py`
  are verbatim copies from the desktop app (Hebrish-ASR/Hebrish-App `desktop/hebrish/`, commit
  noted in each file). Re-copy when the app changes; don't edit them in place.
- Models: `MODELS` in `config.py` (sherpa engine; keys, CDN manifests, `feature_dim`, from the
  app's `shared/catalogue.json`). The NeMo engine is Sabich-only, pinned by
  `HEBRISH_NEMO_REVISION`; bump it and `HEBRISH_NEMO_LABEL` together for a new Sabich version.
- Don't claim support for a model or hardware that hasn't been run: say "not verified" in the
  README instead. No benchmark numbers from this server; benchmarks are on hebrish.app.
- Hebrew words in romanized form only; Hebrish is Daniel Rosehill's one-person project (no
  "we"); no compute costs.
- Python: `uv` only.
