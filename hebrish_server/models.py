"""Downloading the model files into MODELS_DIR, once."""

import hashlib
import json
import logging
import shutil
import urllib.request
from pathlib import Path

from . import config

log = logging.getLogger(__name__)
# The CDN answers 403 to Python's default User-Agent
UA = {"User-Agent": "hebrish-server/0.1 (+https://github.com/Hebrish-ASR/Server)"}


def _open(url: str, timeout: float = 30):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)


def _get_json(url: str) -> dict:
    with _open(url) as r:
        return json.load(r)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def sherpa() -> tuple[Path, dict]:
    """(folder with encoder/decoder/joiner/tokens, the CDN manifest). Files already present with
    the manifest's size and sha256 are not fetched again."""
    m = _get_json(config.SHERPA_MANIFEST)
    folder = config.MODELS_DIR / "sherpa" / m["model"] / m["version"].replace(" ", "-")
    folder.mkdir(parents=True, exist_ok=True)
    for name, meta in m["files"].items():
        dest = folder / name
        # a manifest can list a file without a hash (parakeet-sabich-110m tokens.txt, 2026-10-31):
        # then its size is the check
        want = meta.get("sha256")
        if dest.exists() and dest.stat().st_size == meta["size"] and (want is None or _sha256(dest) == want):
            continue
        log.info("downloading %s (%d bytes)", name, meta["size"])
        tmp = dest.with_suffix(dest.suffix + ".part")
        with _open(m["base_url"] + name, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        if tmp.stat().st_size != meta["size"] or (want is not None and _sha256(tmp) != want):
            tmp.unlink()
            raise RuntimeError(f"{name}: size or sha256 does not match the manifest")
        tmp.rename(dest)
    (folder / "manifest.json").write_text(json.dumps(m, indent=1))
    return folder, m


def nemo() -> Path:
    """The .nemo file from the gated Hugging Face repo (HF_TOKEN in the environment)."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(config.NEMO_REPO, config.NEMO_FILE, revision=config.NEMO_REVISION,
                                cache_dir=str(config.MODELS_DIR / "hf")))
