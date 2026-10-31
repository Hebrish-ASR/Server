"""NeMo on PyTorch with the full-precision .nemo checkpoint. Runs on an NVIDIA GPU (CUDA torch),
an AMD GPU (ROCm torch: torch.cuda is HIP there) or the CPU. Greedy decoding, like the
benchmarks."""

import logging
import threading
import time

import numpy as np

from .. import config, models
from ..segment import SAMPLE_RATE

log = logging.getLogger(__name__)


def device_info(torch) -> dict:
    out = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
           "hip_runtime": getattr(torch.version, "hip", None), "gpu_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        out["gpu_backend"] = "rocm" if out["hip_runtime"] else "cuda"
        out["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    return out


class NemoEngine:
    def __init__(self):
        import torch
        import nemo
        import nemo.collections.asr as nemo_asr
        t0 = time.monotonic()
        dev = config.DEVICE
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("HEBRISH_DEVICE=cuda but torch sees no GPU " + str(device_info(torch)))
        self.device = torch.device(dev)
        path = models.nemo()
        m = nemo_asr.models.ASRModel.restore_from(str(path), map_location=self.device)
        m.eval()
        m.freeze()
        self._m, self._torch = m, torch
        self._lock = threading.Lock()
        self.info = {"engine": "nemo", "nemo": nemo.__version__, "model_id": "sabich", "model": config.NEMO_LABEL,
                     "source": f"{config.NEMO_REPO}/{config.NEMO_FILE}", "revision": config.NEMO_REVISION,
                     "device": dev, **device_info(torch)}
        self.decode(np.zeros(SAMPLE_RATE, np.float32))          # warm-up
        self.info["load_s"] = round(time.monotonic() - t0, 1)

    def decode(self, x: np.ndarray) -> str:
        with self._lock, self._torch.inference_mode():
            out = self._m.transcribe([x], batch_size=1, verbose=False)
        if isinstance(out, tuple):          # older NeMo: (best, all)
            out = out[0]
        h = out[0]
        return h.text if hasattr(h, "text") else str(h)
