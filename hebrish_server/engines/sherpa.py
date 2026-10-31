"""sherpa-onnx offline transducer on the int8 build: the apps' engine, so the same text.
CPU by default; HEBRISH_DEVICE=cuda needs a CUDA build of sherpa-onnx (not in the images)."""

import threading
import time

import numpy as np

from .. import config, models
from ..segment import SAMPLE_RATE


class SherpaEngine:
    def __init__(self):
        import sherpa_onnx
        t0 = time.monotonic()
        folder, manifest = models.sherpa()
        self.provider = "cuda" if config.DEVICE == "cuda" else "cpu"
        self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            model_type="nemo_transducer", encoder=str(folder / "encoder.int8.onnx"),
            decoder=str(folder / "decoder.int8.onnx"), joiner=str(folder / "joiner.int8.onnx"),
            tokens=str(folder / "tokens.txt"), decoding_method="greedy_search",
            num_threads=config.THREADS, sample_rate=SAMPLE_RATE, feature_dim=config.FEATURE_DIM, provider=self.provider)
        self._lock = threading.Lock()
        self.info = {"engine": "sherpa-onnx", "sherpa_onnx": sherpa_onnx.__version__,
                     "model_id": config.MODEL, "model": manifest["version"], "run": manifest.get("run"),
                     "source": manifest["base_url"], "revision": manifest.get("revision"),
                     "device": self.provider, "threads": config.THREADS}
        self.decode(np.zeros(SAMPLE_RATE, np.float32))          # warm-up
        self.info["load_s"] = round(time.monotonic() - t0, 1)

    def decode(self, x: np.ndarray) -> str:
        with self._lock:
            s = self._rec.create_stream()
            s.accept_waveform(SAMPLE_RATE, x)
            self._rec.decode_stream(s)
            return s.result.text
