"""An engine decodes one prepared piece of audio (16 kHz float32, tail pad included) to text.
Cutting, padding and paragraphs are the same for every engine (pipeline.py)."""

from .. import config


def load():
    if config.ENGINE == "nemo":
        if config.MODEL != "sabich":
            raise ValueError(f"the nemo engine serves only Sabich; HEBRISH_MODEL={config.MODEL!r} needs the sherpa engine")
        from .nemo import NemoEngine
        return NemoEngine()
    if config.ENGINE == "sherpa":
        from .sherpa import SherpaEngine
        return SherpaEngine()
    raise ValueError(f"HEBRISH_ENGINE must be sherpa or nemo, not {config.ENGINE!r}")
