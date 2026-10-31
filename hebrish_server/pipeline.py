"""A whole file through the engine, cut and laid out the way the desktop app does it."""

import time

import numpy as np

from . import config
from .segment import SAMPLE_RATE, MIN_TAKE_S, TAIL_PAD_S, drop_unk, normalize, split_points, stretch_pauses

NONE, TIME, MODEL = "none", "time", "model"
MODES = (NONE, TIME)


# Copied verbatim from Hebrish-ASR/Hebrish-App desktop/hebrish/paragraphs.py (commit ca269f5).
# The Paragraph model mode (MODEL) is not on the server yet.
class Pieces:
    """One transcript's pieces, in order, each with whether a Time-based paragraph starts there.

    `every` is the Time-based spacing in samples (PARAGRAPH_S * SAMPLE_RATE): a piece starts a
    paragraph when at least that much audio has been decoded since the last one started (the first
    piece always does). A piece with no text is dropped, and a paragraph start it carried moves to
    the next piece."""

    def __init__(self, every: int):
        self.every = every
        self.items: list[tuple[str, bool]] = []
        self._at = None
        self._carry = False

    def add(self, text: str, done: int, mode: str) -> str | None:
        """Record a piece that ends `done` samples into the audio. Returns what goes before it in
        `mode` ("" for the first piece, "\\n\\n" or " "), or None when it has no text."""
        new = self._at is None or done - self._at >= self.every
        if new:
            self._at = done
        new = new or self._carry
        if not text.strip():
            self._carry = new
            return None
        self._carry = False
        self.items.append((text, new))
        return self._sep(len(self.items) - 1, mode)

    def _sep(self, i: int, mode: str) -> str:
        if i == 0:
            return ""
        return "\n\n" if mode == TIME and self.items[i][1] else " "

    def text(self, mode: str, breaker: "object" = None) -> str:
        """The transcript laid out in `mode`. MODEL needs `breaker` and may raise; the caller
        falls back to TIME."""
        if mode == MODEL:
            return breaker.apply(self.text(NONE))
        return "".join(self._sep(i, mode) + t for i, (t, _new) in enumerate(self.items))


# -- tokenizer: BERT uncased WordPiece (spec step 2) --------------------------------


def transcribe(engine, samples: np.ndarray, paragraphs: str = NONE, on_progress=None) -> dict:
    """Decode `samples` piece by piece (the app's _prepare with voice detection off, then
    split_points and a 1 s tail pad per piece). on_progress(done_s, total_s, text_so_far)."""
    audio_s = len(samples) / SAMPLE_RATE
    if len(samples) < MIN_TAKE_S * SAMPLE_RATE:
        return {"text": "", "pieces": 0, "audio_s": audio_s, "decode_s": 0.0}
    x = stretch_pauses(normalize(samples))
    pad = np.zeros(int(TAIL_PAD_S * SAMPLE_RATE), np.float32)
    pieces = Pieces(config.PARAGRAPH_S * SAMPLE_RATE)
    spent, cuts = 0.0, split_points(x)
    for a, b in cuts:
        t0 = time.monotonic()
        text = drop_unk(engine.decode(np.concatenate([x[a:b], pad]))).strip()
        spent += time.monotonic() - t0
        pieces.add(text, b, paragraphs)
        if on_progress:
            on_progress(b / len(x), pieces.text(paragraphs))
    return {"text": pieces.text(paragraphs), "pieces": len(cuts), "audio_s": round(audio_s, 2),
            "decode_s": round(spent, 2), "realtime_x": round(audio_s / spent, 1) if spent else None}
