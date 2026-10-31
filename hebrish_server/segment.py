"""Audio preparation and cutting, copied verbatim from the Hebrish desktop app so the server
decodes a file exactly the way the app does (Hebrish-ASR/Hebrish-App, desktop/hebrish/engine.py
lines 15-257 at commit ca269f5, 2026-10-30). Only the import of SAMPLE_RATE changed. Silero VAD
is never loaded here (`silero` stays None), so voice detection is the app's energy rule.
Re-copy when the app's cutting changes; do not edit in place."""

import logging
import re

import numpy as np

SAMPLE_RATE = 16000

log = logging.getLogger(__name__)

PIECE_S = 10          # long takes are decoded in pieces of ~10 s or more, cut at a pause (live_cut)
MIN_TAKE_S = 0.3
# Without trailing silence the model drops the end of the audio: with <0.6 s after the
# last word it lost a whole final sentence (measured 2026-10-18, v4). Releasing
# push-to-talk right as you stop talking produces exactly that, so every chunk is padded.
TAIL_PAD_S = 1.0
# Short pauses between sentences can swallow speech: two sentences 0.3 s apart lost half
# of the second, which decodes fine alone (v4, 2026-10-18). Stretching every pause of
# 0.12-0.6 s to 0.6 s recovered sentences in every test (0.3 s gaps: 6/8 -> 8/8) and the
# only single-sentence change was a fix ("Bokartov" -> "Boker Tov"). Costs ~20% more audio.
PAUSE_MIN_S = 0.12
PAUSE_TARGET_S = 0.6
FRAME = 320                       # 20 ms


UNK = re.compile(r" *<unk> *")


def drop_unk(text: str) -> str:
    """The model's text without `<unk>` (token 0 of the Sabich/Jachnun BPE vocabulary). The model
    writes it at a clause break, where its training text had a mark the vocabulary lacks (em/en
    dash, semicolon, double quote): six times in Daniel's history, 2026-10-21 to 25 (1.003). A
    space takes its place; docs/maintaining.md, "<unk> in the output"."""
    return UNK.sub(" ", text) if "<unk>" in text else text


def normalize(x: np.ndarray, target_dbfs=-20.0, max_gain=10.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    if rms < 1e-5:
        return x
    gain = min(10 ** (target_dbfs / 20) / rms, max_gain)
    return np.clip(x * gain, -1.0, 1.0).astype(np.float32)


def stretch_pauses(x: np.ndarray) -> np.ndarray:
    """Pad quiet runs of PAUSE_MIN_S..PAUSE_TARGET_S out to PAUSE_TARGET_S (inserting zeros in
    the middle of the run). Quiet = below 15% of the way from the noise floor (10th
    percentile frame RMS) to speech level (90th), so it adapts to the mic's gain."""
    n = len(x) // FRAME
    if n < 10:
        return x
    rms = np.sqrt((x[: n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1))
    floor, speech = np.percentile(rms, 10), np.percentile(rms, 90)
    quiet = rms < floor + 0.15 * (speech - floor)
    out, last, i = [], 0, 0
    while i < n:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < n and quiet[j]:
            j += 1
        run = (j - i) * FRAME / SAMPLE_RATE
        if PAUSE_MIN_S <= run < PAUSE_TARGET_S and i > 0 and j < n:     # inner pauses only
            mid = (i + j) // 2 * FRAME
            out += [x[last:mid], np.zeros(int((PAUSE_TARGET_S - run) * SAMPLE_RATE), np.float32)]
            last = mid
        i = j
    out.append(x[last:])
    return np.concatenate(out)


# Voice activity (config `vad`, 0.5.0). Energy-based, no extra model: a frame is speech when
# its RMS is above VAD_ABS (-46 dBFS) and above the noise floor by a third of the way to the
# speech level. A take with under VAD_MIN_SPEECH_S of speech frames, or whose loudest frames
# stay under VAD_QUIET (-40 dBFS), is not decoded: Parakeet turns silence and breathing into
# stray words ("Yeah.", "Mm."). Silences longer than VAD_LONG_S are shortened to VAD_KEEP_S
# (still above PAUSE_TARGET_S and LIVE_PAUSE_S, so pause cuts are unaffected) to save decode time.
VAD_ABS = 0.005
VAD_QUIET = 0.01
VAD_MIN_SPEECH_S = 0.25
VAD_LONG_S = 1.5
VAD_KEEP_S = 0.8
# A piece of a take (live pieces and the last one, decoded after Stop) is judged against the levels
# of the take so far (`ref`), not its own (R-2026-10-25-3). Alone, the few seconds spoken just
# before Stop were often too short and soft to pass: their loudest frames stayed under VAD_QUIET
# on a headset at its default gain (B-46: speech around -41 dBFS), so the last words were dropped
# and nothing said so. With `ref`, a frame is speech when it clears the usual rule on the take's
# levels or stands VAD_REL_DB above the take's noise floor, and in any case VAD_DEAD (a dead or
# muted mic). Not yet measured against the benchmark. A quiet whole take gets the same rule (B-46).
VAD_REL_DB = 15
VAD_DEAD = 10 ** (-70 / 20)
REF_S = 30                 # how much of the take before a piece serves as its `ref`


def _frame_rms(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    return np.sqrt((x[: n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1))


# Silero VAD (vad.Silero), when the add-on is downloaded and `vad_silero` is on: set by
# Controller.apply_vad. It replaces the energy rule below for has_speech and shorten_silences only.
silero = None


def _speech_frames(x: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    if silero is not None:
        try:
            return silero.speech_frames(x, FRAME)
        except Exception:
            log.exception("Silero VAD failed; using the energy rule")
    rms = _frame_rms(x)
    if not len(rms):
        return rms > 1
    levels = rms if ref is None or len(ref) < FRAME else np.concatenate([_frame_rms(ref), rms])
    floor, loud = np.percentile(levels, 10), np.percentile(levels, 95)
    # the 5th percentile: a take spoken with few pauses has its noise floor in under 10% of frames
    rel = np.percentile(levels, 5) * 10 ** (VAD_REL_DB / 20)
    # B-46 (2026-10-25): a quiet take is speech when it rises VAD_REL_DB over its own floor. Up to
    # 1.002 any take whose loudest frames stayed under VAD_QUIET was "no speech", which on a headset
    # at its default gain (speech about -41 dBFS, floor -76) dropped real takes. Quiet and flat
    # (silence, a muted mic) is still nothing.
    if loud < VAD_QUIET and loud < rel:
        return np.zeros(len(rms), bool)
    gate = max(VAD_ABS, floor + (loud - floor) / 3)
    if levels is not rms or loud < VAD_QUIET:
        gate = max(VAD_DEAD, min(gate, rel))
    return rms > gate


def has_speech(x: np.ndarray, ref: np.ndarray | None = None, sp: np.ndarray | None = None) -> bool:
    sp = _speech_frames(x, ref) if sp is None else sp
    return int(sp.sum()) * FRAME >= VAD_MIN_SPEECH_S * SAMPLE_RATE


def shorten_silences(x: np.ndarray, ref: np.ndarray | None = None, sp: np.ndarray | None = None) -> np.ndarray:
    """Cut every non-speech run longer than VAD_LONG_S down to VAD_KEEP_S (keeping its two ends).
    `sp`: the speech frames when the caller has them already (one Silero pass, not two: on a
    90-minute recording each pass is about 25 s)."""
    sp = _speech_frames(x, ref) if sp is None else sp
    n, keep = len(sp), int(VAD_KEEP_S * SAMPLE_RATE / FRAME)
    out, last, i = [], 0, 0
    while i < n:
        if sp[i]:
            i += 1
            continue
        j = i
        while j < n and not sp[j]:
            j += 1
        if (j - i) * FRAME > VAD_LONG_S * SAMPLE_RATE:
            out.append(x[last:(i + keep // 2) * FRAME])
            last = (j - keep // 2) * FRAME
        i = j
    out.append(x[last:])
    return np.concatenate(out) if out else x


def split_points(x: np.ndarray) -> list[tuple[int, int]]:
    """Cut a whole take the way live insertion cuts it while recording: grow the piece in
    0.5 s steps until live_cut finds a pause. Up to 0.3.x this cut every 30 s at the quietest
    100 ms, often inside a sentence, and lost 126 of 786 words on six 50 s test takes where
    pause cuts lost 9 (2026-10-21, docs/maintaining.md, "Live insertion")."""
    n, sr = len(x), SAMPLE_RATE
    out, start = [], 0
    while n - start > (PIECE_S + LIVE_FORCE_S) * sr:
        end = start + PIECE_S * sr
        while (cut := live_cut(x[start:end], PIECE_S)) is None:
            end += sr // 2
        out.append((start, start + cut))
        start += cut
    out.append((start, n))
    return out


# Live insertion: cut the take while it is still being recorded, at a pause of at least
# LIVE_PAUSE_S found in the last LIVE_SEARCH_S seconds (but not the newest LIVE_GUARD_S, which may
# hold the start of a word), once the take holds `interval` seconds. A pause of LIVE_BREAK_S or more
# (a sentence end, or stopping to think) cuts as soon as the take holds LIVE_MIN_S and some speech
# (0.7.1: Daniel, 2026-10-22, text came too late and "the insertion breaks don't make sense").
# With no pause by interval + LIVE_FORCE_S, cut at the quietest 100 ms.
LIVE_SEARCH_S = 6
LIVE_GUARD_S = 0.3
LIVE_PAUSE_S = 0.6      # 0.3 cut inside sentences: 24 vs 10 errors per 786 words at 15 s
LIVE_BREAK_S = 1.0
LIVE_MIN_S = 2.0
LIVE_FORCE_S = 15


def live_split(x: np.ndarray, interval_s: float) -> tuple[int, float] | None:
    """(sample index to cut a live take at, length of the pause cut in s; 0 for a forced cut),
    or None to keep listening. Quiet frames as in stretch_pauses; the cut is the middle of the
    longest quiet run, so both halves keep some silence."""
    sr, n = SAMPLE_RATE, len(x)
    if n < min(interval_s, LIVE_MIN_S) * sr:
        return None
    lo, hi = max(0, n - LIVE_SEARCH_S * sr), n - int(LIVE_GUARD_S * sr)
    m = n // FRAME
    rms = np.sqrt((x[: m * FRAME].reshape(m, FRAME) ** 2).mean(axis=1))
    floor, speech = np.percentile(rms, 10), np.percentile(rms, 90)
    quiet = rms < floor + 0.15 * (speech - floor)
    best, i, a, b = 0, lo // FRAME, 0, 0
    while i < hi // FRAME:
        if quiet[i]:
            j = i
            while j < hi // FRAME and quiet[j]:
                j += 1
            if j - i > best:
                best, a, b = j - i, i, j
            i = j
        else:
            i += 1
    pause, cut = best * FRAME / sr, (a + b) // 2 * FRAME
    if n >= interval_s * sr and pause >= LIVE_PAUSE_S:
        return cut, pause
    if pause >= LIVE_BREAK_S and a > 0 and has_speech(x[:cut]):
        return cut, pause
    if n >= (interval_s + LIVE_FORCE_S) * sr:
        win = sr // 10
        frames = x[lo:hi][: (hi - lo) // win * win].reshape(-1, win)
        return lo + int(np.argmin((frames ** 2).mean(axis=1))) * win + win // 2, 0.0
    return None


def live_cut(x: np.ndarray, interval_s: float) -> int | None:
    """live_split's cut alone."""
    r = live_split(x, interval_s)
    return r[0] if r else None


# Seams between live pieces (0.7.1). Parakeet decodes each piece on its own, so a piece cut at a
# pause inside a sentence comes back as a sentence: "...after the kids are asleep." then "So
# nothing else is running..." (2026-10-22 pacing run, docs/maintaining.md). The full stop of a
# piece cut at a pause shorter than LIVE_BREAK_S is therefore held back with the space after it;
# if the next piece starts with one of these words, the sentence goes on ("asleep so nothing"),
# otherwise the full stop is pasted in front of it, as before.
#
# Context (0.7.6, Daniel 2026-10-23: sentences "reaching sentence boundaries a bit too prematurely",
# "capitals in the middle of sentences"). In real dictation the pause rules above still gave about
# 1.2 capitalised function words a minute mid-sentence ("kind of Dumping", "the hardest part is. Is
# the room noise?"), most after a piece that ended with no mark at all, where nothing was held. So
# each live piece cut at a pause is decoded after the last CONTEXT_S of the piece before it, and
# only the words heard after that point are kept (token timestamps; the cut is the middle of a
# pause, so no word straddles it). The model then sees the sentence go on: it writes the first word
# in the case it has mid-sentence, and its punctuation at the seam replaces the guess the previous
# piece made without hearing what followed. Measured: desktop/docs/maintaining.md, "Context at the seams".
CONTEXT_S = 3.0
CONTINUES = frozenset("""and or but so because since which who whom whose that than then to of for
    with without in on at by from as if when while where whether after before until till unless
    about into onto like plus nor yet although though""".split())


