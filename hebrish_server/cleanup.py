"""Clean-up with S1-mini, as in the desktop app. Prompt, chunking and guard copied verbatim from
Hebrish-ASR/Hebrish-App desktop/hebrish/cleanup.py lines 53-193 (commit ca269f5, 2026-10-30).
The app runs its own llama-server; here the model is served by Ollama (`OllamaClient`), with the
same greedy settings. Ollama has no tokenize endpoint, so tokens are estimated from characters."""

import json
import re
import urllib.request
from dataclasses import dataclass

from . import config

# -- prompt (verbatim from ASR-Post-Processor-Model training/s1mini_lora.py) -------------------
SYSTEM_PROMPT = (
    "You are a text normalizer for speech-to-text transcripts. The input begins "
    "with a control line specifying the styling, structure, and context settings; "
    "clean the transcript to match those settings and output only the cleaned text."
)
CONTROL_LINE = "[Styling: semi-formal] [Structure: prose] [Context: general] [Layers: 1,2]"   # control_line("1,2")
ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
END_OF_TURN = "<|im_end|>"

MAX_INPUT_TOKENS = 900       # the model card: keep inputs under about 1,000 tokens
CTX = 4096                   # system ~60 + input <= 900 (a lone long sentence may exceed it) + output
IDLE_S = 600                 # stop llama-server after 10 minutes without a request
START_TIMEOUT_S = 90         # loading the 650 MB model from a cold disk


def render_prompt(transcript: str) -> str:
    """The full prompt up to and including the assistant prefix, byte-identical to
    s1mini_lora.render_prompt(transcript, "1,2")."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}{END_OF_TURN}\n"
        f"<|im_start|>user\n{CONTROL_LINE}\n{transcript}{END_OF_TURN}\n"
        f"{ASSISTANT_PREFIX}"
    )


def max_tokens(input_tokens: int) -> int:
    """The output ceiling the S1-mini card recommends: 1.3 x input tokens + 32."""
    return int(1.3 * input_tokens) + 32


def timeout_for(n_predict: int) -> float:
    """Seconds one chunk may take: generous for a slow laptop CPU (~10 tokens/s), then give up."""
    return 30.0 + n_predict / 10.0


# -- chunks and the guard ------------------------------------------------------------------

_SENTENCE = re.compile(r"(?<=[.!?…])\s+|\n\s*\n")


def sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENTENCE.split(text)) if s]


def split_chunks(text: str, count, limit: int = MAX_INPUT_TOKENS) -> list[str]:
    """`text` as chunks of whole sentences of at most `limit` tokens each (count(str) -> int).
    A single sentence over the limit is a chunk of its own: never cut mid-sentence."""
    text = text.strip()
    if not text:
        return []
    if count(text) <= limit:
        return [text]
    out, cur, have = [], [], 0
    for s in sentences(text):
        n = count(s) + 1
        if cur and have + n > limit:
            out.append(" ".join(cur))
            cur, have = [], 0
        cur.append(s)
        have += n
    if cur:
        out.append(" ".join(cur))
    return out


def _has(text: str, word: str) -> bool:
    """`word` in `text` as a whole word: exact case for a word with a capital (the dictionary
    writes those as given), any case otherwise."""
    flags = 0 if word != word.lower() else re.IGNORECASE
    pat = r"\s+".join(re.escape(p) for p in word.split())
    return re.search(r"(?<!\w)" + pat + r"(?!\w)", text, flags) is not None


def accept(raw: str, out: str, stop: str = "eos", keep: tuple = ()) -> bool:
    """Whether the model's `out` may replace `raw`: not empty, not cut off by the token limit
    (`stop` == "limit"), and every word of `keep` (the personal dictionary) that is in `raw` is
    still in `out` (research/post-processing, "a guard for S1-mini")."""
    if not out.strip() and raw.strip():
        return False
    if stop == "limit":
        return False
    return all(_has(out, w) for w in keep if _has(raw, w))


def tidy(out: str) -> str:
    """The model's reply as text: the end-of-turn token and any think block removed, and
    paragraphs separated by exactly one blank line."""
    out = out.replace(END_OF_TURN, "")
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", out)]
    return "\n\n".join(p for p in paras if p)


@dataclass
class Result:
    text: str          # what goes in the transcript: cleaned, or raw where a chunk fell back
    chunks: int        # chunks sent to the model
    kept_raw: int      # chunks that fell back to their raw text
    error: str = ""    # the first failure, for the status line
    model: str = ""    # the catalogue id and version that ran (run() sets them), for History
    version: str = ""
    name: str = ""     # its display name (label())

    @property
    def cleaned(self) -> bool:
        """At least one chunk was cleaned."""
        return self.chunks > self.kept_raw


def clean(text: str, client, keep: tuple = ()) -> Result:
    """Clean `text` with `client` (`count(str) -> int`, `complete(prompt, n_predict, timeout) ->
    (content, stop_type)`; `Server` in the app, a fake in the tests). Chunks are joined with a
    paragraph break. Any failure keeps that chunk's raw text, so nothing is lost; a failure to
    count tokens at all returns `text` unchanged."""
    raw = text.strip()
    if not raw:
        return Result(text, 0, 0)
    try:
        chunks = split_chunks(raw, client.count)
    except Exception as e:
        return Result(text, 1, 1, str(e) or type(e).__name__)
    out, kept, first = [], 0, ""
    for chunk in chunks:
        try:
            n_predict = max_tokens(client.count(chunk))
            content, stop = client.complete(render_prompt(chunk), n_predict, timeout_for(n_predict))
            got = tidy(content)
            if accept(chunk, got, stop, keep):
                out.append(got)
                continue
            why = ("cut off" if stop == "limit" else "empty reply" if not got else
                   "changed a dictionary word")
        except Exception as e:
            why = str(e) or type(e).__name__
        kept += 1
        first = first or why
        out.append(chunk)
    return Result("\n\n".join(out), len(chunks), kept, first)




class OllamaClient:
    """`count` and `complete` for clean() against Ollama's /api/generate with a raw prompt."""

    def __init__(self, url: str = config.CLEANUP_URL, model: str = config.CLEANUP_MODEL):
        self.url, self.model = url.rstrip("/"), model

    def count(self, text: str) -> int:
        return max(1, round(len(text) / 3.5))     # over-estimates English for Qwen3's tokenizer: safe side

    def complete(self, prompt: str, n_predict: int, timeout: float) -> tuple[str, str]:
        body = {"model": self.model, "prompt": prompt, "raw": True, "stream": False,
                "options": {"num_predict": n_predict, "temperature": 0, "top_k": 1,
                            "num_ctx": CTX, "stop": [END_OF_TURN]}}
        req = urllib.request.Request(self.url + "/api/generate", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        return d.get("response", ""), "limit" if d.get("done_reason") == "length" else "eos"

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/api/tags", timeout=3) as r:
                names = {m["name"].split(":")[0] for m in json.load(r).get("models", [])}
            return self.model.split(":")[0] in names
        except Exception:
            return False
