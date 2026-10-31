"""In-memory job queue with one worker thread: one decode at a time, so a GPU is never shared
between two requests. Jobs are lost on restart (proof of concept: no database)."""

import queue
import threading
import time
import uuid
from collections import OrderedDict

from . import audio, cleanup, config, pipeline


class Jobs:
    def __init__(self, engine):
        self.engine = engine
        self._jobs: OrderedDict[str, dict] = OrderedDict()
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        threading.Thread(target=self._work, daemon=True, name="jobs").start()

    def submit(self, data: bytes, filename: str, paragraphs: str, clean_up: bool) -> dict:
        if paragraphs not in pipeline.MODES:
            raise ValueError(f"paragraphs must be one of {pipeline.MODES}")
        job = {"id": uuid.uuid4().hex[:12], "file": filename, "status": "queued", "progress": 0.0,
               "paragraphs": paragraphs, "cleanup": clean_up, "created": time.time(),
               "text": "", "raw_text": None, "error": None, "model": self.engine.info["model"]}
        with self._lock:
            self._jobs[job["id"]] = job
            while len(self._jobs) > config.MAX_JOBS_KEPT:
                self._jobs.popitem(last=False)
        self._q.put((job["id"], data))
        return job

    def get(self, jid: str) -> dict | None:
        return self._jobs.get(jid)

    def list(self) -> list[dict]:
        return [{k: v for k, v in j.items() if k not in ("text", "raw_text")} for j in reversed(self._jobs.values())]

    def wait(self, jid: str, timeout: float = 3600) -> dict:
        t0 = time.monotonic()
        while (j := self._jobs[jid])["status"] in ("queued", "running", "cleaning"):
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(jid)
            time.sleep(0.2)
        return j

    def _work(self):
        while True:
            jid, data = self._q.get()
            job = self._jobs.get(jid)
            if job is None:
                continue
            job["status"], job["started"] = "running", time.time()
            try:
                samples = audio.decode(data)

                def progress(frac, text):
                    job["progress"], job["text"] = round(frac, 3), text
                job.update(pipeline.transcribe(self.engine, samples, job["paragraphs"], progress))
                if job["cleanup"] and job["text"]:
                    job["status"], job["raw_text"] = "cleaning", job["text"]
                    t0 = time.monotonic()
                    r = cleanup.clean(job["text"], cleanup.OllamaClient())
                    job["text"], job["cleanup_s"] = r.text, round(time.monotonic() - t0, 2)
                    job["cleanup_result"] = {"chunks": r.chunks, "kept_raw": r.kept_raw, "error": r.error}
                job["status"], job["progress"] = "done", 1.0
            except Exception as e:
                job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
            job["finished"] = time.time()
