"""HTTP API and the web UI. No authentication: bind it to localhost, a LAN or a tailnet only."""

import logging
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import cleanup, config, engines
from .jobs import Jobs

log = logging.getLogger("hebrish_server")
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Hebrish Server", version="0.1.0",
              description="Hebrish model transcription (Sabich by default), paragraphs and clean-up on your own hardware.")
state: dict = {"engine": None, "jobs": None, "error": None}


def _load():
    try:
        e = engines.load()
        state["engine"], state["jobs"] = e, Jobs(e)
        log.info("engine ready: %s", e.info)
    except Exception as e:
        log.exception("engine failed to load")
        state["error"] = f"{type(e).__name__}: {e}"


@app.on_event("startup")
def _startup():
    threading.Thread(target=_load, daemon=True).start()     # the UI is up while the model loads


def _jobs() -> Jobs:
    if state["jobs"] is None:
        raise HTTPException(503, state["error"] or "model is still loading")
    return state["jobs"]


@app.get("/api/health")
def health():
    e = state["engine"]
    return {"ready": e is not None, "error": state["error"], "engine": e.info if e else {"engine": config.ENGINE},
            "cleanup": {"url": config.CLEANUP_URL, "model": config.CLEANUP_MODEL,
                        "available": cleanup.OllamaClient().available()}}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), paragraphs: str = Form("none"), cleanup_: bool = Form(False, alias="cleanup")):
    try:
        return _jobs().submit(await file.read(), file.filename or "audio", paragraphs, cleanup_)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/jobs")
def list_jobs():
    return _jobs().list()


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    j = _jobs().get(jid)
    if j is None:
        raise HTTPException(404, "no such job (jobs are kept in memory and lost on restart)")
    return j


class CleanupIn(BaseModel):
    text: str


@app.post("/api/cleanup")
def cleanup_text(body: CleanupIn):
    r = cleanup.clean(body.text, cleanup.OllamaClient())
    return {"text": r.text, "chunks": r.chunks, "kept_raw": r.kept_raw, "error": r.error}


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(file: UploadFile = File(...), model: str = Form("sabich"),
                                response_format: str = Form("json"), paragraphs: str = Form("none"),
                                cleanup_: bool = Form(False, alias="cleanup")):
    """OpenAI-compatible: `model` is accepted and ignored (the server runs one model)."""
    jobs = _jobs()
    j = jobs.submit(await file.read(), file.filename or "audio", paragraphs, cleanup_)
    j = jobs.wait(j["id"])
    if j["status"] == "error":
        raise HTTPException(422, j["error"])
    if response_format == "text":
        return PlainTextResponse(j["text"])
    if response_format == "verbose_json":
        return {"text": j["text"], "duration": j.get("audio_s"), "model": j["model"], "language": "en"}
    return {"text": j["text"]}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
