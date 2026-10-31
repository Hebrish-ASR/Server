const $ = (id) => document.getElementById(id);
let current = null, showRaw = false;

async function health() {
  try {
    const h = await (await fetch("/api/health")).json();
    const e = h.engine;
    if (h.error) { $("status").innerHTML = `<span class="err">Model failed to load: ${h.error}</span>`; return; }
    if (!h.ready) { $("status").textContent = `Loading the model (${e.engine})…`; setTimeout(health, 2000); return; }
    const dev = e.gpu ? `${e.gpu} (${e.gpu_backend})` : `${e.device}${e.threads ? ", " + e.threads + " threads" : ""}`;
    $("status").textContent = `${e.model} on ${e.engine}, ${dev}. Clean-up: ${h.cleanup.available ? h.cleanup.model : "not available"}.`;
    $("cleanup").disabled = !h.cleanup.available;
  } catch { $("status").innerHTML = `<span class="err">Server not reachable.</span>`; }
}

async function submit(blob, name) {
  const fd = new FormData();
  fd.append("file", blob, name);
  fd.append("paragraphs", $("paragraphs").value);
  fd.append("cleanup", $("cleanup").checked);
  const r = await fetch("/api/jobs", { method: "POST", body: fd });
  const j = await r.json();
  if (!r.ok) { $("job-meta").innerHTML = `<span class="err">${j.detail}</span>`; return; }
  current = j.id; showRaw = false; $("out").value = "";
  poll(j.id); jobs();
}

async function poll(id) {
  const j = await (await fetch(`/api/jobs/${id}`)).json();
  if (id !== current) return;
  render(j);
  if (["queued", "running", "cleaning"].includes(j.status)) setTimeout(() => poll(id), 700);
  else jobs();
}

function render(j) {
  const pct = Math.round((j.progress || 0) * 100);
  let meta = `${j.file}: ${j.status}${j.status === "running" ? " " + pct + "%" : ""}`;
  if (j.status === "done") meta += `, ${fmt(j.audio_s)} of audio in ${j.decode_s}s (${j.realtime_x}× realtime)` + (j.cleanup_s ? `, clean-up ${j.cleanup_s}s` : "");
  if (j.cleanup_result?.error) meta += `; clean-up kept raw text for ${j.cleanup_result.kept_raw} of ${j.cleanup_result.chunks} chunks (${j.cleanup_result.error})`;
  $("job-meta").innerHTML = j.error ? `<span class="err">${j.error}</span>` : meta;
  $("out").value = showRaw && j.raw_text ? j.raw_text : j.text || "";
  $("raw").hidden = !j.raw_text;
  $("raw").textContent = showRaw ? "Show cleaned text" : "Show raw transcript";
}

const fmt = (s) => s == null ? "" : s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${Math.round(s)}s`;

async function jobs() {
  const r = await fetch("/api/jobs");
  if (!r.ok) return;
  $("jobs").innerHTML = (await r.json()).map((j) =>
    `<tr class="click" data-id="${j.id}"><td>${j.file}</td><td>${j.status}</td><td>${fmt(j.audio_s)}</td>` +
    `<td>${j.realtime_x ? j.realtime_x + "×" : ""}</td><td>${j.paragraphs}${j.cleanup ? ", clean-up" : ""}</td></tr>`).join("");
  document.querySelectorAll("#jobs tr").forEach((tr) => tr.onclick = () => { current = tr.dataset.id; showRaw = false; poll(current); });
}

$("file").onchange = (e) => e.target.files[0] && submit(e.target.files[0], e.target.files[0].name);
const drop = $("drop");
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); const f = e.dataTransfer.files[0]; if (f) submit(f, f.name); };
$("copy").onclick = () => navigator.clipboard?.writeText($("out").value);
$("raw").onclick = () => { showRaw = !showRaw; poll(current); };

let rec = null;
$("rec").onclick = async () => {
  if (rec) { rec.stop(); return; }
  if (!navigator.mediaDevices) { $("rec-note").textContent = "The browser allows the microphone only over HTTPS or on localhost; upload a file instead."; return; }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const chunks = [];
  rec = new MediaRecorder(stream);
  rec.ondataavailable = (e) => chunks.push(e.data);
  rec.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    rec = null; $("rec").textContent = "● Record"; $("rec-note").textContent = "";
    submit(new Blob(chunks, { type: chunks[0]?.type || "audio/webm" }), `recording-${new Date().toISOString().slice(11, 19)}.webm`);
  };
  rec.start(); $("rec").textContent = "■ Stop"; $("rec-note").textContent = "Recording…";
};

health(); jobs();
