"""
Koba Document Scanner — app.py
=================================
Flask web application.  Run:
    python app.py
Then open  http://localhost:5000  in your browser.

Requirements (beyond processor.py deps):
    pip install flask
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
import queue
from pathlib import Path

from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_file,
    url_for,
)

from processor import Config, process_pdf

# ── app setup ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB upload limit

WORK_ROOT = Path(tempfile.mkdtemp(prefix="koba_scanner_"))
JOBS: dict[str, dict] = {}   # job_id → state dict
log = logging.getLogger("scanner.web")

job_queue = queue.Queue()

def _worker_loop():
    while True:
        job_id = job_queue.get()
        if job_id is None:
            break
        try:
            if JOBS[job_id].get("state") == "cancelled":
                continue
            _run_job_sync(job_id)
        except Exception as e:
            log.error(f"Job {job_id} failed: {e}")
            JOBS[job_id]["state"] = "error"
            JOBS[job_id]["error"] = str(e)
        finally:
            job_queue.task_done()

threading.Thread(target=_worker_loop, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# HTML template (single-file, no external template directory needed)
# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Koba Document Scanner</title>
<style>
/* ── reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: #12131a;
  color: #dde1f0;
  min-height: 100vh;
}

/* ── header ── */
.header {
  background: linear-gradient(135deg, #1c1e2e 0%, #0e2244 100%);
  border-bottom: 2px solid #c9542a;
  padding: 1.6rem 2rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}
.header-icon { font-size: 2rem; }
.header h1  { font-size: 1.5rem; color: #f0a070; letter-spacing: 1px; }
.header p   { font-size: .85rem; color: #8090b0; margin-top: .2rem; }

/* ── layout ── */
.container { max-width: 860px; margin: 2rem auto; padding: 0 1.2rem; }

/* ── card ── */
.card {
  background: #1a1d2e;
  border: 1px solid #2a3050;
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1.4rem;
}
.card-title {
  font-size: .78rem;
  font-weight: 700;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: #c9542a;
  margin-bottom: 1rem;
}

/* ── drop zone ── */
.drop-zone {
  border: 2px dashed #c9542a;
  border-radius: 10px;
  padding: 2.5rem;
  text-align: center;
  cursor: pointer;
  transition: background .15s, border-color .15s;
  background: #0e111c;
}
.drop-zone:hover, .drop-zone.over {
  background: #151828;
  border-color: #f0a070;
}
.drop-zone .dz-icon { font-size: 2.8rem; margin-bottom: .6rem; }
.drop-zone .dz-main { color: #c0cce0; font-size: 1rem; }
.drop-zone .dz-hint { color: #50607a; font-size: .8rem; margin-top: .4rem; }
#file-input { display: none; }

/* ── settings grid ── */
.settings-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: .9rem 1.4rem;
}
@media (max-width: 540px) { .settings-grid { grid-template-columns: 1fr; } }

label { display: block; font-size: .8rem; color: #8090b0; margin-bottom: .25rem; }
select, input[type=number], input[type=range] {
  width: 100%;
  padding: .55rem .7rem;
  background: #0e111c;
  border: 1px solid #2a3050;
  border-radius: 7px;
  color: #dde1f0;
  font-size: .9rem;
  outline: none;
  transition: border-color .15s;
}
select:focus, input:focus { border-color: #c9542a; }
.range-row { display: flex; align-items: center; gap: .7rem; }
.range-row input[type=range] { flex: 1; }
.range-val { min-width: 3rem; text-align: right; font-size: .85rem; color: #f0a070; }

/* ── button ── */
.btn {
  display: inline-flex; align-items: center; gap: .5rem;
  padding: .7rem 1.8rem;
  border: none; border-radius: 8px;
  font-size: .95rem; font-weight: 600;
  cursor: pointer;
  transition: filter .15s, opacity .15s;
}
.btn:disabled { opacity: .4; cursor: not-allowed; }
.btn-primary  { background: #c9542a; color: #fff; }
.btn-primary:not(:disabled):hover  { filter: brightness(1.15); }
.btn-download { background: #1a6640; color: #fff; text-decoration: none; }
.btn-download:not([hidden]):hover  { filter: brightness(1.2); }

/* ── progress ── */
.progress-wrap {
  background: #0e111c;
  border-radius: 99px;
  height: 8px;
  overflow: hidden;
  margin: .8rem 0;
}
.progress-fill {
  height: 100%;
  border-radius: 99px;
  background: linear-gradient(90deg, #c9542a, #f0a070);
  transition: width .4s ease;
  width: 0%;
}
.status-line {
  display: flex; align-items: center; gap: .6rem;
  color: #a0b0c8; font-size: .88rem;
}

/* ── spinner ── */
.spin {
  display: inline-block;
  width: 14px; height: 14px;
  border: 2px solid #c9542a44;
  border-top-color: #c9542a;
  border-radius: 50%;
  animation: spin .7s linear infinite;
  flex-shrink: 0;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── log box ── */
.log-box {
  background: #080b14;
  border: 1px solid #1e2540;
  border-radius: 8px;
  padding: .8rem 1rem;
  max-height: 180px;
  overflow-y: auto;
  font-family: 'Consolas', 'Fira Code', monospace;
  font-size: .75rem;
  color: #70c090;
  white-space: pre-wrap;
  word-break: break-all;
  margin-top: .8rem;
}

/* ── results ── */
.summary-row {
  display: flex; flex-wrap: wrap; gap: .6rem;
  align-items: center; margin-bottom: 1rem;
}
.chip {
  background: #0e111c;
  border: 1px solid #2a3050;
  border-radius: 99px;
  padding: .2rem .75rem;
  font-size: .8rem;
  color: #a0d0a0;
}

.page-table { width: 100%; border-collapse: collapse; font-size: .83rem; }
.page-table th {
  background: #0e111c;
  color: #7090c0;
  padding: .5rem .7rem;
  text-align: left;
  font-weight: 600;
}
.page-table td { padding: .45rem .7rem; border-bottom: 1px solid #1e2540; }
.badge {
  display: inline-block;
  padding: .1rem .5rem;
  border-radius: 99px;
  font-size: .72rem;
  font-weight: 700;
}
.ok   { background: #1a4a2a; color: #60d080; }
.fail { background: #4a1a1a; color: #e06060; }

.hidden { display: none !important; }

/* ── settings collapse ── */
#settings-toggle {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 1rem;
  margin-left: auto;
  color: #c9542a;
  padding: 0;
  transition: opacity .2s;
}
#settings-toggle:hover { opacity: .7; }

#settings-container.collapsed {
  display: none !important;
}
</style>
</head>
<body>

<div class="header">
  <div class="header-icon">🏛️</div>
  <div>
    <h1>Koba Document scanner</h1>
  </div>
</div>

<div class="container">

  <!-- ── upload ── -->
  <div class="card">
    <div class="card-title">📄 Upload Scanned PDF</div>
    <div class="drop-zone" id="dz" onclick="document.getElementById('file-input').click()">
      <div class="dz-icon">📂</div>
      <div class="dz-main" id="dz-label">Drop your PDF here or click to browse</div>
      <div class="dz-hint">Multi-page PDFs supported · max 500 MB</div>
    </div>
    <input type="file" id="file-input" accept=".pdf">
  </div>

  <!-- ── settings ── -->
  <div class="card">
    <div class="card-title" style="display:flex;align-items:center;gap:.5rem">
      <span>⚙️ Settings</span>
      <button id="settings-toggle">⚙️</button>
    </div>
    <div id="settings-container" class="settings-grid collapsed">

      <div>
        <label>Render DPI</label>
        <select id="s-dpi">
          <option value="200">200 — Fast preview</option>
          <option value="300" selected>300 — Recommended</option>
          <option value="400">400 — High quality</option>
          <option value="600">600 — Archival</option>
        </select>
      </div>

      <div>
        <label>Skew Detection Method</label>
        <select id="s-method">
          <option value="hough">Hough Lines — fast, good for text lines</option>
          <option value="projection" selected>Projection Profile — slow, best for dense handwriting</option>
        </select>
      </div>

      <div>
        <label>Max Skew Angle: <span id="skew-val">15</span>°</label>
        <div class="range-row">
          <input type="range" id="s-maxskew" min="2" max="45" value="15"
                 oninput="document.getElementById('skew-val').textContent=this.value">
          <span class="range-val" id="skew-val2"></span>
        </div>
      </div>

      <div>
        <label>Border Padding: <span id="pad-val">2.0</span> cm</label>
        <div class="range-row">
          <input type="range" id="s-padding" min="0" max="5" step="0.5" value="2"
                 oninput="document.getElementById('pad-val').textContent=parseFloat(this.value).toFixed(1)">
        </div>
      </div>

      <div>
        <label>Output Format</label>
        <select id="s-outfmt">
          <option value="both">PDF + individual images</option>
          <option value="pdf" selected>PDF only</option>
          <option value="images">Images only</option>
        </select>
      </div>

      <div>
        <label>Image Format</label>
        <select id="s-imgfmt" onchange="document.getElementById('jpeg-quality-container').style.display = this.value === 'jpg' ? 'block' : 'none'">
          <option value="png" selected>PNG (lossless)</option>
          <option value="jpg">JPEG (smaller)</option>
          <option value="tiff">TIFF (archival)</option>
        </select>
      </div>

      <div id="jpeg-quality-container" style="display:none">
        <label>JPEG Quality: <span id="jq-val">85</span>%</label>
        <div class="range-row">
          <input type="range" id="s-jq" min="10" max="100" value="85"
                 oninput="document.getElementById('jq-val').textContent=this.value">
        </div>
      </div>

    </div><!-- /settings-container -->

    <div style="margin-top:1.2rem; display:flex; gap:.8rem; align-items:center; flex-wrap:wrap">
      <button class="btn btn-primary" id="run-btn" onclick="startJob()" disabled>
        ⚡ Process PDF
      </button>
      <label style="display:flex;align-items:center;gap:.4rem;cursor:pointer;font-size:.82rem;color:#7080a0">
        <input type="checkbox" id="s-debug"> Save debug images
      </label>
    </div>
  </div>

  <!-- ── progress ── -->
  <div class="card hidden" id="progress-card">
    <div class="card-title">🔄 Processing</div>
    <div class="status-line">
      <span class="spin" id="spin"></span>
      <span id="status-text">Starting…</span>
      <div style="margin-left:auto; display:flex; gap:0.5rem">
        <button class="btn" id="btn-pause" onclick="jobAction('pause')" style="padding:0.4rem 0.8rem; font-size:0.8rem">⏸ Pause</button>
        <button class="btn" id="btn-resume" onclick="jobAction('resume')" style="display:none; padding:0.4rem 0.8rem; font-size:0.8rem">▶ Resume</button>
        <button class="btn" id="btn-cancel" onclick="jobAction('cancel')" style="padding:0.4rem 0.8rem; font-size:0.8rem; background:#4a1a1a;">⏹ Cancel</button>
      </div>
    </div>
    <div class="progress-wrap">
      <div class="progress-fill" id="pbar"></div>
    </div>
    <div class="log-box" id="log-box"></div>
  </div>

  <!-- ── results ── -->
  <div class="card hidden" id="results-card">
    <div class="card-title">✅ Results</div>
    <div class="summary-row" id="chips"></div>
    <div style="margin-bottom:1.2rem">
      <a class="btn btn-download" id="dl-btn" href="#" hidden>⬇️ Download Cleaned PDF</a>
    </div>
    <table class="page-table">
      <thead>
        <tr>
          <th>Page</th>
          <th>Status</th>
          <th>Skew corrected</th>
          <th>Crop box</th>
        </tr>
      </thead>
      <tbody id="ptable"></tbody>
    </table>
  </div>

</div><!-- /container -->

<script>
// ── state ──────────────────────────────────────────────────────────────────
let selectedFile = null;
let pollId       = null;
let activeJobId  = null;

// ── drag-and-drop ──────────────────────────────────────────────────────────
const dz = document.getElementById('dz');
dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('over'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('over');
  const f = e.dataTransfer.files[0];
  if (f) pickFile(f);
});
document.getElementById('file-input').addEventListener('change', e => {
  if (e.target.files[0]) pickFile(e.target.files[0]);
});

// ── settings toggle ────────────────────────────────────────────────────────
document.getElementById('settings-toggle').addEventListener('click', () => {
  const container = document.getElementById('settings-container');
  container.classList.toggle('collapsed');
});

// On page load, collapse settings by default
document.addEventListener('DOMContentLoaded', () => {
  const container = document.getElementById('settings-container');
  if (container && !container.classList.contains('collapsed')) {
    container.classList.add('collapsed');
  }
});

function pickFile(f) {
  if (!f.name.toLowerCase().endsWith('.pdf')) {
    alert('Please select a PDF file.'); return;
  }
  selectedFile = f;
  document.getElementById('dz-label').textContent =
    '✅  ' + f.name + '   (' + (f.size / 1048576).toFixed(2) + ' MB)';
  document.getElementById('run-btn').disabled = false;
}

// ── start job ──────────────────────────────────────────────────────────────
async function startJob() {
  if (!selectedFile) return;

  // reset UI
  setCardVisible('progress-card', true);
  setCardVisible('results-card',  false);
  document.getElementById('run-btn').disabled = true;
  document.getElementById('spin').style.display = '';
  document.getElementById('log-box').textContent = '';
  document.getElementById('pbar').style.width = '3%';
  document.getElementById('status-text').textContent = 'Uploading…';

  const fd = new FormData();
  fd.append('file',       selectedFile);
  fd.append('dpi',        document.getElementById('s-dpi').value);
  fd.append('method',     document.getElementById('s-method').value);
  fd.append('max_skew',   document.getElementById('s-maxskew').value);
  fd.append('padding_cm', document.getElementById('s-padding').value);
  fd.append('out_format', document.getElementById('s-outfmt').value);
  fd.append('img_format', document.getElementById('s-imgfmt').value);
  fd.append('jpeg_quality', document.getElementById('s-jq').value);
  fd.append('debug',      document.getElementById('s-debug').checked ? '1' : '0');

  try {
    const res  = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok || data.error) { showError(data.error || 'Upload failed'); return; }
    activeJobId = data.job_id;
    startPolling(activeJobId);
  } catch (err) {
    showError(err.message);
  }
}

// ── polling ────────────────────────────────────────────────────────────────
function startPolling(jobId) {
  if (pollId) clearInterval(pollId);
  pollId = setInterval(() => pollJob(jobId), 1200);
  
  // reset buttons
  document.getElementById('btn-pause').style.display = 'inline-flex';
  document.getElementById('btn-resume').style.display = 'none';
  document.getElementById('btn-cancel').style.display = 'inline-flex';
}

async function jobAction(action) {
  if (!activeJobId) return;
  try {
    const res = await fetch('/api/action/' + activeJobId, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action})
    });
    if (!res.ok) console.error(await res.text());
  } catch(err) {
    console.error(err);
  }
}

async function pollJob(jobId) {
  try {
    const res  = await fetch('/api/status/' + jobId);
    const data = await res.json();
    applyStatus(data, jobId);
    if (data.state === 'done' || data.state === 'error' || data.state === 'cancelled') {
      clearInterval(pollId);
    }
  } catch (_) { /* transient network error, retry next tick */ }
}

function applyStatus(data, jobId) {
  // log
  const lb = document.getElementById('log-box');
  if (data.log && data.log.length) {
    lb.textContent = data.log.join('\n');
    lb.scrollTop = lb.scrollHeight;
  }

  if (data.state === 'queued') {
    document.getElementById('status-text').textContent = 'Waiting in queue…';
  } else if (data.state === 'running') {
    const pct = Math.min(90, 5 + (data.done_pages || 0) / Math.max(data.total_pages || 1, 1) * 85);
    document.getElementById('pbar').style.width = pct + '%';
    document.getElementById('status-text').textContent =
      `Processing page ${data.done_pages || 0} / ${data.total_pages || '?'}…`;
    document.getElementById('btn-pause').style.display = 'inline-flex';
    document.getElementById('btn-resume').style.display = 'none';
    document.getElementById('spin').style.display = 'inline-block';
  } else if (data.state === 'paused') {
    document.getElementById('status-text').textContent = 'Paused';
    document.getElementById('btn-pause').style.display = 'none';
    document.getElementById('btn-resume').style.display = 'inline-flex';
    document.getElementById('spin').style.display = 'none';
  } else if (data.state === 'cancelled') {
    document.getElementById('status-text').textContent = '❌  Cancelled';
    document.getElementById('btn-pause').style.display = 'none';
    document.getElementById('btn-resume').style.display = 'none';
    document.getElementById('btn-cancel').style.display = 'none';
    document.getElementById('run-btn').disabled = false;
    document.getElementById('spin').style.display = 'none';
  } else if (data.state === 'done') {
    document.getElementById('pbar').style.width = '100%';
    document.getElementById('spin').style.display = 'none';
    document.getElementById('status-text').textContent = '✅  Done!';
    document.getElementById('run-btn').disabled = false;
    document.getElementById('btn-pause').style.display = 'none';
    document.getElementById('btn-resume').style.display = 'none';
    document.getElementById('btn-cancel').style.display = 'none';
    showResults(data, jobId);

  } else if (data.state === 'error') {
    document.getElementById('spin').style.display = 'none';
    document.getElementById('status-text').textContent = '❌  ' + (data.error || 'Unknown error');
    document.getElementById('run-btn').disabled = false;
    document.getElementById('btn-pause').style.display = 'none';
    document.getElementById('btn-resume').style.display = 'none';
    document.getElementById('btn-cancel').style.display = 'none';
  }
}

// ── results ────────────────────────────────────────────────────────────────
function showResults(data, jobId) {
  setCardVisible('results-card', true);

  // chips
  const chips = document.getElementById('chips');
  chips.innerHTML = '';
  addChip(chips, `${data.ok_pages} / ${data.total_pages} pages OK`);
  if (data.has_pdf) addChip(chips, 'PDF ready');

  // download button
  const dlBtn = document.getElementById('dl-btn');
  if (data.has_pdf) {
    dlBtn.href   = '/api/download/' + jobId;
    dlBtn.hidden = false;
  }

  // table
  const tbody = document.getElementById('ptable');
  tbody.innerHTML = '';
  (data.page_results || []).forEach(r => {
    const tr = document.createElement('tr');
    const skewStr = r.success ? (r.skew >= 0 ? '+' : '') + r.skew.toFixed(2) + '°' : '—';
    const cropStr = r.crop
      ? `${r.crop[0]},${r.crop[1]}  ${r.crop[2]}×${r.crop[3]}`
      : (r.success ? 'full page' : '—');
    tr.innerHTML = `
      <td>${r.page}</td>
      <td><span class="badge ${r.success ? 'ok' : 'fail'}">${r.success ? 'OK' : 'FAIL'}</span></td>
      <td>${skewStr}</td>
      <td style="font-family:monospace;font-size:.78rem">${cropStr}</td>`;
    tbody.appendChild(tr);
  });
}

function addChip(parent, text) {
  const span = document.createElement('span');
  span.className = 'chip';
  span.textContent = text;
  parent.appendChild(span);
}

// ── helpers ────────────────────────────────────────────────────────────────
function setCardVisible(id, visible) {
  document.getElementById(id).classList.toggle('hidden', !visible);
}

function showError(msg) {
  document.getElementById('spin').style.display = 'none';
  document.getElementById('status-text').textContent = '❌  ' + msg;
  document.getElementById('run-btn').disabled = false;
}
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Background job
# ─────────────────────────────────────────────────────────────────────────────

def _run_job_sync(job_id: str) -> None:
    job = JOBS[job_id]
    if job.get("state") == "cancelled":
        return
    job["state"] = "running"
    
    log_lines: list[str] = []

    def log_cb(msg: str) -> None:
        log_lines.append(msg)
        job["log"] = log_lines[-120:]

    def progress_cb(done: int, total: int) -> None:
        job["done_pages"]  = done
        job["total_pages"] = total

    try:
        results = process_pdf(
            job["pdf_path"], 
            job["out_dir"], 
            job["cfg"], 
            progress_cb=progress_cb,
            job_control=job,
            log_cb=log_cb
        )

        if job.get("state") == "cancelled":
            return

        # Find output PDF
        out_pdf = next(Path(job["out_dir"]).glob("*_cleaned.pdf"), None)

        job.update(
            state        = "done",
            ok_pages     = sum(1 for r in results if r.success),
            total_pages  = len(results),
            has_pdf      = out_pdf is not None,
            output_pdf   = str(out_pdf) if out_pdf else None,
            page_results = [
                {
                    "page":    r.page_num,
                    "success": r.success,
                    "skew":    r.skew_angle,
                    "crop":    list(r.crop_box) if r.crop_box else None,
                    "error":   r.error,
                }
                for r in results
            ],
        )

    except Exception as exc:
        import traceback
        job.update(state="error", error=str(exc))
        log_lines.append(traceback.format_exc())
        job["log"] = log_lines[-120:]


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify(error="No file part in request"), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify(error="No file selected"), 400
    if not f.filename.lower().endswith(".pdf"):
        return jsonify(error="Only PDF files are accepted"), 400

    job_id  = uuid.uuid4().hex
    job_dir = WORK_ROOT / job_id
    job_dir.mkdir(parents=True)

    pdf_path = job_dir / "input.pdf"
    f.save(str(pdf_path))

    out_dir = str(job_dir / "output")

    cfg = Config(
        dpi           = int(request.form.get("dpi", 300)),
        skew_method   = request.form.get("method", "projection"),
        max_skew_deg  = float(request.form.get("max_skew", 15)),
        padding_cm    = float(request.form.get("padding_cm", 2.0)),
        output_format = request.form.get("out_format", "pdf"),
        image_format  = request.form.get("img_format", "png"),
        jpeg_quality  = int(request.form.get("jpeg_quality", 85)),
        save_debug    = request.form.get("debug", "0") == "1",
    )

    JOBS[job_id] = {
        "state":       "queued",
        "done_pages":  0,
        "total_pages": 0,
        "log":         [],
        "pdf_path":    str(pdf_path),
        "out_dir":     out_dir,
        "cfg":         cfg,
    }

    job_queue.put(job_id)

    return jsonify(job_id=job_id)


@app.route("/api/action/<job_id>", methods=["POST"])
def api_action(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify(error="Job not found"), 404
    
    action = request.json.get("action")
    if action == "pause":
        if job["state"] == "running":
            job["state"] = "paused"
    elif action == "resume":
        if job["state"] == "paused":
            job["state"] = "running"
    elif action == "cancel":
        if job["state"] in ("queued", "running", "paused"):
            job["state"] = "cancelled"
            job["log"].append("Job cancelled by user.")
            
    return jsonify(ok=True, state=job["state"])

@app.route("/api/status/<job_id>")
def api_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify(error="Job not found"), 404
    # Return everything except the internal output_pdf path
    safe = {k: v for k, v in job.items() if k != "output_pdf"}
    return jsonify(safe)


@app.route("/api/download/<job_id>")
def api_download(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.get("state") != "done":
        return jsonify(error="Not ready"), 404
    pdf_path = job.get("output_pdf")
    if not pdf_path or not Path(pdf_path).exists():
        return jsonify(error="PDF not found"), 404
    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=Path(pdf_path).name,
        mimetype="application/pdf",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    port = int(os.environ.get("PORT", 5000))
    print(f"\n[Koba] Document scanner")
    print(f"   -> http://localhost:{port}\n")
    # use_reloader=False is important when background threads are in play
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
