"""Agent server entry point. load_dotenv must run before agent imports (auth config)."""

# ruff: noqa: E402
import os
from pathlib import Path

from dotenv import load_dotenv

# Load env vars from .env before any other imports (agent needs auth config)
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

import logging

from databricks_ai_bridge.long_running import LongRunningAgentServer
from mlflow.genai.agent_server import setup_mlflow_git_based_version_tracking

logger = logging.getLogger(__name__)

# Need to import the agent to register the functions with the server
import agent_server.agent  # noqa: F401

from agent_server.utils import replace_fake_id


class AgentServer(LongRunningAgentServer):
    def transform_stream_event(self, event, response_id):
        return replace_fake_id(event, response_id)


agent_server = AgentServer(
    "ResponsesAgent",
    enable_chat_proxy=True,
    task_timeout_seconds=float(os.getenv("TASK_TIMEOUT_SECONDS", "3600")),
    poll_interval_seconds=float(os.getenv("POLL_INTERVAL_SECONDS", "1.0")),
)

# Define the app as a module level variable to enable multiple workers
app = agent_server.app  # noqa: F841
try:
    setup_mlflow_git_based_version_tracking()
except Exception:
    logger.info("Git-based version tracking not available (expected in deployed apps).")


# Serve a polished chat UI at root so the Databricks "Open" button shows a demo.
import json as _json
import os as _os
from pathlib import Path as _Path

from fastapi.responses import HTMLResponse, JSONResponse


def _load_sample_questions() -> list[dict]:
    """Pick five canonical sample Q&A questions to expose as one-click chips in the UI."""
    fixtures_path = _os.getenv("FIXTURES_PATH")
    candidate_paths: list[_Path] = []
    if fixtures_path:
        sample_path = _Path(fixtures_path).parent / "sample_qa.json"
        candidate_paths.append(sample_path)
    candidate_paths.append(
        _Path(__file__).parent / "fixtures" / "sample_qa.json"
    )
    for p in candidate_paths:
        try:
            if p.exists():
                with open(p) as f:
                    raw = _json.load(f)
                wanted_ids = {"QA-001", "QA-006", "QA-007", "QA-014", "QA-012"}
                picked = [q for q in raw if q.get("qa_id") in wanted_ids]
                if picked:
                    return picked
        except Exception as exc:
            logger.info("Could not read sample Q&A from %s: %s", p, exc)
    return _BUILTIN_SAMPLE_QUESTIONS


# Fallback embedded samples — always shown if the volume is unreachable.
_BUILTIN_SAMPLE_QUESTIONS = [
    {
        "label": "PPE requirements",
        "qa_id": "QA-001",
        "question": "What PPE is required for workers entering a refinery process unit?",
        "category": "ppe",
    },
    {
        "label": "Oil spill response",
        "qa_id": "QA-007",
        "question": "What actions must be taken in the first 15 minutes of discovering an oil spill?",
        "category": "spill_response",
    },
    {
        "label": "Emergency contacts",
        "qa_id": "QA-006",
        "question": "What is the phone number for the PEMEX Emergency Operations Center?",
        "category": "emergency_contacts",
    },
    {
        "label": "LOTO procedure",
        "qa_id": "QA-014",
        "question": "What are the six steps of the LOTO (Lockout/Tagout) procedure?",
        "category": "loto",
    },
    {
        "label": "Contractor TRIR threshold",
        "qa_id": "QA-012",
        "question": "What TRIR threshold must contractors meet for pre-qualification?",
        "category": "contractor_prequalification",
    },
]


@app.get("/api/config")
async def api_config():
    """UI bootstraps from this — shows which prompt alias and LLM are running."""
    return JSONResponse(
        {
            "prompt_alias": _os.getenv("AGENT_PROMPT_VERSION", "v1"),
            "llm_endpoint": _os.getenv("LLM_ENDPOINT_NAME", "databricks-claude-sonnet-4-5"),
            "experiment": _os.getenv("MLFLOW_EXPERIMENT_NAME", ""),
            "samples": _load_sample_questions(),
        }
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    return _HOME_HTML


@app.get("/chat", response_class=HTMLResponse)
async def chat():
    return _CHAT_UI_HTML


_HOME_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>PEMEX KA Workshop</title>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<style>
  :root {
    --bg:#0b0d12; --panel:#11141b; --panel-2:#161a23; --line:#222837;
    --text:#e6e9ef; --muted:#8b94a7; --accent:#7c5cff; --accent-2:#22d3ee;
    --green:#34d399; --yellow:#fbbf24; --red:#f87171; --blue:#60a5fa;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  html,body { height:100%; background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif; font-size:14px; }

  /* ── Hero ── */
  .hero { text-align:center; padding:64px 24px 48px; }
  .hero .logo { display:inline-flex; align-items:center; gap:12px; margin-bottom:24px; }
  .hero .dot { width:14px; height:14px; border-radius:50%;
    background:linear-gradient(135deg,var(--accent),var(--accent-2));
    box-shadow:0 0 20px var(--accent); }
  .hero h1 { font-size:32px; font-weight:800; letter-spacing:-.5px;
    background:linear-gradient(135deg,var(--text),var(--muted));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .hero .sub { margin-top:10px; color:var(--muted); font-size:15px; max-width:560px; margin-inline:auto; line-height:1.6; }
  .hero .tag { display:inline-block; margin-top:14px; padding:4px 12px; border-radius:999px;
    background:rgba(124,92,255,.12); border:1px solid rgba(124,92,255,.3);
    color:var(--accent); font-size:12px; font-weight:600; letter-spacing:.4px; text-transform:uppercase; }

  /* ── Steps ── */
  .steps-section { max-width:900px; margin:0 auto; padding:0 24px 64px; }
  .steps-section h2 { font-size:13px; font-weight:600; text-transform:uppercase;
    letter-spacing:1.2px; color:var(--muted); margin-bottom:20px; }
  .steps { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; }

  .step { background:var(--panel); border:1px solid var(--line); border-radius:14px;
    padding:20px; display:flex; flex-direction:column; gap:10px; position:relative;
    transition:.2s; }
  .step:hover { border-color:var(--accent); transform:translateY(-2px); }
  .step .num { font-size:11px; font-weight:700; color:var(--muted); letter-spacing:.6px;
    text-transform:uppercase; }
  .step h3 { font-size:15px; font-weight:700; }
  .step p { color:var(--muted); font-size:13px; line-height:1.55; flex:1; }
  .step .badge { align-self:flex-start; padding:3px 9px; border-radius:999px;
    font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:.3px; }
  .badge.setup  { background:rgba(96,165,250,.1);  color:var(--blue); }
  .badge.trace  { background:rgba(34,211,238,.1);  color:var(--accent-2); }
  .badge.eval   { background:rgba(251,191,36,.1);  color:var(--yellow); }
  .badge.optim  { background:rgba(124,92,255,.1);  color:var(--accent); }
  .badge.compare{ background:rgba(52,211,153,.1);  color:var(--green); }
  .badge.deploy { background:rgba(248,113,113,.1); color:var(--red); }

  /* ── Concepts bar ── */
  .concepts { max-width:900px; margin:0 auto 48px; padding:0 24px; }
  .concepts h2 { font-size:13px; font-weight:600; text-transform:uppercase;
    letter-spacing:1.2px; color:var(--muted); margin-bottom:16px; }
  .concepts-grid { display:flex; flex-wrap:wrap; gap:10px; }
  .concept { background:var(--panel-2); border:1px solid var(--line); border-radius:8px;
    padding:10px 14px; display:flex; flex-direction:column; gap:3px; }
  .concept .label { font-size:12px; font-weight:700; color:var(--text); }
  .concept .desc  { font-size:11px; color:var(--muted); }

  /* ── CTA ── */
  .cta { text-align:center; padding:0 24px 72px; }
  .cta p { color:var(--muted); margin-bottom:20px; font-size:14px; }
  .btn { display:inline-flex; align-items:center; gap:8px; padding:14px 32px;
    background:linear-gradient(135deg,var(--accent),var(--accent-2));
    color:#fff; border:none; border-radius:12px; font-size:15px; font-weight:700;
    cursor:pointer; text-decoration:none; letter-spacing:.2px;
    box-shadow:0 4px 24px rgba(124,92,255,.3); transition:.2s; }
  .btn:hover { transform:translateY(-2px); box-shadow:0 8px 32px rgba(124,92,255,.45); }

  .divider { border:none; border-top:1px solid var(--line); max-width:900px;
    margin:0 auto 48px; }
</style>
</head>
<body>

<div class="hero">
  <div class="logo">
    <div class="dot"></div>
    <span style="font-weight:800;font-size:18px;letter-spacing:-.3px;">PEMEX Knowledge Assistant</span>
  </div>
  <h1>MLflow GenAI Workshop</h1>
  <p class="sub">Learn how to evaluate, trace, and systematically optimize a Databricks Knowledge Assistant using MLflow's GenAI evaluation suite.</p>
  <span class="tag">Prompt Optimization Lab</span>
</div>

<div class="steps-section">
  <h2>Lab Workflow</h2>
  <div class="steps">

    <div class="step">
      <div class="num">Step 01</div>
      <h3>Setup Data &amp; Knowledge Assistant</h3>
      <p>Generate synthetic PEMEX procedure documents, upload to UC Volume, and spin up the Knowledge Assistant via the Databricks API.</p>
      <span class="badge setup">Setup</span>
    </div>

    <div class="step">
      <div class="num">Step 02</div>
      <h3>Explore Auto-Generated Traces</h3>
      <p>The KA instruments every request automatically. Learn to query traces with <code style="color:var(--accent-2)">mlflow.search_traces()</code>, drill into spans, and snapshot to Delta.</p>
      <span class="badge trace">Tracing</span>
    </div>

    <div class="step">
      <div class="num">Step 03</div>
      <h3>Evaluate V1 (Baseline)</h3>
      <p>Run <code style="color:var(--accent-2)">mlflow.genai.evaluate()</code> against 30 Q&amp;A pairs with four scorers — answer quality, safety, groundedness, and completeness.</p>
      <span class="badge eval">Evaluation</span>
    </div>

    <div class="step">
      <div class="num">Step 04</div>
      <h3>Optimize with GEPA</h3>
      <p>Use <code style="color:var(--accent-2)">mlflow.genai.optimize_prompts()</code> to automatically generate improved KA instructions. Registered in the MLflow Prompt Registry.</p>
      <span class="badge optim">Optimization</span>
    </div>

    <div class="step">
      <div class="num">Step 05</div>
      <h3>Evaluate &amp; Compare</h3>
      <p>Evaluate the optimized instructions against the same dataset. Side-by-side comparison of V1 vs optimized across all scorer dimensions.</p>
      <span class="badge compare">Comparison</span>
    </div>

    <div class="step">
      <div class="num">Step 06</div>
      <h3>Apply &amp; Demo</h3>
      <p>Push the optimized instructions to the live KA endpoint. Try the updated assistant below — compare responses before and after optimization.</p>
      <span class="badge deploy">Deploy</span>
    </div>

  </div>
</div>

<hr class="divider" />

<div class="concepts">
  <h2>Key Concepts</h2>
  <div class="concepts-grid">
    <div class="concept">
      <div class="label">Auto-Tracing</div>
      <div class="desc">KA generates MLflow traces automatically — no instrumentation needed</div>
    </div>
    <div class="concept">
      <div class="label">Prompt Registry</div>
      <div class="desc">Version KA instructions with aliases (v1, optimized) in MLflow</div>
    </div>
    <div class="concept">
      <div class="label">GEPA</div>
      <div class="desc">Gradient-free prompt optimization using evaluation feedback</div>
    </div>
    <div class="concept">
      <div class="label">Custom Scorers</div>
      <div class="desc">answer_quality, Safety, Guidelines (groundedness &amp; completeness)</div>
    </div>
    <div class="concept">
      <div class="label">Delta Snapshots</div>
      <div class="desc">Trace data materialized to UC tables for SQL analytics</div>
    </div>
    <div class="concept">
      <div class="label">Databricks Apps</div>
      <div class="desc">One-click deploy — this UI is served from a DAB-managed app</div>
    </div>
  </div>
</div>

<div class="cta">
  <p>The KA is live — try it out</p>
  <a href="/chat" class="btn">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
    Open Demo Chat
  </a>
</div>

</body>
</html>"""


_CHAT_UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>PEMEX Knowledge Assistant</title>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<style>
  :root {
    --bg:#0b0d12; --panel:#11141b; --panel-2:#161a23; --line:#222837;
    --text:#e6e9ef; --muted:#8b94a7; --accent:#7c5cff; --accent-2:#22d3ee;
    --benign:#34d399; --suspicious:#fbbf24; --malicious:#f87171; --info:#60a5fa;
  }
  * { box-sizing: border-box; }
  html,body { margin:0; padding:0; height:100%; background:var(--bg); color:var(--text);
              font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif; font-size:14px; }
  a { color:var(--accent-2); text-decoration:none; }
  a:hover { text-decoration:underline; }
  code,pre { font-family:"JetBrains Mono","SF Mono",ui-monospace,monospace; }

  .layout { display:grid; grid-template-columns: 320px 1fr; height:100vh; }

  /* ── Sidebar ─────────────────────────────────────────── */
  .sidebar { background:var(--panel); border-right:1px solid var(--line); display:flex; flex-direction:column; }
  .brand { padding:18px 20px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:10px; }
  .brand .dot { width:10px; height:10px; border-radius:50%; background:linear-gradient(135deg,var(--accent),var(--accent-2)); box-shadow:0 0 12px var(--accent); }
  .brand .name { font-weight:700; letter-spacing:.2px; }
  .sidebar h3 { margin:18px 20px 8px; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:1px; color:var(--muted); }
  .samples { padding:0 12px 12px; overflow-y:auto; flex:1; }
  .sample {
    background:var(--panel-2); border:1px solid var(--line); border-radius:10px; padding:10px 12px;
    margin-bottom:8px; cursor:pointer; transition:.15s; display:flex; flex-direction:column; gap:4px;
  }
  .sample:hover { border-color:var(--accent); transform:translateY(-1px); }
  .sample .row { display:flex; justify-content:space-between; align-items:center; gap:8px; }
  .sample .lbl { font-weight:600; }
  .sample .id { color:var(--muted); font-size:11px; font-family:"JetBrains Mono",monospace; }
  .sample .title { color:var(--muted); font-size:12px; line-height:1.3; }

  .pill { display:inline-flex; align-items:center; gap:4px; padding:2px 8px; border-radius:999px;
          font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.4px; }
  .pill.benign { background:rgba(52,211,153,.12); color:var(--benign); }
  .pill.suspicious { background:rgba(251,191,36,.12); color:var(--suspicious); }
  .pill.malicious { background:rgba(248,113,113,.12); color:var(--malicious); }
  .pill.info { background:rgba(96,165,250,.12); color:var(--info); }
  .pill.muted { background:rgba(139,148,167,.12); color:var(--muted); }

  .footer { border-top:1px solid var(--line); padding:14px 20px; color:var(--muted); font-size:11px; line-height:1.6; }
  .footer .kv { display:flex; justify-content:space-between; gap:8px; }
  .footer code { color:var(--text); font-size:11px; }

  /* ── Main column ─────────────────────────────────────── */
  .main { display:flex; flex-direction:column; min-width:0; }
  .header { padding:14px 24px; border-bottom:1px solid var(--line); background:var(--panel);
            display:flex; align-items:center; justify-content:space-between; gap:12px; }
  .header h1 { margin:0; font-size:16px; font-weight:600; }
  .header .badges { display:flex; gap:8px; flex-wrap:wrap; }
  .badge { display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:6px;
           font-size:11px; background:var(--panel-2); color:var(--muted); border:1px solid var(--line); }
  .badge b { color:var(--text); font-weight:600; }
  .badge.optimized b { color:var(--accent-2); }
  .badge.v1 b { color:var(--suspicious); }

  .chat { flex:1; overflow-y:auto; padding:24px 24px 12px; display:flex; flex-direction:column; gap:18px; }
  .msg { max-width: 760px; }
  .msg.user { align-self:flex-end; }
  .msg .bubble {
    background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px 16px;
    line-height:1.5; white-space:pre-wrap; word-break:break-word;
  }
  .msg.user .bubble {
    background:linear-gradient(135deg, rgba(124,92,255,.18), rgba(34,211,238,.10));
    border-color:rgba(124,92,255,.35);
  }

  .verdict-card {
    background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:0; overflow:hidden;
  }
  .verdict-head { display:flex; align-items:center; gap:12px; padding:14px 18px; border-bottom:1px solid var(--line);
                  background:linear-gradient(180deg, rgba(255,255,255,.02), transparent); }
  .verdict-head .label { color:var(--muted); font-size:12px; }
  .verdict-head .v { font-size:18px; font-weight:700; text-transform:uppercase; letter-spacing:.5px; }
  .verdict-head .v.benign { color:var(--benign); }
  .verdict-head .v.suspicious { color:var(--suspicious); }
  .verdict-head .v.malicious { color:var(--malicious); }
  .verdict-head .conf { margin-left:auto; display:flex; align-items:center; gap:10px; min-width:200px; }
  .verdict-head .conf .bar { flex:1; height:6px; background:var(--line); border-radius:3px; overflow:hidden; }
  .verdict-head .conf .bar > i { display:block; height:100%; background:linear-gradient(90deg,var(--accent),var(--accent-2)); }
  .verdict-head .conf .num { font-variant-numeric:tabular-nums; color:var(--muted); font-size:12px; min-width:40px; text-align:right; }

  .verdict-body { padding:14px 18px; display:flex; flex-direction:column; gap:14px; }
  .vsec { display:flex; flex-direction:column; gap:6px; }
  .vsec .h { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.6px; font-weight:600; }
  .vsec .b { line-height:1.55; }
  .actions { list-style:none; padding:0; margin:0; display:flex; flex-direction:column; gap:6px; }
  .actions li { background:var(--panel-2); border:1px solid var(--line); border-left:3px solid var(--accent);
                padding:8px 12px; border-radius:6px; }

  .meta-row { display:flex; flex-wrap:wrap; gap:8px; padding:10px 18px; border-top:1px solid var(--line);
              background:rgba(255,255,255,.015); }
  .meta-row .chip { display:inline-flex; align-items:center; gap:6px; background:var(--panel-2); border:1px solid var(--line);
                    padding:4px 10px; border-radius:999px; font-size:11px; color:var(--muted); cursor:pointer; }
  .meta-row .chip:hover { color:var(--text); border-color:var(--accent); }
  .meta-row .chip code { color:var(--text); }

  .raw { margin-top:10px; }
  .raw summary { cursor:pointer; color:var(--muted); font-size:12px; padding:6px 0; }
  .raw pre { background:var(--panel-2); border:1px solid var(--line); border-radius:8px;
             padding:12px; overflow-x:auto; font-size:12px; line-height:1.5; max-height:280px; }

  /* ── Input ─────────────────────────────────────────────────── */
  .input-row { padding:14px 24px 18px; border-top:1px solid var(--line); background:var(--panel);
               display:flex; gap:10px; }
  textarea {
    flex:1; padding:12px 14px; min-height:54px; max-height:240px; resize:none;
    background:var(--panel-2); color:var(--text); border:1px solid var(--line); border-radius:10px;
    font-family:"JetBrains Mono",monospace; font-size:13px; line-height:1.45;
  }
  textarea:focus { outline:none; border-color:var(--accent); }
  button.send {
    padding:0 22px; background:linear-gradient(135deg,var(--accent),var(--accent-2));
    color:white; border:none; border-radius:10px; cursor:pointer; font-weight:600; letter-spacing:.3px;
  }
  button.send:disabled { opacity:.55; cursor:not-allowed; }

  .spinner { display:inline-block; width:14px; height:14px; border:2px solid rgba(255,255,255,.2);
             border-top-color:var(--accent-2); border-radius:50%; animation:spin .8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .toast {
    position:fixed; bottom:24px; left:50%; transform:translateX(-50%);
    background:var(--panel); border:1px solid var(--line); padding:10px 16px; border-radius:8px;
    color:var(--text); box-shadow:0 8px 24px rgba(0,0,0,.4); opacity:0; pointer-events:none; transition:.2s;
  }
  .toast.show { opacity:1; }
</style>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="brand">
      <div class="dot"></div>
      <div class="name">PEMEX Knowledge Assistant</div>
    </div>
    <h3>Sample questions</h3>
    <div id="samples" class="samples"></div>
    <div class="footer">
      <div class="kv"><span>Prompt</span> <code id="cfg-prompt">…</code></div>
      <div class="kv"><span>LLM</span> <code id="cfg-llm">…</code></div>
      <div class="kv" id="cfg-exp-row" style="display:none"><span>Experiment</span> <code id="cfg-exp"></code></div>
      <div style="margin-top:8px; font-size:10px; color:var(--muted);">PEMEX Knowledge Assistant · MLflow on Databricks</div>
    </div>
  </aside>

  <section class="main">
    <header class="header">
      <h1>PEMEX Procedure Q&amp;A</h1>
      <div class="badges">
        <span class="badge" id="b-prompt"><span>Prompt</span><b id="b-prompt-val">…</b></span>
        <span class="badge"><span>Latency</span><b id="b-lat">—</b></span>
        <span class="badge"><span>Trace</span><b id="b-trace">—</b></span>
      </div>
    </header>

    <div class="chat" id="chat"></div>

    <div class="input-row">
      <textarea id="input" rows="3" placeholder="Pick a sample question ←  or ask about any PEMEX procedure…"
                onkeydown="if(event.key==='Enter'&&(event.metaKey||event.ctrlKey)){event.preventDefault();send()}"></textarea>
      <button class="send" id="btn" onclick="send()">Ask <small style="opacity:.7">⌘↵</small></button>
    </div>
  </section>
</div>

<div class="toast" id="toast">Copied!</div>

<script>
const $ = (id) => document.getElementById(id);
const chat = $('chat'), input = $('input'), btn = $('btn');
let CFG = {};

function toast(msg){ const t=$('toast'); t.textContent=msg; t.classList.add('show'); setTimeout(()=>t.classList.remove('show'),1500); }
function copy(text){ navigator.clipboard.writeText(text).then(()=>toast('Copied: '+text.slice(0,40))); }

function tagFor(severity){
  const s=(severity||'').toLowerCase();
  if(s==='critical'||s==='high') return 'malicious';
  if(s==='medium') return 'suspicious';
  if(s==='low'||s==='info') return 'benign';
  return 'muted';
}

function renderSamples(samples){
  const root=$('samples'); root.innerHTML='';
  for(const s of samples){
    const div=document.createElement('div'); div.className='sample';
    div.innerHTML=`
      <div class="row">
        <span class="lbl">${s.label || s.question}</span>
        <span class="pill muted">${s.category || ''}</span>
      </div>
      <div class="row">
        <span class="title">${s.question || ''}</span>
        <span class="id">${s.qa_id || ''}</span>
      </div>`;
    div.onclick=()=>{
      input.value = s.question || s.label || '';
      input.focus();
    };
    root.appendChild(div);
  }
}

async function loadConfig(){
  try{
    const r=await fetch('/api/config'); CFG=await r.json();
    $('cfg-prompt').textContent = CFG.prompt_alias || '—';
    $('cfg-llm').textContent = CFG.llm_endpoint || '—';
    if(CFG.experiment){ $('cfg-exp-row').style.display='flex'; $('cfg-exp').textContent = CFG.experiment.split('/').pop(); }
    const bp=$('b-prompt-val'); bp.textContent = CFG.prompt_alias || '—';
    $('b-prompt').classList.add(CFG.prompt_alias === 'optimized' ? 'optimized' : 'v1');
    renderSamples(CFG.samples||[]);
  }catch(e){ console.error(e); }
}

function tryParseVerdict(text){
  // Pull the first JSON object that has a "verdict" field
  const m = text.match(/\{[\s\S]*?"verdict"[\s\S]*?\}/);
  if(!m) return null;
  try { return JSON.parse(m[0]); } catch { return null; }
}

function renderUserMsg(text){
  const wrap=document.createElement('div'); wrap.className='msg user';
  const b=document.createElement('div'); b.className='bubble';
  // Render as code block if it looks like JSON
  if(text.includes('{')) { const pre=document.createElement('pre'); pre.style.margin='0'; pre.textContent=text; b.appendChild(pre); }
  else { b.textContent=text; }
  wrap.appendChild(b); chat.appendChild(wrap); chat.scrollTop=chat.scrollHeight;
}

function renderLoading(){
  const phrases=['Searching PEMEX documentation…','Retrieving relevant procedures…','Checking compliance guidelines…','Reading emergency protocols…','Composing answer…'];
  const wrap=document.createElement('div'); wrap.className='msg assistant';
  const b=document.createElement('div'); b.className='bubble';
  b.innerHTML=`<span class="spinner"></span> <span class="loading-text">${phrases[0]}</span>`;
  wrap.appendChild(b); chat.appendChild(wrap); chat.scrollTop=chat.scrollHeight;
  let i=0;
  const t=setInterval(()=>{ i=(i+1)%phrases.length; const el=b.querySelector('.loading-text'); if(el) el.textContent=phrases[i]; }, 1400);
  return { node: wrap, stop: ()=>clearInterval(t) };
}

function renderVerdict(parsed, rawText, meta){
  const wrap=document.createElement('div'); wrap.className='msg assistant';
  const card=document.createElement('div'); card.className='verdict-card';

  if(parsed){
    const v=String(parsed.verdict||'').toLowerCase();
    const conf=Math.max(0,Math.min(1,Number(parsed.confidence)||0));
    card.innerHTML=`
      <div class="verdict-head">
        <div class="label">Verdict</div>
        <div class="v ${v}">${v||'unknown'}</div>
        <div class="conf">
          <span style="color:var(--muted); font-size:11px;">Confidence</span>
          <div class="bar"><i style="width:${(conf*100).toFixed(0)}%"></i></div>
          <div class="num">${(conf*100).toFixed(0)}%</div>
        </div>
      </div>
      <div class="verdict-body">
        ${parsed.evidence_summary ? `<div class="vsec"><div class="h">Evidence summary</div><div class="b">${escapeHtml(parsed.evidence_summary)}</div></div>` : ''}
        ${parsed.reasoning ? `<div class="vsec"><div class="h">Reasoning</div><div class="b">${escapeHtml(parsed.reasoning)}</div></div>` : ''}
        ${Array.isArray(parsed.recommended_actions) && parsed.recommended_actions.length ? `
          <div class="vsec"><div class="h">Recommended actions</div>
            <ul class="actions">${parsed.recommended_actions.map(a=>`<li>${escapeHtml(String(a))}</li>`).join('')}</ul>
          </div>` : ''}
      </div>`;
  } else {
    // Fallback — agent didn't return parseable JSON; just dump the text
    const body=document.createElement('div'); body.className='verdict-body';
    const pre=document.createElement('pre'); pre.style.margin='0'; pre.style.whiteSpace='pre-wrap'; pre.textContent=rawText;
    body.appendChild(pre); card.appendChild(body);
  }

  // Meta row: trace ID + latency + raw response toggle
  const meta_row=document.createElement('div'); meta_row.className='meta-row';
  if(meta.trace_id){
    const chip=document.createElement('span'); chip.className='chip';
    chip.innerHTML = `<span>trace_id</span> <code>${meta.trace_id.slice(0,16)}…</code>`;
    chip.title='Click to copy full trace ID';
    chip.onclick=()=>copy(meta.trace_id);
    meta_row.appendChild(chip);
  }
  if(meta.trace_url){
    const a=document.createElement('a'); a.className='chip'; a.href=meta.trace_url; a.target='_blank'; a.rel='noopener';
    a.innerHTML='<span>↗</span> Open in MLflow';
    meta_row.appendChild(a);
  }
  if(meta.latency_ms!=null){
    const chip=document.createElement('span'); chip.className='chip';
    chip.innerHTML=`<span>latency</span> <code>${(meta.latency_ms/1000).toFixed(1)}s</code>`;
    meta_row.appendChild(chip);
  }
  if(meta_row.children.length) card.appendChild(meta_row);

  // Raw response (collapsible)
  if(rawText){
    const det=document.createElement('details'); det.className='raw'; det.style.padding='0 18px 14px';
    det.innerHTML=`<summary>Raw agent response</summary><pre>${escapeHtml(rawText)}</pre>`;
    card.appendChild(det);
  }

  wrap.appendChild(card); chat.appendChild(wrap); chat.scrollTop=chat.scrollHeight;
}

function escapeHtml(s){
  return String(s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function send(){
  const text=input.value.trim(); if(!text) return;
  renderUserMsg(text); input.value=''; btn.disabled=true;
  const loader=renderLoading();
  const t0=performance.now();
  try{
    const r=await fetch('/invocations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({input:[{role:'user',content:text}]})});
    const data=await r.json();
    loader.stop(); loader.node.remove();
    if(!r.ok){
      const wrap=document.createElement('div'); wrap.className='msg assistant';
      wrap.innerHTML=`<div class="bubble" style="border-color:var(--malicious); color:var(--malicious);">Error: ${escapeHtml(data.detail||r.statusText)}</div>`;
      chat.appendChild(wrap); return;
    }
    const texts=[];
    for(const item of data.output||[]) if(item.type==='message') for(const c of item.content||[]) if(c.type==='output_text') texts.push(c.text);
    const rawText=texts.join('\n');
    const parsed=tryParseVerdict(rawText);
    const co=data.custom_outputs||{};
    const latency=performance.now()-t0;

    // Header badges
    if(co.trace_id){
      const tBadge=$('b-trace');
      tBadge.innerHTML = co.trace_url
        ? `<a href="${co.trace_url}" target="_blank" rel="noopener" style="color:var(--accent-2)">${co.trace_id.slice(0,12)}…</a>`
        : co.trace_id.slice(0,12)+'…';
    }
    $('b-lat').textContent = (latency/1000).toFixed(1)+'s';

    renderVerdict(parsed, rawText, { trace_id: co.trace_id, trace_url: co.trace_url, latency_ms: latency });
  }catch(e){
    loader.stop(); loader.node.remove();
    const wrap=document.createElement('div'); wrap.className='msg assistant';
    wrap.innerHTML=`<div class="bubble" style="border-color:var(--malicious); color:var(--malicious);">Error: ${escapeHtml(e.message)}</div>`;
    chat.appendChild(wrap);
  } finally {
    btn.disabled=false; input.focus();
  }
}

loadConfig();
</script>
</body></html>"""


def main():
    agent_server.run(app_import_string="agent_server.start_server:app")
