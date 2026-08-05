"""
Fire, People & AI-Generated Footage Detector — Streamlit frontend.

I redid the theme from the old dark "hacker terminal" look to something
brighter and friendlier — a warm "daylight" palette instead of neon-on-black.
Figured it reads better in front of an audience than a glitchy CRT theme.

While I was in here I also found and fixed a few real bugs:
  - the uploader only accepted .mp4/.mov/.avi, but the backend (config.py,
    and the README) also support .mkv — now this file just imports
    config.py directly instead of keeping its own separate copy of the
    allowed extensions, so the two can't drift apart again
  - every upload was sent to the backend labeled "video/mp4" no matter what
    was actually uploaded — now it uses the content type the browser/
    Streamlit already detected
  - the AI-generated threshold (0.7) was hardcoded here separately from
    config.py's AI_GENERATED_THRESHOLD — now it's imported instead of typed
    twice
  - the results panel used to live entirely inside `if st.button(...)`.
    That's fine until you add a second button (like the download button
    below) — clicking it triggers a Streamlit rerun, the analyze button's
    state resets to False, and the whole results panel would just vanish.
    Results now live in session_state so they survive reruns from other
    widgets on the page
  - added a client-side file-size check so an oversized file gets an
    instant message instead of uploading the whole thing just to be
    rejected by the backend afterward
  - the "SYSTEM ONLINE" badge in the old UI was static text — it said that
    regardless of whether the backend was actually reachable. It's now a
    real health check against GET /health
"""
import json
import os
from pathlib import Path

import requests
import streamlit as st

# config.py sits right next to this file and is already the single source of
# truth for thresholds/paths on the backend side — importing it here too
# means the uploader's accepted extensions and the AI-generated threshold
# shown in the UI can't quietly go out of sync with MAIN.py again.
try:
    import config
    ALLOWED_EXTENSIONS = sorted(ext.lstrip(".") for ext in config.ALLOWED_EXTENSIONS)
    MAX_UPLOAD_MB = config.MAX_UPLOAD_MB
    AI_GENERATED_THRESHOLD = config.AI_GENERATED_THRESHOLD
    FIRE_CONFIDENCE_THRESHOLD = config.FIRE_CONFIDENCE_THRESHOLD
except ImportError:
    # Fallback in case this ever gets launched from a folder that doesn't
    # have config.py next to it — keeps the app usable instead of crashing.
    ALLOWED_EXTENSIONS = ["avi", "mkv", "mov", "mp4"]
    MAX_UPLOAD_MB = 100
    AI_GENERATED_THRESHOLD = 0.7
    FIRE_CONFIDENCE_THRESHOLD = 0.5

# Backend URL can be overridden with an env var for deployment; still
# defaults to local dev so `streamlit run UI.py` works out of the box.
API_URL = os.environ.get("DETECTOR_API_URL", "http://localhost:8000/analyze")
HEALTH_URL = API_URL.replace("/analyze", "/health")

st.set_page_config(
    page_title="Fire, People & AI Detector",
    page_icon="🔥",
    layout="centered",
)

# Keeping the last completed analysis in session_state means it survives
# reruns triggered by *other* widgets on the page (see the download button
# note above) instead of only existing for the one script-run where the
# "Analyze" button happened to be True.
if "result" not in st.session_state:
    st.session_state.result = None
if "result_filename" not in st.session_state:
    st.session_state.result_filename = None

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --bg-1:#FFF9F2; --bg-2:#FFE9D8; --bg-3:#FFDCC4;
  --card:#FFFFFF;
  --ink:#2B2016; --ink-soft:#8B7A68; --ink-faint:#C9BBAC;
  --coral:#FF6B45; --coral-deep:#E0502B;
  --amber:#FFB020; --teal:#12B886;
  --line:#F1E3D4;
  --shadow: 0 10px 30px -12px rgba(224,80,43,0.18);
}

html, body, [class*="css"], .stApp{
  background: linear-gradient(160deg,var(--bg-1) 0%,var(--bg-2) 55%,var(--bg-3) 100%) !important;
  color: var(--ink) !important;
  font-family:'Plus Jakarta Sans', sans-serif !important;
}
#MainMenu, footer, header {visibility:hidden;}
.block-container {padding-top: 1rem !important; max-width: 720px !important;}

/* ── scrollbar ──
   Some OS/browser combos (macOS in particular) hide the scrollbar track
   entirely until you're mid-scroll, which on a page this tall can read as
   "the page doesn't scroll" even though it does — worth forcing it visible
   for anything shown on a projector/demo machine where that ambiguity
   isn't great.

   Streamlit doesn't actually scroll `html`/`body` — the real scrolling
   element is an inner wrapper div Streamlit renders around the whole app
   (data-testid stAppViewContainer / stMain in recent versions, a plain
   section.main in older ones). Forcing overflow-y:scroll on html/body
   alone reserves a track there, but that track has nothing to scroll —
   the actual overflowing content is scrolling inside the inner container,
   so you'd see a gutter that does nothing when you try to use it. Listing
   Streamlit's known container selectors alongside html/body targets
   whichever one is the real scrolling element across versions, while
   leaving html/body in as a harmless no-op fallback.

   `scroll` (not `auto`) reserves the track permanently so it doesn't pop
   in/out and shift layout width as content changes. The -webkit- rules
   theme Chrome/Edge/Safari's scrollbar to match the coral/amber palette
   instead of the default OS grey; Firefox does the same via
   `scrollbar-color` below since it doesn't support the ::-webkit-scrollbar
   pseudo-elements. */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main {
  overflow-y: scroll !important;
  scrollbar-width: thin; /* Firefox */
  scrollbar-color: var(--coral) var(--bg-2); /* Firefox: thumb track */
}
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: var(--bg-2); }
::-webkit-scrollbar-thumb {
  background: var(--coral);
  border-radius: 10px;
  border: 2px solid var(--bg-2);
}
::-webkit-scrollbar-thumb:hover { background: var(--coral-deep); }

/* ── ambient background blobs (pure CSS, no per-frame JS) ──
   The old version ran a cellular-automaton fire simulation plus 220 ember
   particles on a canvas, redrawn every animation frame, forever, even when
   nothing on screen needed to update. Looked cool but was needless CPU/
   battery drain for a form + a results panel. This gets a similar ambient
   motion out of three blurred, slowly-drifting gradient circles instead. */
.blob-field{position:fixed;inset:0;z-index:0;overflow:hidden;pointer-events:none;}
.blob{position:absolute;border-radius:50%;filter:blur(60px);opacity:.55;}
.blob-a{width:420px;height:420px;top:-120px;left:-100px;background:radial-gradient(circle,var(--coral),transparent 70%);animation:drift-a 22s ease-in-out infinite;}
.blob-b{width:360px;height:360px;top:8%;right:-140px;background:radial-gradient(circle,var(--amber),transparent 70%);animation:drift-b 26s ease-in-out infinite;}
.blob-c{width:300px;height:300px;bottom:-120px;left:28%;background:radial-gradient(circle,var(--teal),transparent 70%);opacity:.35;animation:drift-c 30s ease-in-out infinite;}
@keyframes drift-a{0%,100%{transform:translate(0,0)}50%{transform:translate(40px,60px)}}
@keyframes drift-b{0%,100%{transform:translate(0,0)}50%{transform:translate(-50px,40px)}}
@keyframes drift-c{0%,100%{transform:translate(0,0)}50%{transform:translate(30px,-40px)}}
@media (prefers-reduced-motion: reduce){.blob{animation:none !important;}}

.stApp > div{position:relative;z-index:1;}

/* ── hero ── */
.hero{text-align:center;padding:1.5rem 1rem 1.75rem;}
.logo-badge{width:88px;height:88px;margin:0 auto 1rem;border-radius:28px;
  background:linear-gradient(135deg,var(--coral),var(--amber));
  display:flex;align-items:center;justify-content:center;
  box-shadow:var(--shadow);position:relative;}
.logo-badge::after{content:'';position:absolute;inset:-8px;border-radius:34px;
  border:1.5px solid rgba(255,107,69,.25);animation:ring-pulse 2.4s ease-in-out infinite;}
@keyframes ring-pulse{0%,100%{transform:scale(1);opacity:.6}50%{transform:scale(1.08);opacity:.15}}
.logo-emoji{font-size:38px;animation:float-fire 3s ease-in-out infinite;}
@keyframes float-fire{0%,100%{transform:translateY(0)}50%{transform:translateY(-4px)}}

.brand-title{font-family:'Fraunces',serif;font-size:30px;font-weight:600;
  color:var(--ink);letter-spacing:-0.01em;margin-bottom:6px;}
.brand-sub{font-size:14px;color:var(--ink-soft);margin-bottom:1.1rem;max-width:420px;
  margin-left:auto;margin-right:auto;line-height:1.5;}

.status-pill{display:inline-flex;align-items:center;gap:8px;padding:.4rem 1rem;
  border-radius:100px;background:#FFFFFFB0;border:1px solid var(--line);
  font-size:12px;color:var(--ink-soft);}
.status-dot{width:7px;height:7px;border-radius:50%;background:var(--ink-faint);transition:background .3s;}
.status-dot.dot-ok{background:var(--teal);box-shadow:0 0 8px var(--teal);}
.status-dot.dot-warn{background:var(--amber);box-shadow:0 0 8px var(--amber);}
.status-dot.dot-bad{background:var(--coral-deep);box-shadow:0 0 8px var(--coral-deep);}

/* ── uploader ── */
[data-testid="stFileUploader"]{
  background:var(--card) !important;border:1.5px dashed #F0C9A8 !important;
  border-radius:20px !important;padding:1.6rem !important;
  box-shadow:var(--shadow) !important;transition:border-color .25s,transform .25s !important;
}
[data-testid="stFileUploader"]:hover{border-color:var(--coral) !important;transform:translateY(-2px);}
[data-testid="stFileUploader"] label{color:var(--ink-soft) !important;font-size:14px !important;}

/* ── primary button ── */
.stButton > button{
  width:100% !important;padding:.9rem !important;border:none !important;
  border-radius:100px !important;
  background:linear-gradient(135deg,var(--coral),var(--amber)) !important;
  color:#fff !important;font-family:'Plus Jakarta Sans',sans-serif !important;
  font-size:15px !important;font-weight:600 !important;letter-spacing:.01em !important;
  box-shadow:0 10px 24px -8px rgba(224,80,43,.45) !important;
  transition:transform .2s, box-shadow .2s !important;
}
.stButton > button:hover{transform:translateY(-2px);box-shadow:0 14px 28px -8px rgba(224,80,43,.55) !important;}

/* ── download button (secondary/quiet style) ── */
[data-testid="stDownloadButton"] > button{
  width:100% !important;border-radius:100px !important;
  background:var(--card) !important;color:var(--coral-deep) !important;
  border:1.5px solid #F0C9A8 !important;font-weight:600 !important;
  box-shadow:none !important;
}
[data-testid="stDownloadButton"] > button:hover{background:#FFF3EA !important;}

/* ── divider ── */
.sep{display:flex;align-items:center;gap:10px;margin:1.4rem 0;}
.sep-l,.sep-r{flex:1;height:1px;background:var(--line);}
.sep-dot{width:6px;height:6px;border-radius:50%;background:var(--coral);}

/* ── progress bar ── */
.stProgress > div > div > div{background:linear-gradient(90deg,var(--coral),var(--amber)) !important;}

/* ── verdict card ── */
.verd{border-radius:20px;padding:1.3rem 1.4rem;margin:1.2rem 0;box-shadow:var(--shadow);}
.verd.ok{background:#EAFBF3;border:1px solid #BEEBD7;}
.verd.warning{background:#FFF7E6;border:1px solid #FFE1A6;}
.verd.danger{background:#FFF0EC;border:1px solid #FFC8B8;}
.verd-row{display:flex;align-items:center;gap:14px;}
.verd-icon{width:52px;height:52px;border-radius:16px;flex-shrink:0;background:#FFFFFFAA;
  display:flex;align-items:center;justify-content:center;font-size:24px;}
.verd-tag{font-family:'Fraunces',serif;font-size:16px;font-weight:600;color:var(--ink);margin-bottom:3px;}
.verd-msg{font-size:13px;color:var(--ink-soft);line-height:1.5;}
.verd-pill{font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
  padding:5px 12px;border-radius:100px;background:#FFFFFFCC;color:var(--coral-deep);flex-shrink:0;}
.verd-subflag{font-size:11px;color:#9A6B00;background:#FFF1CC;border:1px solid #FFE1A0;
  border-radius:8px;padding:4px 10px;display:inline-block;margin-top:8px;}

/* ── metrics ── */
[data-testid="metric-container"]{
  background:var(--card) !important;border-radius:16px !important;padding:.9rem !important;
  border:1px solid var(--line) !important;box-shadow:0 6px 18px -10px rgba(43,32,22,.12) !important;
  transition:transform .2s !important;
}
[data-testid="metric-container"]:hover{transform:translateY(-3px);}
[data-testid="stMetricLabel"]{font-size:11px !important;color:var(--ink-soft) !important;letter-spacing:.02em !important;}
[data-testid="stMetricValue"]{font-family:'Fraunces',serif !important;font-size:20px !important;color:var(--ink) !important;font-weight:600 !important;}
[data-testid="stMetricDelta"]{font-size:11px !important;color:var(--coral-deep) !important;}

/* ── panels (chart / chips containers) ── */
.panel{background:var(--card);border:1px solid var(--line);border-radius:18px;
  padding:1.2rem 1.3rem;margin:1rem 0;box-shadow:0 6px 18px -10px rgba(43,32,22,.1);}
.p-title{font-family:'Fraunces',serif;font-size:13px;font-weight:600;color:var(--ink);margin-bottom:12px;}

.chips{display:flex;flex-wrap:wrap;gap:7px;}
.chip{font-family:'JetBrains Mono',monospace;font-size:11px;padding:5px 10px;
  border-radius:8px;background:#FFF3EA;color:var(--coral-deep);border:1px solid #FFE1CC;}

[data-testid="stVideo"]{border-radius:16px;overflow:hidden;border:1px solid var(--line);box-shadow:var(--shadow);}
.stAlert{border-radius:14px !important;}
</style>

<div class="blob-field">
  <div class="blob blob-a"></div>
  <div class="blob blob-b"></div>
  <div class="blob blob-c"></div>
</div>

<div class="hero">
  <div class="logo-badge"><span class="logo-emoji">🔥</span></div>
  <div class="brand-title">Fire, People &amp; AI Detector</div>
  <div class="brand-sub">Upload a clip and it'll check it for fire, whether anyone's nearby, and whether the footage looks AI-generated.</div>
  <div class="status-pill">
    <span class="status-dot" id="status-dot"></span>
    <span id="status-text">Checking backend…</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Real backend health check (replaces a hardcoded "SYSTEM ONLINE" badge that
# used to show regardless of whether the backend was actually up). The
# backend's CORS config already allows any origin, so this fetch from the
# browser works against a locally-running FastAPI server.
#
# Three states, not two: people_model_loaded alone can't tell a genuine
# load failure apart from a silent fallback to generic yolov8n.pt weights
# (see PEOPLE_DETECTOR.load_people_model / MAIN.py's /health docstring) —
# both report people_model_loaded: true. The extra
# people_model_using_fallback flag is what lets this distinguish "healthy"
# from "technically running, but not with the trained model," which
# matters a lot for anyone judging this project's actual detection
# accuracy rather than just whether the server responds.
_health_script = """
<script>
fetch("__HEALTH_URL__")
  .then(r => r.json())
  .then(d => {
    const dot = window.parent.document.getElementById("status-dot");
    const txt = window.parent.document.getElementById("status-text");
    if (!dot || !txt) return;
    if (!d.fire_model_loaded || !d.people_model_loaded) {
      txt.textContent = "Backend up, a model failed to load";
      dot.classList.add("dot-warn");
    } else if (d.people_model_using_fallback) {
      txt.textContent = "Online — using fallback person detector";
      dot.classList.add("dot-warn");
    } else {
      txt.textContent = "All models online";
      dot.classList.add("dot-ok");
    }
  })
  .catch(() => {
    const dot = window.parent.document.getElementById("status-dot");
    const txt = window.parent.document.getElementById("status-text");
    if (!dot || !txt) return;
    txt.textContent = "Backend unreachable";
    dot.classList.add("dot-bad");
  });
</script>
""".replace("__HEALTH_URL__", HEALTH_URL)
st.markdown(_health_script, unsafe_allow_html=True)


def call_api(uploaded_file):
    """
    🎓 In plain English: this is the "bridge" between what the user sees
    (this Streamlit page) and the actual AI work happening on the backend
    server (MAIN.py). It uploads the user's video to the FastAPI /analyze
    endpoint over HTTP, shows a progress bar while it waits, and then
    either hands back the JSON results dictionary or shows a friendly error
    message if something went wrong (wrong file type, server down, etc.).

    Sends the video to the backend and returns the parsed JSON result, or
    None if something went wrong (in which case an error is already shown
    to the user here, so the caller doesn't need to).
    """
    prog = st.progress(0, text="Uploading your clip…")
    try:
        prog.progress(15, text="Uploading your clip…")
        response = requests.post(
            API_URL,
            files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type or "application/octet-stream")},
            timeout=300,
        )
        prog.progress(85, text="Running the detection models…")

        if response.status_code == 200:
            prog.progress(100, text="Done!")
            prog.empty()
            return response.json()

        prog.empty()
        if response.status_code == 413:
            st.error(f"That file is larger than the {MAX_UPLOAD_MB} MB limit.")
        elif response.status_code == 503:
            st.error("A model isn't loaded on the server — check the backend logs.")
        else:
            # Catches everything else: 400 (bad extension / failed content
            # sniff), 422, or anything the two branches above don't cover.
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text or f"HTTP {response.status_code}"
            st.error(f"The server rejected the request: {detail}")
        return None

    except requests.exceptions.ConnectionError:
        prog.empty()
        st.error("Can't reach the backend — make sure it's running: `py -3.12 -m uvicorn MAIN:app --reload`")
        return None
    except requests.exceptions.Timeout:
        prog.empty()
        st.error("The upload timed out — check your connection or try a smaller file.")
        return None
    except requests.exceptions.RequestException as e:
        # Catch-all for anything requests can raise that isn't a connection
        # error or a timeout (a dropped connection mid-upload, for example).
        # Previously there was no handler for this case, so it would surface
        # as Streamlit's raw traceback instead of a normal error message.
        prog.empty()
        st.error(f"Something went wrong talking to the backend: {e}")
        return None


def render_dashboard(r: dict, filename: str):
    """
    🎓 In plain English: this function takes the raw JSON dictionary that
    came back from the backend (fire_detected, people_near_fire, ai score,
    etc.) and turns it into the actual visual results the user sees:
    the big colored verdict card at the top, the row of stat "metric"
    boxes, the fire-confidence line chart, and the list of flagged frame
    numbers. It's purely presentation — all the real detection work already
    happened on the backend before this function is ever called.

    Renders the verdict card, metrics, chart, and frame chips for one result.
    """
    ai_available = r.get("ai_check_available", True)
    ai_prob = r.get("ai_generated_probability")
    # Present once the backend flags a clip as AI-generated: fire/people
    # detection isn't just "not found" in that case, it was never run at
    # all (see MAIN.py's run_analysis — running expensive fire/people
    # models against footage already known to be synthetic would only
    # produce findings that need to be discarded). These flags drive the
    # metrics and panels below so the UI doesn't imply "no fire" when the
    # honest answer is "not checked."
    fire_skipped = r.get("fire_check_skipped", False)
    people_skipped = r.get("people_check_skipped", False)

    # Verdict styling is driven off the structured fields the backend
    # returns, not by substring-matching r["verdict"] — a check like
    # "fire detected" in v would also match "no fire detected".
    if ai_available and ai_prob is not None and ai_prob > AI_GENERATED_THRESHOLD:
        icon, cls, tag, pill = "🤖", "danger", "Likely AI-generated", "FLAGGED"
    elif r["fire_detected"] and r["people_near_fire"]:
        icon, cls, tag, pill = "🚨", "danger", "Fire detected near people", "ALERT"
    elif r["fire_detected"]:
        icon, cls, tag, pill = "🔥", "warning", "Fire detected", "WARNING"
    else:
        icon, cls, tag, pill = "✅", "ok", "All clear", "CLEAR"

    ai_flag_html = ""
    if not ai_available:
        ai_flag_html = '<div class="verd-subflag">⚠ AI-generated check unavailable — model not loaded</div>'
    elif fire_skipped or people_skipped:
        ai_flag_html = '<div class="verd-subflag">🚫 Fire/people detection skipped — footage flagged as AI-generated</div>'

    # Kept as one line on purpose: st.markdown(unsafe_allow_html=True) parses
    # raw HTML until it hits a blank line, then falls back to normal
    # markdown. When ai_flag_html was "", its placeholder line went blank
    # and silently cut the HTML parse short, dumping the rest of the tags as
    # literal text in a code box.
    verdict_html = (
        f'<div class="verd {cls}"><div class="verd-row">'
        f'<div class="verd-icon">{icon}</div>'
        f'<div style="flex:1"><div class="verd-tag">{tag}</div>'
        f'<div class="verd-msg">{r["verdict"]}</div>{ai_flag_html}</div>'
        f'<div class="verd-pill">{pill}</div>'
        f'</div></div>'
    )
    st.markdown(verdict_html, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if fire_skipped:
            st.metric("🔥 Fire", "Skipped", "AI-generated")
        else:
            st.metric("🔥 Fire", "Yes" if r["fire_detected"] else "No", f"{r['max_fire_confidence']*100:.1f}% conf")
    with c2:
        if people_skipped:
            st.metric("🧍 People", "Skipped", "AI-generated")
        else:
            st.metric("🧍 People", "Yes" if r["people_detected"] else "No", f"Peak {r['peak_people_count']}")
    with c3:
        if people_skipped:
            st.metric("📏 Proximity", "—", "Skipped")
        else:
            prox = r.get("closest_person_to_fire_px")
            st.metric("📏 Proximity", f"{prox:.0f}px" if prox is not None else "—",
                       "Near fire" if r["people_near_fire"] else "n/a")
    with c4:
        ai_display = f"{ai_prob*100:.1f}%" if ai_prob is not None else "N/A"
        st.metric("🎭 AI score", ai_display, None if ai_available else "unavailable")
    with c5:
        st.metric("🎞 Frames", r["total_frames_analyzed"], f"{r.get('processing_time_seconds', '—')}s")

    if fire_skipped:
        # Replaces the sparkline chart + frame chips below — there's no
        # per-frame fire data to show when the fire model never ran, and an
        # empty chart/chip panel (or worse, no panel at all with no
        # explanation) would read as a bug rather than a deliberate skip.
        st.markdown("""
        <div class="panel">
          <div class="p-title">Fire / people detection</div>
          <div style="font-size:13px;color:var(--ink-soft);">
            Skipped — this clip was flagged as AI-generated, so fire and
            people checks weren't run against it.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        # ── fire confidence timeline (SVG sparkline) ──
        timeline = r.get("fire_confidence_timeline", [])
        if timeline:
            w, h, pad = 640, 120, 10
            n = len(timeline)
            step = (w - 2 * pad) / max(1, n - 1)
            pts = " ".join(
                f"{pad + i*step:.1f},{h - pad - v*(h - 2*pad):.1f}"
                for i, v in enumerate(timeline)
            )
            area_pts = f"{pad},{h-pad} " + pts + f" {w-pad},{h-pad}"
            thresh_y = h - pad - FIRE_CONFIDENCE_THRESHOLD * (h - 2 * pad)
            flagged_dots = "".join(
                f'<circle cx="{pad + i*step:.1f}" cy="{h - pad - timeline[i]*(h - 2*pad):.1f}" r="3.5" fill="var(--amber)" stroke="#fff" stroke-width="1" />'
                for i in r["fire_flagged_frames"]
            )
            st.markdown(f"""
            <div class="panel">
              <div class="p-title">Fire confidence — per frame</div>
              <svg viewBox="0 0 {w} {h}" style="width:100%;height:auto;display:block;">
                <line x1="{pad}" y1="{thresh_y:.1f}" x2="{w-pad}" y2="{thresh_y:.1f}"
                      stroke="#E0502B55" stroke-width="1" stroke-dasharray="4,4" />
                <polygon points="{area_pts}" fill="url(#fireGrad)" opacity="0.30" />
                <polyline points="{pts}" fill="none" stroke="#FF6B45" stroke-width="2.5" stroke-linejoin="round" />{flagged_dots}
                <defs>
                  <linearGradient id="fireGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#FF6B45"/>
                    <stop offset="100%" stop-color="#FF6B45" stop-opacity="0"/>
                  </linearGradient>
                </defs>
              </svg>
            </div>""", unsafe_allow_html=True)

        # Frame chips now actually populate — the backend used to hardcode
        # "fire_flagged_frames" to an empty list in its response even though it
        # had already computed the real list internally, so this section (and
        # the flagged dots on the chart above) silently never showed anything.
        # Fixed in MAIN.py alongside this.
        if r["fire_flagged_frames"]:
            chips = "".join(f'<span class="chip">#{str(i).zfill(2)}</span>' for i in r["fire_flagged_frames"])
            st.markdown(f"""
            <div class="panel">
              <div class="p-title">Fire flagged frames — {len(r["fire_flagged_frames"])} / {r["total_frames_analyzed"]}</div>
              <div class="chips">{chips}</div>
            </div>""", unsafe_allow_html=True)

    with st.expander("See the full raw response"):
        st.json(r)

    st.download_button(
        "⬇ Download report (JSON)",
        data=json.dumps(r, indent=2),
        file_name=f"{Path(filename).stem}_report.json",
        mime="application/json",
    )


# ── FILE UPLOADER ─────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    f"Upload a video ({' · '.join(e.upper() for e in ALLOWED_EXTENSIONS)}) — max {MAX_UPLOAD_MB} MB",
    type=ALLOWED_EXTENSIONS,
)

if uploaded:
    st.video(uploaded)

    size_mb = uploaded.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        # Checking this on the client means an oversized file gets an
        # instant answer instead of uploading the whole thing first only to
        # be rejected by the backend's own size check afterward.
        st.error(f"This file is {size_mb:.0f} MB — the limit is {MAX_UPLOAD_MB} MB.")
    else:
        st.markdown(
            '<div class="sep"><div class="sep-l"></div><div class="sep-dot"></div><div class="sep-r"></div></div>',
            unsafe_allow_html=True,
        )

        if st.button("🔍  Analyze this video"):
            result = call_api(uploaded)
            if result is not None:
                st.session_state.result = result
                st.session_state.result_filename = uploaded.name

        # Only render a result if it belongs to the file that's currently
        # uploaded — if someone swaps in a different video, the old
        # dashboard won't keep showing under the new preview until they
        # actually re-run the scan.
        if st.session_state.result is not None and st.session_state.result_filename == uploaded.name:
            render_dashboard(st.session_state.result, uploaded.name)
