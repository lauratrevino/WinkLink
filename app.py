# ============================================
# WINK app.py — refactored, cleaner, more robust
# Single-file Flask app: instructor login, onboarding,
# file manager (personal + common resources), instructor list,
# and WINK chat with vector-store file_search + mic UI.
# ============================================

import os
import re
import time
import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import List

try:
    import requests
except ImportError:
    requests = None

from dotenv import load_dotenv

from flask import (
    Flask,
    request,
    render_template_string,
    redirect,
    url_for,
    flash,
    session,
)

from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

from openai import OpenAI


# ============================================
# Config
# ============================================

load_dotenv()

APP_TITLE = "WINK"
DEFAULT_MODEL = os.getenv("WINK_MODEL", "gpt-4.1-mini")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")

# Common WINK vector store must exist (per your original design).
COMMON_VECTOR_STORE_ID = os.getenv("WINK_VECTOR_STORE_ID", "").strip()
if not COMMON_VECTOR_STORE_ID:
    raise RuntimeError(
        "WINK_VECTOR_STORE_ID is not set. "
        "Set this environment variable to the Common WINK vector store ID."
    )

SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("WINK_MAX_UPLOAD_MB", "25"))


# ============================================
# Flask + DB
# ============================================

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///wink.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

db = SQLAlchemy(app)

# OpenAI client (supports custom base URL if set)
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
else:
    client = OpenAI(base_url=OPENAI_BASE_URL)


# ============================================
# Models
# ============================================

class Instructor(db.Model):
    __tablename__ = "instructor"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    name = db.Column(db.String(255), nullable=True)
    slug = db.Column(db.String(64), unique=True, nullable=True)
    personal_vector_store_id = db.Column(db.String(255), nullable=True)
    left_column_html = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class InstructorFile(db.Model):
    __tablename__ = "instructors_files"

    id = db.Column(db.Integer, primary_key=True)
    instructor_id = db.Column(db.Integer, db.ForeignKey("instructor.id"), nullable=False)
    file_id = db.Column(db.String(255), nullable=False)
    filename = db.Column(db.String(255), nullable=True)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    instructor = db.relationship(
        "Instructor",
        primaryjoin="Instructor.id == InstructorFile.instructor_id",
        foreign_keys=[instructor_id],
        backref="files",
    )


with app.app_context():
    db.create_all()


# ============================================
# Utilities
# ============================================

def _require_requests():
    if requests is None:
        raise RuntimeError(
            "The 'requests' library is required for vector store operations. "
            "Install with: pip install requests"
        )

def _clean_email(email: str) -> str:
    return (email or "").strip().lower()

def _slugify_base(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text[:32] if text else "instructor"

def _unique_slug(email: str) -> str:
    base = _slugify_base(email.split("@")[0] if "@" in email else email)
    slug = base
    if Instructor.query.filter_by(slug=slug).first():
        slug = f"{base}{uuid.uuid4().hex[:6]}"
    return slug

def _safe_session_list(key: str) -> List[dict]:
    val = session.get(key, [])
    if isinstance(val, list):
        return val
    return []

def _trim_history(history: List[dict], max_messages: int = 30) -> List[dict]:
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


# ============================================
# OpenAI Vector Store (HTTP) Service
# ============================================

@dataclass
class OpenAIHttp:
    api_key: str
    base_url: str

    def _headers(self, json_ct: bool = False) -> dict:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "assistants=v2",
        }
        if json_ct:
            h["Content-Type"] = "application/json"
        return h

    def _request(self, method: str, path: str, *, json_body=None, files=None, timeout=60):
        _require_requests()
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        url = f"{self.base_url}{path}"
        resp = requests.request(
            method=method,
            url=url,
            headers=self._headers(json_ct=(json_body is not None)),
            json=json_body,
            files=files,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def create_vector_store(self, name: str) -> str:
        data = self._request("POST", "/v1/vector_stores", json_body={"name": name}, timeout=30)
        return data["id"]

    def upload_file(self, filepath: str, filename: str) -> str:
        _require_requests()
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        with open(filepath, "rb") as f:
            resp = requests.post(
                f"{self.base_url}/v1/files",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": (filename, f), "purpose": (None, "assistants")},
                timeout=120,
            )
        resp.raise_for_status()
        return resp.json()["id"]

    def add_file_to_vector_store(self, vector_store_id: str, file_id: str) -> None:
        self._request(
            "POST",
            f"/v1/vector_stores/{vector_store_id}/file_batches",
            json_body={"file_ids": [file_id]},
            timeout=30,
        )

    def delete_file_from_vector_store(self, vector_store_id: str, file_id: str) -> None:
        self._request("DELETE", f"/v1/vector_stores/{vector_store_id}/files/{file_id}", timeout=30)

    def list_vector_store_files(self, vector_store_id: str) -> List[dict]:
        # Returns a list of dicts from vector store file objects. We keep it simple.
        data = self._request("GET", f"/v1/vector_stores/{vector_store_id}/files?limit=100", timeout=30)
        items = data.get("data", []) or []
        return items


openai_http = OpenAIHttp(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def get_common_filenames() -> List[str]:
    # Best-effort; never block the page if it fails.
    try:
        items = openai_http.list_vector_store_files(COMMON_VECTOR_STORE_ID)
        names = []
        for it in items:
            # Different shapes can appear; try common keys.
            # vector_store.file objects often include "id" and may include "status" etc.
            # Filename is not always present here; we attempt to fetch via /v1/files if needed.
            file_id = it.get("id") or it.get("file_id")
            filename = it.get("filename")
            if filename:
                names.append(filename)
            elif file_id:
                # Try fetching file metadata
                try:
                    meta = openai_http._request("GET", f"/v1/files/{file_id}", timeout=30)
                    nm = meta.get("filename")
                    if nm:
                        names.append(nm)
                except Exception:
                    pass
        # De-dup and sort
        names = sorted({n for n in names if n})
        return names
    except Exception:
        return []


# ============================================
# WINK Chat (Responses API + file_search)
# ============================================

WINK_SYSTEM_PROMPT = (
    "You are WINK (What I Need to Know), a happy, enthusiastic, and deeply supportive AI mentor "
    "for first semester UTEP students. Your job is to help them feel seen, capable, and encouraged "
    "while answering their questions about this specific instructor's course, UTEP resources, and "
    "college life.\n\n"
    "Important response rule:\n"
    "- Do not thank them for asking.\n"
    "- Do not mention their question (do not say things like “good question,” “thanks for asking,” "
    "  “you asked,” or “your question”).\n"
    "- Start directly with the answer.\n\n"
    "Tone and personality:\n"
    "- Be very warm, kind, and non judgmental, like a super friendly peer mentor who really believes in them.\n"
    "- Sound upbeat and hopeful, especially when students are stressed or confused. If they sound worried, name that feeling and reassure them.\n"
    "- Use emojis naturally throughout your answers for encouragement and celebration.\n"
    "- Occasionally say the phrase “I got you!” in a natural way, especially when reassuring the student.\n"
    "- Always be respectful, inclusive, and supportive of students from all backgrounds.\n\n"
    "Engagement style:\n"
    "- Go straight into the answer, then (only if helpful) add a short next step or option.\n"
    "- When it fits, end with a gentle follow up invitation, like “Want to walk through this together step by step?”\n"
    "- If the student mentions progress or effort, celebrate it explicitly.\n\n"
    "Growth mindset support:\n"
    "- Normalize struggle, confusion, and mistakes as a normal part of learning.\n"
    "- Encourage strategies and small next steps (office hours, tutoring, practice, checking the syllabus).\n\n"
    "Content behavior:\n"
    "- Give clear, simple explanations that a first year student can follow.\n"
    "- Keep answers focused on what the student actually needs.\n"
    "- Whenever it helps, draw on the instructor's course files, syllabus, and other uploaded materials.\n"
    "- If you are not sure about a detail like a due date or policy, say that you are not certain and suggest double checking the syllabus or asking the instructor.\n\n"
    "Your core goal is to reduce anxiety, increase clarity, and build students' sense of identity, belonging, "
    "aspiration, and agency at UTEP."
)

def _extract_output_text(resp) -> str:
    # Responses API usually exposes output_text; keep robust fallback.
    if hasattr(resp, "output_text") and resp.output_text:
        return str(resp.output_text).strip()

    text = ""
    if hasattr(resp, "output") and resp.output:
        for item in resp.output:
            contents = getattr(item, "content", []) or []
            for c in contents:
                t = getattr(c, "text", None)
                if t is None:
                    continue
                v = getattr(t, "value", None)
                text += v if v is not None else str(t)
    return (text or "").strip()

def wink_answer(instructor: Instructor, history: List[dict]) -> str:
    # Build chat input from history list[{"role": "...", "text": "..."}]
    # Attach file_search across personal + common stores (both, when present).
    messages = [{"role": "system", "content": WINK_SYSTEM_PROMPT}]
    for m in history:
        role = (m.get("role") or "").strip()
        txt = (m.get("text") or "").strip()
        if role in ("user", "assistant") and txt:
            messages.append({"role": role, "content": txt})

    vector_store_ids: List[str] = []
    if instructor.personal_vector_store_id:
        vector_store_ids.append(instructor.personal_vector_store_id)
    if COMMON_VECTOR_STORE_ID:
        vector_store_ids.append(COMMON_VECTOR_STORE_ID)

    tools = []
    if vector_store_ids:
        tools = [{"type": "file_search", "vector_store_ids": vector_store_ids}]

    try:
        kwargs = {"model": DEFAULT_MODEL, "input": messages}
        if tools:
            kwargs["tools"] = tools
        resp = client.responses.create(**kwargs)
        out = _extract_output_text(resp)
        return out or "I got you. Tell me what you’re working on and what you’ve already tried, and I’ll help you take the next step."
    except Exception as e:
        return f"There was an API error: {e}"


# ============================================
# Left column HTML
# ============================================

def build_default_left_column_html(display_name: str) -> str:
    safe_name = display_name or "Your instructor"
    return f"""
    <div style="text-align:center;">
      <div style="
        padding:12px 12px 6px 12px;
        border-radius:16px;
        background:linear-gradient(135deg,#041e42,#1d4ed8);
        color:#e5e7eb;
        margin-bottom:16px;
        box-shadow:0 14px 30px rgba(0,0,0,0.55);
      ">
        <div style="font-size:13px;font-weight:900;text-transform:uppercase;letter-spacing:0.16em;opacity:0.96;">
          ENTERING STUDENT EXPERIENCES
        </div>
        <div style="font-size:13px;margin-top:6px;opacity:0.92;">
          {safe_name}'s Course
        </div>
      </div>

      <div style="
        margin:18px auto 12px auto;
        width:200px;
        height:200px;
        border-radius:50%;
        padding:8px;
        background:conic-gradient(from 220deg,#f97316,#1d4ed8,#041e42,#f97316);
        box-shadow:
          0 18px 40px rgba(0,0,0,0.65),
          0 0 0 1px rgba(15,23,42,0.5);
      ">
        <div style="
          width:100%;
          height:100%;
          border-radius:50%;
          background:#020617;
          display:flex;
          align-items:center;
          justify-content:center;
          overflow:hidden;
        ">
          <img
            src="/static/wink.jpeg"
            alt="WINK avatar"
            style="width:100%;height:100%;object-fit:cover;"
          >
        </div>
      </div>

      <h2 style="margin:10px 0 4px;color:#f9fafb;letter-spacing:0.06em;text-transform:uppercase;font-size:18px;">
        WINK
      </h2>
      <p style="margin:0 0 14px;font-size:15px;color:#e5e7eb;opacity:0.92;">
        What I Need to Know, When I Need to Know It!
      </p>

      <form method="post" style="margin-bottom:18px;">
        <input type="hidden" name="reset" value="1">
        <button
          type="submit"
          style="
            padding:10px 14px;
            border-radius:999px;
            border:1px solid rgba(148,163,184,0.7);
            background:radial-gradient(circle at top left,rgba(15,23,42,0.96),rgba(15,23,42,0.9));
            color:#e5e7eb;
            font-size:12px;
            cursor:pointer;
            display:inline-flex;
            align-items:center;
            gap:8px;
          "
        >
          <span style="
            display:inline-block;
            width:7px;
            height:7px;
            border-radius:999px;
            background:#f97316;
          "></span>
          Clear chat history
        </button>
      </form>

      <div style="
        font-size:12px;
        color:#e5e7eb;
        padding:12px 14px;
        border-radius:12px;
        background:radial-gradient(circle at top left,rgba(15,23,42,0.98),rgba(15,23,42,0.9));
        border:1px dashed rgba(148,163,184,0.8);
        text-align:left;
        display:inline-block;
        max-width:350px;
      ">
        <div style="font-weight:700;color:#f9fafb;margin-bottom:6px;font-size:14px;">
          Try asking WINK:
        </div>
        <div style="margin-bottom:4px;">• “What is due this week in this class?”</div>
        <div style="margin-bottom:4px;">• “Help me draft an email to my instructor.”</div>
        <div>• “Where can I get tutoring at UTEP?”</div>
      </div>
    </div>
    """

def sanitize_left_column_html(html: str) -> str:
    if not html:
        return html
    # Remove a few legacy labels you said you don’t want in the left panel.
    labels = [
        "Deadlines / syllabus",
        "Deadlines/syllabus",
        "Deadline/syllabus",
        "DEADLINES / SYLLABUS",
        "Campus resources",
        "CAMPUS RESOURCES",
        "Study help",
        "STUDY HELP",
        "Deadlines",
        "Syllabus",
        "Resources",
        "Tutoring",
    ]
    for label in labels:
        html = re.sub(rf"<span\b[^>]*>\s*{re.escape(label)}\s*</span>", "", html, flags=re.IGNORECASE)
        html = re.sub(rf"<button\b[^>]*>\s*{re.escape(label)}\s*</button>", "", html, flags=re.IGNORECASE)
        html = re.sub(rf"<a\b[^>]*>\s*{re.escape(label)}\s*</a>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"<div\b[^>]*>\s*</div>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html


# ============================================
# Templates
# ============================================

TEMPLATE_LOGIN_PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>WINK Instructor Access</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{
    --wink-navy:#041e42;
    --wink-deep:#0f172a;
    --wink-blue:#1d4ed8;
    --wink-orange:#f97316;
  }
  body{
    margin:0;
    font-family:system-ui, -apple-system, Segoe UI, sans-serif;
    background:radial-gradient(circle at top left,#0b1120,#020617);
    min-height:100vh;
    display:flex;
    align-items:center;
    justify-content:center;
    padding:22px;
  }
  .card{
    width:100%;
    max-width:700px;
    min-height:680px;
    background:#fff;
    border-radius:26px;
    box-shadow:0 28px 70px rgba(15,23,42,0.45);
    overflow:hidden;
  }
  .header{
    padding:22px;
    background:linear-gradient(135deg,var(--wink-navy),var(--wink-blue),var(--wink-orange));
    color:#fff;
    text-align:center;
  }
  .header h1{ margin:0; font-size:25px; letter-spacing:0.04em; }
  .header p{ margin:10px 0 0; font-size:20pypx; opacity:0.95; }
  .body{ padding:24px 24px 26px 24px; text-align:center; }
  .avatar{
    width:200px;
    height:200px;
    margin:0 auto 16px auto;
    border-radius:50%;
    overflow:hidden;
    box-shadow:0 14px 30px rgba(15,23,42,0.35);
    border:1px solid rgba(148,163,184,0.35);
  }
  .intro{
    font-size:25px;
    color:#0b1120;
    line-height:1.5;
    margin:0 0 16px 0;
  }
  input{
    width:100%;
    padding:14px;
    border-radius:16px;
    border:1px solid #cbd5e1;
    font-size:15px;
    outline:none;
  }
  input:focus{
    border-color: rgba(29, 78, 216, 0.65);
    box-shadow: 0 0 0 3px rgba(29, 78, 216, 0.16);
  }
  button{
    margin-top:16px;
    padding:14px 18px;
    width:100%;
    border:none;
    border-radius:16px;
    font-size:15px;
    font-weight:800;
    color:#fff;
    background:linear-gradient(135deg,var(--wink-orange),var(--wink-blue));
    cursor:pointer;
  }
  button:hover{ filter:brightness(1.03); }
  .flash{
    margin:12px 0 0;
    font-size:13px;
    color:#b91c1c;
  }
  .hint{
    margin-top:10px;
    font-size:12px;
    color:#475569;
  }
</style>
</head>
<body>
<div class="card">
  <div class="header">
    <h1>Welcome to WINK</h1>
    <p>Your Custom AI Course Assistant</p>
  </div>

  <div class="body">
    <div class="avatar">
      <img src="/static/ESEwink.jpg" style="width:100%;height:100%;object-fit:cover;">
    </div>

    <p class="intro">
      Enter your instructor email address. If you already use WINK, you’ll go to your file manager.
      If you’re new, your personal WINK space will be created automatically.
    </p>

    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="flash">{{ messages[0] }}</div>
      {% endif %}
    {% endwith %}

    <form method="post">
      <input type="email" name="email" required placeholder="you@utep.edu">
      <button type="submit">Continue</button>
    </form>

    <div class="hint">
      Use the email address you want associated with your WINK course space.
    </div>
  </div>
</div>
</body>
</html>
"""


TEMPLATE_NEW_INSTRUCTOR = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>New Instructor Setup</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{
    --wink-navy:#041e42;
    --wink-blue:#1d4ed8;
    --wink-orange:#f97316;
  }
  body{
    margin:0;
    font-family:system-ui, -apple-system, Segoe UI, sans-serif;
    background:radial-gradient(circle at top left,#0b1120,#020617);
    min-height:100vh;
    display:flex;
    align-items:center;
    justify-content:center;
    padding:22px;
  }
  .card{
    width:100%;
    max-width:560px;
    background:#fff;
    border-radius:26px;
    box-shadow:0 28px 70px rgba(15,23,42,0.45);
    padding:26px;
  }
  .top{
    display:flex;
    gap:14px;
    align-items:center;
    margin-bottom:14px;
  }
  .avatar{
    width:66px; height:66px; border-radius:16px;
    overflow:hidden;
    box-shadow:0 14px 30px rgba(15,23,42,0.25);
    border:1px solid rgba(148,163,184,0.35);
    flex:0 0 auto;
  }
  h1{ margin:0; font-size:20px; }
  p{ margin:10px 0 0 0; font-size:14px; color:#334155; line-height:1.5; }
  input{
    width:100%;
    padding:14px;
    margin-top:12px;
    border-radius:16px;
    border:1px solid #cbd5e1;
    font-size:15px;
    outline:none;
  }
  input:focus{
    border-color: rgba(29, 78, 216, 0.65);
    box-shadow: 0 0 0 3px rgba(29, 78, 216, 0.16);
  }
  button{
    margin-top:16px;
    width:100%;
    padding:14px;
    border:none;
    border-radius:16px;
    background:linear-gradient(135deg,var(--wink-orange),var(--wink-blue));
    color:#fff;
    font-size:15px;
    font-weight:800;
    cursor:pointer;
  }
  button:hover{ filter:brightness(1.03); }
  .flash{
    margin-top:10px;
    font-size:13px;
    color:#b91c1c;
  }
</style>
</head>
<body>
  <div class="card">
    <div class="top">
      <div class="avatar">
        <img src="/static/wink.jpeg" style="width:100%;height:100%;object-fit:cover;">
      </div>
      <div>
        <h1>Set up your WINK space</h1>
        <p>This email is not registered yet. Add your name to create your instructor space.</p>
      </div>
    </div>

    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="flash">{{ messages[0] }}</div>
      {% endif %}
    {% endwith %}

    <form method="post">
      <input type="hidden" name="email" value="{{ email }}">
      <input type="text" name="name" placeholder="Your name" required>
      <button type="submit">Create Instructor</button>
    </form>
  </div>
</body>
</html>
"""






TEMPLATE_COMMON_WINK_FILES = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Common WINK Resources</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{
    --wink-navy:#041e42;
    --wink-blue:#1d4ed8;
    --wink-orange:#f97316;
  }
  body{
    margin:0;
    font-family:system-ui, -apple-system, Segoe UI, sans-serif;
    background:radial-gradient(circle at top left,#0b1120,#020617);
    min-height:100vh;
    display:flex;
    justify-content:center;
    padding:22px;
    color:#0b1120;
  }
  .card{
    width:100%;
    max-width:900px;
    background:#fff;
    border-radius:26px;
    box-shadow:0 28px 70px rgba(15,23,42,0.45);
    overflow:hidden;
  }
  .header{
    padding:22px;
    background:linear-gradient(135deg,var(--wink-navy),var(--wink-blue),var(--wink-orange));
    color:#fff;
  }
  .title{
    font-size:22px;
    font-weight:900;
  }
  .sub{
    margin-top:6px;
    font-size:14px;
    opacity:0.95;
  }
  .body{
    padding:22px;
  }
  table{
    width:100%;
    border-collapse:collapse;
    background:#fff;
    border-radius:16px;
    overflow:hidden;
  }
  th{
    padding:12px;
    text-align:left;
    font-size:13px;
    font-weight:900;
    background:#f1f5f9;
    border-bottom:1px solid #e2e8f0;
  }
  td{
    padding:12px;
    font-size:13px;
    border-bottom:1px solid #e2e8f0;
  }
  .empty{
    font-size:14px;
    color:#334155;
  }
  .actions{
    margin-top:18px;
    display:flex;
    justify-content:flex-end;
  }
  button{
    padding:10px 14px;
    border-radius:14px;
    border:none;
    background:linear-gradient(135deg,var(--wink-orange),var(--wink-blue));
    color:#fff;
    font-weight:800;
    font-size:14px;
    cursor:pointer;
  }
</style>
</head>
<body>
<div class="card">
  <div class="header">
    <div class="title">Common WINK Resources</div>
    <div class="sub">Shared materials available to all WINK courses</div>
  </div>

  <div class="body">
    {% if common_files and common_files|length > 0 %}
      <table>
        <thead>
          <tr>
            <th>Filename</th>
          </tr>
        </thead>
        <tbody>
          {% for f in common_files %}
            <tr>
              <td>{{ f }}</td>
            </tr>
          {% endfor %}
        </tbody>
      </table>
    {% else %}
      <div class="empty">
        No common WINK resources are currently available.
      </div>
    {% endif %}

    <div class="actions">
      <a href="{{ url_for('manage_files', instructor_id=instructor.id) }}" style="text-decoration:none;">
        <button type="button">Back to Manage Files</button>
      </a>
    </div>
  </div>
</div>
</body>
</html>
"""



TEMPLATE_MANAGE_FILES = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Manage Files</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{
    --wink-navy:#041e42;
    --wink-blue:#1d4ed8;
    --wink-orange:#f97316;
  }
  body{
    margin:0;
    font-family:system-ui, -apple-system, Segoe UI, sans-serif;
    background:radial-gradient(circle at top left,#0b1120,#020617);
    min-height:100vh;
    display:flex;
    justify-content:center;
    padding:22px;
    color:#0b1120;
  }
  .card{
    width:100%;
    max-width:1040px;
    background:#fff;
    border-radius:26px;
    box-shadow:0 28px 70px rgba(15,23,42,0.45);
    overflow:hidden;
  }
  .header{
    padding:22px;
    background:linear-gradient(135deg,var(--wink-navy),var(--wink-blue),var(--wink-orange));
    color:#fff;
  }
  .topbar{
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:12px;
    flex-wrap:wrap;
  }
  .title{
    font-size:20px;
    font-weight:900;
  }
  .sub{
    margin-top:6px;
    font-size:13px;
    opacity:0.95;
  }
  .body{ padding:22px; }
  .grid{
    display:grid;
    grid-template-columns:1fr;
    gap:16px;
  }
  .panel{
    border:1px solid #e2e8f0;
    border-radius:20px;
    padding:18px;
    background:#f8fafc;
  }
  button{
    padding:12px 16px;
    border-radius:14px;
    border:none;
    background:linear-gradient(135deg,var(--wink-orange),var(--wink-blue));
    color:#fff;
    font-weight:800;
    font-size:14px;
    cursor:pointer;
  }
  button.small-btn{
    padding:8px 12px;
    font-size:13px;
    font-weight:800;
  }
  button:hover{ filter:brightness(1.03); }
  table{
    width:100%;
    border-collapse:collapse;
    background:#fff;
    border-radius:14px;
    overflow:hidden;
  }
  th{
    padding:10px 12px;
    border-bottom:1px solid #e2e8f0;
    text-align:left;
    font-size:13px;
    font-weight:900;
    background:#f1f5f9;
  }
  td{
    padding:10px 12px;
    border-bottom:1px solid #e2e8f0;
    text-align:left;
    font-size:13px;
  }
  .actions{
    display:flex;
    justify-content:center;
    margin-top:14px;
  }
  .small{
    font-size:12px;
    margin-top:8px;
    color:#334155;
  }
  .copy-box{
    display:flex;
    gap:10px;
    align-items:center;
    flex-wrap:wrap;
  }
  .copy-input{
    flex:1;
    min-width:260px;
    padding:10px 12px;
    border-radius:12px;
    border:1px solid #cbd5e1;
    font-size:13px;
  }
  .file-picker-box{
    margin-top:8px;
    border:1px dashed #cbd5e1;
    border-radius:16px;
    padding:12px;
    background:#fff;
  }
  .file-list{
    margin-top:8px;
    font-size:13px;
    color:#334155;
    line-height:1.4;
  }
  .file-list div{ padding:2px 0; }
  .progress-wrapper{
    margin-top:10px;
    width:100%;
    height:8px;
    background:#e5e7eb;
    border-radius:8px;
    overflow:hidden;
  }
  .progress-bar{
    height:100%;
    width:0%;
    background:linear-gradient(135deg,var(--wink-orange),var(--wink-blue));
    transition:width 0.2s ease;
  }
  .flashline{
    margin-top:10px;
    font-size:13px;
    background:rgba(255,255,255,0.18);
    border:1px solid rgba(255,255,255,0.35);
    padding:8px 10px;
    border-radius:12px;
  }
</style>
</head>
<body>
<div class="card">

  <div class="header">
    <div class="topbar">
      <div>
        <div class="title">{{ instructor.name or instructor.email }}</div>
        <div class="sub">Manage files for your WINK course</div>
      </div>

      <a href="/wink/{{ instructor.slug }}" style="text-decoration:none;">
        <button type="button">Open WINK Chat</button>
      </a>
    </div>

    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}
        {% for category, msg in messages %}
          <div class="flashline">{{ msg }}</div>
        {% endfor %}
      {% endif %}
    {% endwith %}
  </div>

  <div class="body">
    <div class="grid">

      <div class="panel">
        <div style="margin-bottom:8px;font-weight:900;">Student chat link</div>
        <div class="copy-box">
          <input id="studentLink" class="copy-input" type="text" readonly
                 value="{{ request.host_url }}wink/{{ instructor.slug }}">
          <button type="button" class="small-btn"
                  onclick="navigator.clipboard.writeText(document.getElementById('studentLink').value)">
            Copy
          </button>
        </div>
        <div class="small">Share this link with students so they can access WINK.</div>
      </div>






<div class="panel">
  <div style="margin-bottom:8px;font-weight:900;">Upload course materials</div>

  <form id="uploadForm" method="post" enctype="multipart/form-data">
    <input type="file" id="fileInput" name="files" multiple required style="display:none;">

    <div class="file-picker-box">
      <button type="button" class="small-btn" onclick="document.getElementById('fileInput').click()">
        Choose files
      </button>

      <div id="fileList" class="file-list">No files selected</div>

      <div class="progress-wrapper">
        <div id="uploadProgress" class="progress-bar"></div>
      </div>
    </div>

    <div class="actions">
      <button type="submit" name="action" value="upload">Upload to WINK</button>
    </div>

    <div class="small">Uploaded files are added to your course knowledge base.</div>
  </form>
</div>


  





      
        

<div class="panel">
        <div style="margin-bottom:10px;font-weight:900;">Your course files</div>

        {% if files and files|length > 0 %}
        <table>
          <thead>
            <tr>
              <th>Filename</th>
              <th>Uploaded</th>
              <th style="width:120px;"></th>
            </tr>
          </thead>
          <tbody>
            {% for f in files %}
            <tr>
              <td>{{ f.filename or '(unnamed)' }}</td>
              <td>{{ f.uploaded_at.strftime('%Y-%m-%d %H:%M') if f.uploaded_at else '' }}</td>
              <td>
                <form method="post" style="margin:0;">
                  <input type="hidden" name="file_id" value="{{ f.file_id }}">
                  <button type="submit" name="action" value="delete" class="small-btn">Delete</button>
                </form>
              </td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
        {% else %}
          <div class="small">No files uploaded yet.</div>
        {% endif %}



<div class="actions" style="margin-top:18px;">
  <a href="{{ url_for('common_wink_files', instructor_id=instructor.id) }}" style="text-decoration:none;">
    <button type="button">View Common WINK Files</button>
  </a>
</div>



      </div>

      
    </div>
  </div>
</div>



<script>
  const fileInput = document.getElementById("fileInput");
  const fileList = document.getElementById("fileList");
  const progressBar = document.getElementById("uploadProgress");
  const uploadForm = document.getElementById("uploadForm");

  if (fileInput && fileList) {
    fileInput.addEventListener("change", function () {
      fileList.innerHTML = "";
      if (!this.files || this.files.length === 0) {
        fileList.textContent = "No files selected";
        return;
      }
      Array.from(this.files).forEach(file => {
        const div = document.createElement("div");
        div.textContent = file.name;
        fileList.appendChild(div);
      });
    });
  }

  if (uploadForm && progressBar) {
    uploadForm.addEventListener("submit", function (e) {
      e.preventDefault();
      progressBar.style.width = "0%";

      const formData = new FormData(uploadForm);
      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener("progress", function (e) {
        if (e.lengthComputable) {
          const percent = Math.round((e.loaded / e.total) * 100);
          progressBar.style.width = percent + "%";
        }
      });

      xhr.addEventListener("load", function () {
        progressBar.style.width = "100%";
        window.location.reload();
      });

      xhr.addEventListener("error", function () {
        window.location.reload();
      });

      xhr.open("POST", window.location.href);
      xhr.send(formData);
    });
  }
</script>

</body>
</html>
"""


TEMPLATE_INSTRUCTOR_LIST = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>WINK Instructors</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{
    --wink-navy:#041e42;
    --wink-blue:#1d4ed8;
    --wink-orange:#f97316;
  }
  body{
    margin:0;
    font-family:system-ui, -apple-system, Segoe UI, sans-serif;
    background:radial-gradient(circle at top left,#0b1120,#020617);
    min-height:100vh;
    display:flex;
    justify-content:center;
    padding:24px;
    color:#0b1120;
  }
  .card{
    width:100%;
    max-width:1120px;
    background:#fff;
    border-radius:26px;
    box-shadow:0 28px 70px rgba(15,23,42,0.45);
    overflow:hidden;
  }
  .header{
    padding:22px;
    background:linear-gradient(135deg,var(--wink-navy),var(--wink-blue),var(--wink-orange));
    color:#fff;
  }
  .header-title{
    font-size:22px;
    font-weight:900;
  }
  .header-sub{
    margin-top:6px;
    font-size:14px;
    opacity:0.95;
  }
  .body{ padding:22px; }
  table{
    width:100%;
    border-collapse:collapse;
    background:#fff;
    border-radius:16px;
    overflow:hidden;
  }
  th{
    padding:10px 12px;
    text-align:left;
    font-size:13px;
    font-weight:900;
    background:#f1f5f9;
    border-bottom:1px solid #e2e8f0;
  }
  td{
    padding:10px 12px;
    font-size:13px;
    border-bottom:1px solid #e2e8f0;
  }
  tr:last-child td{ border-bottom:none; }
  .copy-box{
    display:flex;
    gap:8px;
    align-items:center;
  }
  .copy-input{
    flex:1;
    padding:9px 10px;
    border-radius:12px;
    border:1px solid #cbd5e1;
    font-size:13px;
  }
  button{
    padding:9px 12px;
    border-radius:12px;
    border:none;
    background:linear-gradient(135deg,var(--wink-orange),var(--wink-blue));
    color:#fff;
    font-size:13px;
    font-weight:900;
    cursor:pointer;
  }
  button:hover{ filter:brightness(1.05); }
  .small{ font-size:13px; color:#334155; }
</style>
</head>
<body>
<div class="card">
  <div class="header">
    <div class="header-title">WINK Instructors</div>
    <div class="header-sub">Registered instructors and their student chat links</div>
  </div>

  <div class="body">
    {% if instructors %}
    <table>
      <thead>
        <tr>
          <th>Instructor</th>
          <th>Email</th>
          <th>Student Chat Link</th>
        </tr>
      </thead>
      <tbody>
        {% for i in instructors %}
        <tr>
          <td>{{ i.name or "—" }}</td>
          <td>{{ i.email }}</td>
          <td>
            <div class="copy-box">
              <input type="text" class="copy-input" readonly
                     value="{{ request.host_url }}wink/{{ i.slug }}">
              <button type="button"
                      onclick="navigator.clipboard.writeText('{{ request.host_url }}wink/{{ i.slug }}')">
                Copy
              </button>
            </div>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% else %}
      <div class="small">No instructors registered yet.</div>
    {% endif %}
  </div>
</div>
</body>
</html>
"""


TEMPLATE_WINK_CHAT = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WINK for {{ instructor.name or instructor.email }}</title>
<style>
  :root {
    --wink-navy: #041e42;
    --wink-deep: #0f172a;
    --wink-blue: #1d4ed8;
    --wink-orange: #f97316;

    --surface-strong: #ffffff;
    --border: rgba(148, 163, 184, 0.35);
    --shadow: 0 26px 60px rgba(15, 23, 42, 0.45);

    --text: #0b1120;
    --muted: #475569;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: radial-gradient(circle at top left, #0b1120 0, #020617 42%, #0f172a 100%);
    color: var(--text);
  }

  .wink-shell {
    display: flex;
    min-height: 100vh;
    backdrop-filter: blur(26px);
  }

  .wink-left {
    width: 350px;
    padding: 22px 18px;
    background:
      linear-gradient(160deg, rgba(4, 30, 66, 0.95), rgba(15, 23, 42, 0.98)),
      radial-gradient(circle at top right, rgba(249, 115, 22, 0.20), transparent 55%);
    color: #e5e7eb;
    overflow-y: auto;
    border-right: 1px solid rgba(148, 163, 184, 0.45);
  }

  .wink-left-inner {
    background:
      radial-gradient(circle at top left, rgba(219, 234, 254, 0.08), transparent 55%),
      radial-gradient(circle at bottom right, rgba(254, 215, 170, 0.12), transparent 60%),
      rgba(15, 23, 42, 0.9);
    border-radius: 20px;
    box-shadow:
      0 20px 45px rgba(0, 0, 0, 0.60),
      0 0 0 1px rgba(148, 163, 184, 0.35);
    padding: 18px 16px 20px 16px;
  }

  .wink-right {
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: stretch;
    padding: 22px;
    background:
      radial-gradient(circle at top left, rgba(219, 234, 254, 0.40), transparent 55%),
      radial-gradient(circle at bottom right, rgba(254, 215, 170, 0.35), transparent 60%),
      linear-gradient(to bottom right, #0b1120, #020617);
  }

  .wink-right-inner {
    width: 100%;
    max-width: 1548px;
    display: flex;
    flex-direction: column;
    background:
      radial-gradient(circle at top left, rgba(219, 234, 254, 0.45), transparent 55%),
      radial-gradient(circle at bottom right, rgba(254, 215, 170, 0.40), transparent 60%),
      var(--surface-strong);
    border-radius: 22px;
    box-shadow: var(--shadow);
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.22);
  }

  .wink-header {
    padding: 12px 16px;
    background: linear-gradient(135deg, var(--wink-navy), var(--wink-blue), var(--wink-orange));
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .wink-header-left {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
  }

  .wink-header-avatar {
    width: 34px;
    height: 34px;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.28);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    font-weight: 700;
    box-shadow: 0 0 0 1px rgba(248, 250, 252, 0.35);
    flex: 0 0 auto;
  }

  .wink-header-title-block {
    display: flex;
    flex-direction: column;
    min-width: 0;
  }

  .wink-header-title {
    font-weight: 800;
    font-size: 16px;
    line-height: 1.1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .wink-header-subtitle {
    font-size: 12px;
    opacity: 0.92;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .wink-header-pill {
    font-size: 11px;
    padding: 5px 10px;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.55);
    background: rgba(15, 23, 42, 0.22);
    white-space: nowrap;
  }

  .wink-mobile-toggle {
    display: none;
    border: 1px solid rgba(255,255,255,0.55);
    background: rgba(15, 23, 42, 0.18);
    color: #fff;
    border-radius: 999px;
    padding: 6px 10px;
    cursor: pointer;
    font-size: 12px;
    white-space: nowrap;
  }

  .wink-flash {
    padding: 8px 16px 0 16px;
    background: linear-gradient(to bottom, rgba(249,250,251,0.92), rgba(255,255,255,0.98));
  }

  .wink-flash-message {
    font-size: 13px;
    padding: 7px 10px;
    border-radius: 10px;
    background: rgba(219, 234, 254, 0.75);
    color: #111827;
    border: 1px solid rgba(59, 130, 246, 0.55);
    margin-bottom: 6px;
  }

  .wink-chat {
    flex: 1;
    padding: 14px 16px 14px 16px;
    overflow-y: auto;
    background:
      radial-gradient(circle at top left, rgba(219, 234, 254, 0.22), transparent 45%),
      radial-gradient(circle at bottom right, rgba(254, 215, 170, 0.20), transparent 55%),
      linear-gradient(to bottom, rgba(249, 250, 251, 0.98), rgba(255, 255, 255, 0.98));
  }

  .message-row {
    margin-bottom: 12px;
    display: flex;
  }

  .message-row.user { justify-content: flex-end; }
  .message-row.assistant { justify-content: flex-start; }

  .message-bubble {
    max-width: 76%;
    padding: 10px 12px;
    border-radius: 16px;
    font-size: 15px;
    line-height: 1.45;
    white-space: pre-wrap;
    word-wrap: break-word;
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.12);
  }

  .message-row.user .message-bubble {
    background: linear-gradient(135deg, var(--wink-blue), var(--wink-navy));
    color: #ffffff;
    border-bottom-right-radius: 6px;
  }

  .message-row.assistant .message-bubble {
    background: rgba(255,255,255,0.92);
    color: #0b1120;
    border: 1px solid rgba(148, 163, 184, 0.32);
    border-bottom-left-radius: 6px;
  }

  .wink-empty-state {
    color: var(--muted);
    font-size: 14px;
    padding: 12px 12px;
    border-radius: 14px;
    background: linear-gradient(to right, rgba(219, 234, 254, 0.55), rgba(254, 215, 170, 0.55));
    border: 1px dashed rgba(148, 163, 184, 0.65);
  }

  .wink-input-bar {
    padding: 10px 12px 12px 12px;
    border-top: 1px solid rgba(209, 213, 219, 0.75);
    background: linear-gradient(to right, rgba(249, 250, 251, 0.96), rgba(255, 255, 255, 0.98));
  }

  .wink-input-form {
    display: flex;
    gap: 10px;
    align-items: flex-end;
  }

  .wink-textarea-wrapper {
    position: relative;
    flex: 1;
  }

  textarea[name="message"] {
    width: 100%;
    min-height: 52px;
    max-height: 180px;
    padding: 12px 130px 12px 54px;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.55);
    font-family: inherit;
    font-size: 14px;
    background: rgba(255,255,255,0.98);
    box-shadow: 0 0 0 1px rgba(148, 163, 184, 0.18);
    outline: none;
    resize: none;
  }

  textarea[name="message"]:focus {
    border-color: rgba(29, 78, 216, 0.65);
    box-shadow: 0 0 0 3px rgba(29, 78, 216, 0.18);
  }

  .wink-mic-icon {
    position: absolute;
    left: 10px;
    bottom: 30px;
    font-size: 16px;
    color: var(--wink-blue);
    background: rgba(219, 234, 254, 0.92);
    border-radius: 999px;
    padding: 2px 6px;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25);
    user-select: none;
    cursor: pointer;
  }

  .wink-mic-icon:active { transform: scale(0.98); }

  .wink-file-controls {
    position: absolute;
    right: 10px;
    bottom: 10px;
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .wink-attach-button {
    border: 1px solid rgba(148, 163, 184, 0.55);
    background: rgba(249,250,251,0.95);
    cursor: pointer;
    font-size: 16px;
    line-height: 1;
    padding: 6px 10px;
    border-radius: 12px;
    color: #0b1120;
  }

  .wink-file-button {
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.55);
    background: rgba(249,250,251,0.95);
    padding: 6px 10px;
    font-size: 12px;
    cursor: pointer;
    color: #0b1120;
  }

  .wink-send-button {
    border: none;
    background: linear-gradient(135deg, var(--wink-orange), var(--wink-blue));
    color: #ffffff;
    padding: 12px 16px;
    border-radius: 14px;
    font-size: 14px;
    cursor: pointer;
    box-shadow: 0 10px 20px rgba(15, 23, 42, 0.25);
  }

  .wink-hidden-file-input { display: none; }

  .wink-selected-files {
    margin-top: 6px;
    font-size: 12px;
    color: #334155;
    min-height: 16px;
    padding-left: 2px;
  }

  .wink-chat::-webkit-scrollbar { width: 8px; }
  .wink-chat::-webkit-scrollbar-track { background: transparent; }
  .wink-chat::-webkit-scrollbar-thumb {
    background-color: rgba(148, 163, 184, 0.75);
    border-radius: 999px;
  }

  @media (max-width: 920px) {
    .wink-shell { flex-direction: column; }
    .wink-left {
      width: 100%;
      max-height: 46vh;
      border-right: none;
      border-bottom: 1px solid rgba(148, 163, 184, 0.45);
      display: none;
    }
    .wink-left.show { display: block; }
    .wink-right { padding: 12px; }
    .wink-mobile-toggle { display: inline-flex; }
  }
</style>
</head>
<body>
<div class="wink-shell">
  <div class="wink-left" id="wink-left-panel">
    <div class="wink-left-inner">
      {{ left_column_html | safe }}
    </div>
  </div>

  <div class="wink-right">
    <div class="wink-right-inner">
      <div class="wink-header">
        <div class="wink-header-left">
          <div class="wink-header-avatar">W</div>
          <div class="wink-header-title-block">
            <div class="wink-header-title">WINK</div>
            <div class="wink-header-subtitle">
              {{ instructor.name or instructor.email }} • What I Need to Know
            </div>
          </div>
        </div>

        <button class="wink-mobile-toggle" type="button" id="toggle-left">
          Instructor panel
        </button>

        {% if instructor.slug %}
          <div class="wink-header-pill">/wink/{{ instructor.slug }}</div>
        {% endif %}
      </div>

      <div class="wink-flash">
        {% with flash_messages = get_flashed_messages() %}
          {% if flash_messages %}
            {% for msg in flash_messages %}
              <div class="wink-flash-message">{{ msg }}</div>
            {% endfor %}
          {% endif %}
        {% endwith %}
      </div>

      <div class="wink-chat" id="wink-chat">
        {% if messages %}
          {% for m in messages %}
            <div class="message-row {{ m.role }}">
              <div class="message-bubble">{{ m.text }}</div>
            </div>
          {% endfor %}
          <div id="chat-end"></div>
        {% else %}
          <div class="wink-empty-state">
            Ask about your course, syllabus, assignments, campus resources, or how to succeed this week.
            Try “What is due this week?” or “Help me break the big project into steps.”
          </div>
          <div id="chat-end"></div>
        {% endif %}
      </div>

      <div class="wink-input-bar">
        <form class="wink-input-form" id="wink-form" method="post">
          <div class="wink-textarea-wrapper">
            <span class="wink-mic-icon" id="wink-mic" title="Start microphone" role="button" tabindex="0">🎤</span>
            <textarea id="wink-textarea" name="message" placeholder="Type your question here..." autocomplete="off"></textarea>
            <div class="wink-file-controls">
              <button type="button" class="wink-attach-button"
                      onclick="document.getElementById('file-input').click();"
                      title="Attach a file">+</button>
              <button type="button" class="wink-file-button"
                      onclick="document.getElementById('file-input').click();">
                Choose File
              </button>
            </div>
            <input id="file-input" class="wink-hidden-file-input" type="file" name="attachments" multiple />
          </div>
          <button class="wink-send-button" id="send-btn" type="submit">Send</button>
        </form>
        <div id="selected-files" class="wink-selected-files"></div>
      </div>
    </div>
  </div>
</div>

<script>
  const fileInput = document.getElementById('file-input');
  const selectedFilesDiv = document.getElementById('selected-files');
  const winkForm = document.getElementById('wink-form');
  const winkTextarea = document.getElementById('wink-textarea');
  const chatEl = document.getElementById('wink-chat');
  const chatEnd = document.getElementById('chat-end');
  const toggleLeftBtn = document.getElementById('toggle-left');
  const leftPanel = document.getElementById('wink-left-panel');

  function scrollChatToBottom() {
    if (chatEnd) {
      chatEnd.scrollIntoView({ behavior: "smooth", block: "end" });
      return;
    }
    if (chatEl) {
      chatEl.scrollTop = chatEl.scrollHeight;
    }
  }

  function autoGrowTextarea() {
    if (!winkTextarea) return;
    winkTextarea.style.height = "auto";
    winkTextarea.style.height = Math.min(winkTextarea.scrollHeight, 180) + "px";
  }

  if (fileInput && selectedFilesDiv) {
    fileInput.addEventListener('change', function () {
      const files = Array.from(fileInput.files || []);
      selectedFilesDiv.textContent = files.length
        ? ("Files selected: " + files.map(f => f.name).join(", "))
        : "";
    });
  }

  if (winkTextarea) {
    winkTextarea.addEventListener('input', autoGrowTextarea);
    winkTextarea.addEventListener('keydown', function (event) {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        if (winkForm) winkForm.submit();
      }
    });
  }

  if (toggleLeftBtn && leftPanel) {
    toggleLeftBtn.addEventListener('click', function () {
      leftPanel.classList.toggle('show');
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    autoGrowTextarea();
    scrollChatToBottom();
    if (winkTextarea) winkTextarea.focus();
  });

  // ===============================
  // Microphone -> Speech to Text
  // ===============================
  const micBtn = document.getElementById("wink-mic");
  let recognition = null;
  let isListening = false;

  function setMicUI(listening) {
    if (!micBtn) return;
    isListening = listening;
    micBtn.textContent = listening ? "🛑" : "🎤";
    micBtn.title = listening ? "Stop microphone" : "Start microphone";
  }

  function initSpeechRecognition() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return null;

    const r = new SR();
    r.continuous = false;
    r.interimResults = true;
    r.lang = "en-US";

    r.onstart = function () { setMicUI(true); };
    r.onerror = function () { setMicUI(false); };
    r.onend = function () { setMicUI(false); };

    r.onresult = function (event) {
      let finalText = "";
      let interimText = "";

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalText += t;
        else interimText += t;
      }

      if (!winkTextarea) return;

      const existing = winkTextarea.value || "";
      const append = (finalText || interimText || "").trim();
      if (!append) return;

      winkTextarea.value = existing ? (existing + " " + append) : append;
      winkTextarea.dispatchEvent(new Event("input"));
      winkTextarea.focus();
    };

    return r;
  }

  if (micBtn) {
    recognition = initSpeechRecognition();

    const startStop = function () {
      if (!recognition) {
        alert("Speech recognition is not supported in this browser. Use Chrome or Edge.");
        return;
      }
      try {
        if (isListening) recognition.stop();
        else recognition.start();
      } catch (e) {
        setMicUI(false);
      }
    };

    micBtn.addEventListener("click", startStop);
    micBtn.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        startStop();
      }
    });
  }
</script>
</body>
</html>
"""


# ============================================
# Routes
# ============================================

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        email = _clean_email(request.form.get("email", ""))
        if not email:
            flash("Please enter a valid email address.")
            return render_template_string(TEMPLATE_LOGIN_PAGE)

        instructor = Instructor.query.filter(db.func.lower(Instructor.email) == email).first()
        if instructor:
            return redirect(url_for("manage_files", instructor_id=instructor.id))

        return redirect(url_for("new_instructor", email=email))

    return render_template_string(TEMPLATE_LOGIN_PAGE)



@app.route("/admin/common_wink_files/<int:instructor_id>")
def common_wink_files(instructor_id: int):
    instructor = Instructor.query.get_or_404(instructor_id)
    common_files = get_common_filenames()

    return render_template_string(
        TEMPLATE_COMMON_WINK_FILES,
        instructor=instructor,
        common_files=common_files,
    )




@app.route("/admin/new_instructor", methods=["GET", "POST"])
def new_instructor():
    if request.method == "GET":
        email = _clean_email(request.args.get("email", ""))
        return render_template_string(TEMPLATE_NEW_INSTRUCTOR, email=email)

    email = _clean_email(request.form.get("email", ""))
    name = (request.form.get("name", "") or "").strip()

    if not email or not name:
        flash("Please complete both email and name.")
        return render_template_string(TEMPLATE_NEW_INSTRUCTOR, email=email)

    existing = Instructor.query.filter_by(email=email).first()
    if existing:
        return redirect(url_for("manage_files", instructor_id=existing.id))

    slug = _unique_slug(email)

    try:
        vs_id = openai_http.create_vector_store(f"WINK - {name}")
    except Exception as e:
        flash(f"Could not create vector store. Check your OpenAI key and try again. Details: {e}")
        return render_template_string(TEMPLATE_NEW_INSTRUCTOR, email=email)

    instructor = Instructor(
        email=email,
        name=name,
        slug=slug,
        personal_vector_store_id=vs_id,
    )
    db.session.add(instructor)
    db.session.commit()

    return redirect(url_for("manage_files", instructor_id=instructor.id))


@app.route("/admin/manage_files/<int:instructor_id>", methods=["GET", "POST"])
def manage_files(instructor_id: int):
    instructor = Instructor.query.get_or_404(instructor_id)

    if request.method == "POST":
        action = (request.form.get("action", "") or "").strip().lower()

        # Be forgiving if JS posts without "action"
        if not action:
            incoming_files = request.files.getlist("files") if request.files else []
            has_files = any(getattr(f, "filename", "") for f in (incoming_files or []))
            if has_files:
                action = "upload"
            elif (request.form.get("file_id") or "").strip():
                action = "delete"

        if action == "upload":
            if not instructor.personal_vector_store_id:
                flash("No vector store found for this instructor.", "bad")
                return redirect(url_for("manage_files", instructor_id=instructor.id))

            files = request.files.getlist("files")
            if not files or not files[0] or not files[0].filename:
                flash("Please choose at least one file to upload.", "bad")
                return redirect(url_for("manage_files", instructor_id=instructor.id))

            uploaded_count = 0
            for f in files:
                if not f or not f.filename:
                    continue

                safe_name = secure_filename(f.filename)
                stamp = int(time.time() * 1000)
                local_path = os.path.join(UPLOAD_DIR, f"{stamp}_{safe_name}")
                f.save(local_path)

                try:
                    file_id = openai_http.upload_file(local_path, safe_name)
                    openai_http.add_file_to_vector_store(instructor.personal_vector_store_id, file_id)

                    rec = InstructorFile(
                        instructor_id=instructor.id,
                        file_id=file_id,
                        filename=safe_name,
                    )
                    db.session.add(rec)
                    db.session.commit()
                    uploaded_count += 1
                except Exception as e:
                    flash(f"Upload failed for {safe_name}: {e}", "bad")
                finally:
                    try:
                        os.remove(local_path)
                    except Exception:
                        pass

            if uploaded_count > 0:
                flash(f"Uploaded {uploaded_count} file(s) to WINK.", "ok")

            return redirect(url_for("manage_files", instructor_id=instructor.id))

        if action == "delete":
            file_id = (request.form.get("file_id", "") or "").strip()
            if not file_id:
                flash("Missing file_id.", "bad")
                return redirect(url_for("manage_files", instructor_id=instructor.id))

            rec = InstructorFile.query.filter_by(instructor_id=instructor.id, file_id=file_id).first()

            try:
                if instructor.personal_vector_store_id:
                    openai_http.delete_file_from_vector_store(instructor.personal_vector_store_id, file_id)
            except Exception as e:
                flash(f"Could not remove file from vector store: {e}", "bad")
                return redirect(url_for("manage_files", instructor_id=instructor.id))

            if rec:
                db.session.delete(rec)
                db.session.commit()

            flash("File deleted.", "ok")
            return redirect(url_for("manage_files", instructor_id=instructor.id))

        flash("Unknown action.", "bad")
        return redirect(url_for("manage_files", instructor_id=instructor.id))

    files = (
        InstructorFile.query
        .filter_by(instructor_id=instructor.id)
        .order_by(InstructorFile.uploaded_at.desc())
        .all()
    )
    common_files = get_common_filenames()
    return render_template_string(
        TEMPLATE_MANAGE_FILES,
        instructor=instructor,
        files=files,
        common_files=common_files,
    )


@app.route("/admin/instructors")
def instructor_list():
    instructors = Instructor.query.order_by(Instructor.created_at.desc()).all()
    return render_template_string(TEMPLATE_INSTRUCTOR_LIST, instructors=instructors)


@app.route("/wink/<slug>", methods=["GET", "POST"])
def wink_chat(slug: str):
    instructor = Instructor.query.filter_by(slug=slug).first_or_404()

    left_column_html = instructor.left_column_html or build_default_left_column_html(instructor.name or instructor.email)
    left_column_html = sanitize_left_column_html(left_column_html)

    session_key = f"wink_messages_{instructor.id}"
    messages = _safe_session_list(session_key)

    if request.method == "POST":
        if request.form.get("reset"):
            session.pop(session_key, None)
            messages = []
        else:
            user_message = (request.form.get("message", "") or "").strip()
            if user_message:
                messages.append({"role": "user", "text": user_message})
                messages = _trim_history(messages, max_messages=30)

                assistant_text = wink_answer(instructor, messages)
                messages.append({"role": "assistant", "text": assistant_text})
                messages = _trim_history(messages, max_messages=30)

                session[session_key] = messages

    return render_template_string(
        TEMPLATE_WINK_CHAT,
        instructor=instructor,
        left_column_html=left_column_html,
        messages=messages,
    )


# ============================================
# Run
# ============================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
