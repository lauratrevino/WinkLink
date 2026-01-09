import os
from datetime import datetime
import re

try:
    import requests
except ImportError:
    requests = None


from flask import (
Flask,
request,
render_template_string,
redirect,
url_for,
flash,
session,
)


from dotenv import load_dotenv

load_dotenv()

# MUST exist for all instructors to use the Common WINK vector store
COMMON_VECTOR_STORE_ID = os.getenv("WINK_VECTOR_STORE_ID")

if not COMMON_VECTOR_STORE_ID:
    raise RuntimeError(
        "WINK_VECTOR_STORE_ID is not set. "
        "Set this environment variable to the Common WINK vector store ID."
    )








from openai import OpenAI
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

# ============================================
# Setup
# ============================================

load_dotenv()

client = OpenAI()
db = SQLAlchemy()

COMMON_VECTOR_STORE_ID = os.getenv("WINK_VECTOR_STORE_ID")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-secret")

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///wink.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")

UPLOAD_DIR = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================
# Database models
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
    instructor_id = db.Column(
        db.Integer,
        db.ForeignKey("instructor.id"),
        nullable=False
    )
    file_id = db.Column(db.String(255), nullable=False)
    filename = db.Column(db.String(255), nullable=True)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    instructor = db.relationship(
        "Instructor",
        primaryjoin="Instructor.id == InstructorFile.instructor_id",
        foreign_keys=[instructor_id],
        backref="files"
    )


with app.app_context():
    db.create_all()

# ============================================
# Vector store helpers
# ============================================

def _require_requests():
    if requests is None:
        raise RuntimeError("The 'requests' library is required for vector store operations. Install with: pip install requests")


def create_vector_store_http(name: str) -> str:
    _require_requests()
    resp = requests.post(
        f"{OPENAI_BASE_URL}/v1/vector_stores",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2",
        },
        json={"name": name},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["id"]


def add_file_to_vector_store_http(vector_store_id: str, file_id: str) -> None:
    _require_requests()
    requests.post(
        f"{OPENAI_BASE_URL}/v1/vector_stores/{vector_store_id}/file_batches",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2",
        },
        json={"file_ids": [file_id]},
        timeout=30,
    )


def delete_file_from_vector_store_http(vector_store_id: str, file_id: str) -> None:
    _require_requests()
    requests.delete(
        f"{OPENAI_BASE_URL}/v1/vector_stores/{vector_store_id}/files/{file_id}",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "assistants=v2",
        },
        timeout=30,
    )


def upload_openai_file_http(filepath: str, filename: str) -> str:
    _require_requests()
    with open(filepath, "rb") as f:
        resp = requests.post(
            f"{OPENAI_BASE_URL}/v1/files",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"file": (filename, f), "purpose": (None, "assistants")},
            timeout=60,
        )
    resp.raise_for_status()
    return resp.json()["id"]


# ============================================
# Templates
# ============================================

TEMPLATE_LOGIN_PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>WINK Instructor Access</title>
<style>
body{
  margin:0;
  font-family:system-ui;
  background:radial-gradient(circle at top left,#0b1120,#020617);
  min-height:100vh;
  display:flex;
  align-items:center;
  justify-content:center;
}
.card{
  width:100%;
  max-width:520px;
  background:#fff;
  border-radius:26px;
  box-shadow:0 28px 70px rgba(15,23,42,0.45);
}
.header{
  padding:20px;
  background:linear-gradient(135deg,#041e42,#1d4ed8,#f97316);
  color:#fff;
  text-align:center;
}
.body{padding:22px;text-align:center;}
.avatar{
  width:140px;height:140px;margin:0 auto 14px;border-radius:50%;overflow:hidden;
}
input{
  width:100%;
  padding:14px;
  border-radius:16px;
  border:1px solid #cbd5e1;
  font-size:15px;
}
button{
  margin-top:18px;
  padding:14px 20px;
  border:none;
  border-radius:16px;
  font-size:15px;
  font-weight:700;
  color:#fff;
  background:linear-gradient(135deg,#f97316,#1d4ed8);
  cursor:pointer;
}
.hint{margin-top:10px;font-size:13px;color:#475569;}
.flash{
  margin:14px 0 0;
  font-size:13px;
  color:#b91c1c;
}
</style>
</head>
<body>
<div class="card">
  <div class="header">
    <h1>WINK Instructor Access</h1>
    <p>Manage your course materials</p>
  </div>
  <div class="body">
    <div class="avatar">
      <img src="/static/wink.jpeg" style="width:100%;height:100%;object-fit:cover;">
    </div>

    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="flash">{{ messages[0] }}</div>
      {% endif %}
    {% endwith %}

    <form method="post">
      <input type="email" name="email" required placeholder="you@utep.edu">
      <div class="hint">
        Existing instructors go to file management.
        New instructors are guided to create their WINK page.
      </div>
      <button type="submit">Continue</button>
    </form>
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
<style>
body{
  margin:0;
  font-family:system-ui;
  background:radial-gradient(circle at top left,#0b1120,#020617);
  min-height:100vh;
  display:flex;
  align-items:center;
  justify-content:center;
  padding:20px;
}
.card{
  width:100%;
  max-width:640px;
  background:#fff;
  border-radius:26px;
  box-shadow:0 28px 70px rgba(15,23,42,0.45);
  overflow:hidden;
}
.header{
  padding:20px;
  background:linear-gradient(135deg,#041e42,#1d4ed8,#f97316);
  color:#fff;
}
.body{padding:22px;}
label{
  display:block;
  margin:14px 0 8px;
  font-size:13px;
  color:#334155;
  font-weight:700;
}
input{
  width:100%;
  padding:14px;
  border-radius:16px;
  border:1px solid #cbd5e1;
  font-size:15px;
}
.help{
  margin-top:8px;
  font-size:13px;
  color:#475569;
  line-height:1.35;
}
.row{
  display:grid;
  grid-template-columns:1fr;
  gap:14px;
}
.actions{
  margin-top:18px;
  display:flex;
  gap:12px;
  flex-wrap:wrap;
}
button,a.btn{
  padding:14px 18px;
  border-radius:16px;
  border:none;
  font-weight:700;
  color:#fff;
  background:linear-gradient(135deg,#f97316,#1d4ed8);
  cursor:pointer;
  text-decoration:none;
  display:inline-block;
}
a.link{
  color:#1d4ed8;
  text-decoration:none;
  font-weight:700;
}
.flash{
  margin:14px 0 0;
  font-size:13px;
  color:#b91c1c;
}
.small{
  font-size:12px;
  color:#64748b;
  margin-top:10px;
}
</style>
</head>
<body>
<div class="card">
  <div class="header">
    <h2>New Instructor Setup</h2>
    <div>Let’s create your WINK page and your personal vector store</div>
  </div>
  <div class="body">

    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="flash">{{ messages[0] }}</div>
      {% endif %}
    {% endwith %}

    <form method="post">
      <div class="row">
        <div>
          <label>Email</label>
          <input type="email" name="email" required value="{{ email or '' }}" placeholder="you@utep.edu">
          <div class="help">This should be your instructor email address.</div>
        </div>
        <div>
          <label>Name students will see</label>
          <input type="text" name="name" required placeholder="Dr. Trevino">
          <div class="help">Use the name you want students to call you in your WINK page.</div>
        </div>
      </div>

      <div class="actions">
        <button type="submit">Create my WINK page</button>
        <a class="btn" href="/">Back</a>
      </div>

      <div class="small">Your WINK URL will be created automatically from your email.</div>
    </form>
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
<style>
body{
  margin:0;
  font-family:system-ui;
  background:radial-gradient(circle at top left,#0b1120,#020617);
  min-height:100vh;
  display:flex;
  justify-content:center;
  padding:20px;
}
.card{
  width:100%;
  max-width:980px;
  background:#fff;
  border-radius:26px;
  box-shadow:0 28px 70px rgba(15,23,42,0.45);
  overflow:hidden;
}
.header{
  padding:20px;
  background:linear-gradient(135deg,#041e42,#1d4ed8,#f97316);
  color:#fff;
}
.header a{
  color:#fff;
  text-decoration:none;
  font-weight:800;
}
.sub{
  margin-top:8px;
  opacity:0.95;
}
.body{padding:22px;}
.grid{
  display:grid;
  grid-template-columns:1fr;
  gap:18px;
}
.panel{
  border:1px solid #e2e8f0;
  border-radius:20px;
  padding:18px;
  background:#f8fafc;
}
.upload-box{
  border:2px dashed #cbd5e1;
  border-radius:18px;
  padding:18px;
  text-align:center;
  background:#fff;
}
button{
  padding:12px 16px;
  border-radius:14px;
  border:none;
  background:linear-gradient(135deg,#f97316,#1d4ed8);
  color:#fff;
  font-weight:800;
  cursor:pointer;
}
table{
  width:100%;
  border-collapse:collapse;
  background:#fff;
  border-radius:16px;
  overflow:hidden;
}
th,td{
  padding:12px 10px;
  border-bottom:1px solid #e2e8f0;
  text-align:left;
  font-size:14px;
}
th{background:#f1f5f9;color:#334155;}
.badge{
  display:inline-block;
  padding:4px 10px;
  border-radius:999px;
  background:#0f172a;
  color:#fff;
  font-size:12px;
}
.actions{
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  margin-top:12px;
}
.linkrow{
  display:flex;
  gap:16px;
  flex-wrap:wrap;
  margin-top:10px;
}
.flash-ok{
  margin:10px 0 0;
  font-size:13px;
  color:#166534;
}
.flash-bad{
  margin:10px 0 0;
  font-size:13px;
  color:#b91c1c;
}
.small{
  font-size:12px;
  color:#64748b;
  margin-top:10px;
}
</style>
</head>
<body>
<div class="card">
  <div class="header">
    <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:center;">
      <div>
        <div style="font-size:18px;font-weight:900;">{{ instructor.name or instructor.email }}</div>
        <div class="sub">Manage files for your WINK vector store</div>
      </div>
      <div class="linkrow">
        <a href="/wink/{{ instructor.slug }}">Open WINK Chat</a>
        <a href="/">Instructor Access</a>
      </div>
    </div>

    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}
        {% for category, msg in messages %}
          {% if category == 'ok' %}
            <div class="flash-ok">{{ msg }}</div>
          {% else %}
            <div class="flash-bad">{{ msg }}</div>
          {% endif %}
        {% endfor %}
      {% endif %}
    {% endwith %}
  </div>

  <div class="body">
    <div class="grid">

      <div class="panel">
        <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;justify-content:space-between;">
          <div>
            <span class="badge">Vector Store</span>
            <span style="margin-left:10px;font-size:13px;color:#334155;">
              {{ instructor.personal_vector_store_id or 'Not created yet' }}
            </span>
          </div>
          <div class="small">Upload course materials here so WINK can use them.</div>
        </div>










        <form method="post" enctype="multipart/form-data">
          <input type="file" name="files" multiple required>
          <div class="actions" style="justify-content:center;">
            <button type="submit" name="action" value="upload">Upload to WINK</button>
          </div>
          <div class="small">Anything you upload will be added to your personal vector store.</div>
        </form>
      </div>

      <div class="panel">
        <div style="font-weight:900;color:#0f172a;margin-bottom:12px;">Files currently connected</div>

        {% if files and files|length > 0 %}
          <table>
            <thead>
              <tr>
                <th>Filename</th>
                <th>OpenAI file_id</th>
                <th>Uploaded</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {% for f in files %}
              <tr>
                <td>{{ f.filename or '(unnamed)' }}</td>
                <td style="font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;font-size:12px;">
                  {{ f.file_id }}
                </td>
                <td>{{ f.uploaded_at.strftime('%Y-%m-%d %H:%M') if f.uploaded_at else '' }}</td>
                <td>
                  <form method="post" style="margin:0;">
                    <input type="hidden" name="file_id" value="{{ f.file_id }}">
                    <button type="submit" name="action" value="delete">Delete</button>
                  </form>
                </td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        {% else %}
          <div class="small">No files yet. Upload something above.</div>
        {% endif %}
      </div>

    </div>
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
            width: 450px;
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
            max-width: 1500px;
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
            font-weight: 700;
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

        .message-row.user {
            justify-content: flex-end;
        }

        .message-row.assistant {
            justify-content: flex-start;
        }

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
            resize: none;
            min-height: 52px;
            max-height: 180px;
            padding: 12px 130px 12px 44px;
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.55);
            font-family: inherit;
            font-size: 14px;
            background: rgba(255,255,255,0.98);
            box-shadow: 0 0 0 1px rgba(148, 163, 184, 0.18);
            outline: none;
        }

        textarea[name="message"]:focus {
            border-color: rgba(29, 78, 216, 0.65);
            box-shadow: 0 0 0 3px rgba(29, 78, 216, 0.18);
        }

        .wink-mic-icon {
            position: absolute;
            left: 10px;
            bottom: 11px;
            font-size: 16px;
            color: var(--wink-blue);
            background: rgba(219, 234, 254, 0.92);
            border-radius: 999px;
            padding: 2px 6px;
            box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25);
            user-select: none;
        }

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

        .wink-attach-button:hover {
            filter: brightness(0.98);
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

        .wink-file-button:hover {
            filter: brightness(0.98);
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

        .wink-send-button:hover {
            filter: brightness(1.03);
        }

        .wink-hidden-file-input {
            display: none;
        }

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
                <div class="wink-header-pill">
                    /wink/{{ instructor.slug }}
                </div>
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
                <form class="wink-input-form" id="wink-form" method="post" enctype="multipart/form-data">
                    <div class="wink-textarea-wrapper">
                        <span class="wink-mic-icon" title="Microphone">🎤</span>
                        <textarea
                            id="wink-textarea"
                            name="message"
                            placeholder="Type your question here..."
                            autocomplete="off"
                        ></textarea>
                        <div class="wink-file-controls">
                            <button
                                type="button"
                                class="wink-attach-button"
                                onclick="document.getElementById('file-input').click();"
                                title="Attach a file"
                            >+</button>
                            <button
                                type="button"
                                class="wink-file-button"
                                onclick="document.getElementById('file-input').click();"
                            >
                                Choose File
                            </button>
                        </div>
                        <input
                            id="file-input"
                            class="wink-hidden-file-input"
                            type="file"
                            name="attachments"
                            multiple
                        />
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
      selectedFilesDiv.textContent = files.length ? ("Files selected: " + files.map(f => f.name).join(", ")) : "";
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
</script>
</body>
</html>
"""

def build_default_left_column_html(display_name):
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
        <div style="font-size:16px;font-weight:1000;text-transform:uppercase;letter-spacing:0.16em;opacity:0.96;">
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
      <p style="margin:0 0 14px;font-size:13px;color:#e5e7eb;opacity:0.92;">
        What I Need to Know, right when I need to know it!
      </p>

      <form method="post" style="margin-bottom:18px;">
        <input type="hidden" name="reset" value="1">
        <button
          type="submit"
          style="
            padding:7px 12px;
            border-radius:999px;
            border:1px solid rgba(148,163,184,0.7);
            background:radial-gradient(circle at top left,rgba(15,23,42,0.96),rgba(15,23,42,0.9));
            color:#e5e7eb;
            font-size:12px;
            cursor:pointer;
            display:inline-flex;
            align-items:center;
            gap:6px;
          "
        >
          <span style="
            display:inline-block;
            width:6px;
            height:6px;
            border-radius:999px;
            background:#f97316;
          "></span>
          Clear chat history
        </button>
      </form>

      <div style="
        display:flex;
        flex-wrap:wrap;
        gap:6px;
        justify-content:center;
        margin-bottom:14px;
      ">
        <span style="
          display:inline-block;
          padding:4px 10px;
          border-radius:999px;
          background:rgba(219,234,254,0.18);
          color:#bfdbfe;
          font-size:11px;
          text-transform:uppercase;
          letter-spacing:0.12em;
        ">Wink Users Guide</span>
      </div>

      <div style="
        font-size:12px;
        color:#e5e7eb;
        padding:10px 12px;
        border-radius:12px;
        background:radial-gradient(circle at top left,rgba(15,23,42,0.98),rgba(15,23,42,0.9));
        border:1px dashed rgba(148,163,184,0.8);
        text-align:left;
        display:inline-block;
        max-width:260px;
      ">
        <div style="font-weight:600;color:#f9fafb;margin-bottom:4px;font-size:12px;">
          Try asking WINK:
        </div>
        <div style="margin-bottom:2px;">• “What is due this week in this class?”</div>
        <div style="margin-bottom:2px;">• “Help me draft an email to my instructor.”</div>
        <div>• “Where can I get tutoring at UTEP?”</div>
      </div>
    </div>
    """


def sanitize_left_column_html(html: str) -> str:
    if not html:
        return html

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
        span_pat = re.compile(
            r"<span\b[^>]*>\s*" + re.escape(label) + r"\s*</span>", re.IGNORECASE
        )
        html = span_pat.sub("", html)

        btn_pat = re.compile(
            r"<button\b[^>]*>\s*" + re.escape(label) + r"\s*</button>", re.IGNORECASE
        )
        html = btn_pat.sub("", html)

        a_pat = re.compile(r"<a\b[^>]*>\s*" + re.escape(label) + r"\s*</a>", re.IGNORECASE)
        html = a_pat.sub("", html)

    html = re.sub(r"('s)\s+course\b", r"\1 Course", html, flags=re.IGNORECASE)

    html = re.sub(r"<div\b[^>]*>\s*</div>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"\n{3,}", "\n\n", html)

    return html



def ask_with_optional_vector_store(chat_input, vector_store_ids=None):
    tools = None

    if vector_store_ids:
        if isinstance(vector_store_ids, str):
            vector_store_ids = [vector_store_ids]

        tools = [
            {
                "type": "file_search",
                "vector_store_ids": vector_store_ids
            }
        ]

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=chat_input,
        tools=tools
    )

    return response.output_text.strip()


















# ============================================
# Routes
# ============================================


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        if not email:
            flash("Please enter a valid email address.")
            return render_template_string(TEMPLATE_LOGIN_PAGE)

        instructor = Instructor.query.filter(
            db.func.lower(Instructor.email) == email
        ).first()

        if instructor:
            return redirect(url_for("manage_files", instructor_id=instructor.id))

        return redirect(url_for("new_instructor", email=email))

    return render_template_string(TEMPLATE_LOGIN_PAGE)



@app.route("/admin/new_instructor", methods=["GET", "POST"])
def new_instructor():
    if request.method == "GET":
        email = request.args.get("email", "").strip().lower()
        return render_template_string(TEMPLATE_NEW_INSTRUCTOR, email=email)

    email = request.form.get("email", "").strip().lower()
    name = request.form.get("name", "").strip()

    if not email or not name:
        flash("Please complete both email and name.")
        return render_template_string(TEMPLATE_NEW_INSTRUCTOR, email=email)

    existing = Instructor.query.filter_by(email=email).first()
    if existing:
        return redirect(url_for("manage_files", instructor_id=existing.id))

    slug = re.sub(r"[^a-z0-9]+", "", email.split("@")[0])

    if Instructor.query.filter_by(slug=slug).first():
        suffix = re.sub(r"[^a-z0-9]+", "", str(int(datetime.utcnow().timestamp())))
        slug = f"{slug}{suffix}"

    try:
        vs_id = create_vector_store_http(f"WINK - {name}")
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
def manage_files(instructor_id):
    instructor = Instructor.query.get_or_404(instructor_id)

    if request.method == "POST":
        action = request.form.get("action", "").strip().lower()

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
                local_path = os.path.join(UPLOAD_DIR, f"{int(datetime.utcnow().timestamp())}_{safe_name}")
                f.save(local_path)

                try:
                    file_id = upload_openai_file_http(local_path, safe_name)
                    add_file_to_vector_store_http(instructor.personal_vector_store_id, file_id)

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
            file_id = request.form.get("file_id", "").strip()
            if not file_id:
                flash("Missing file_id.", "bad")
                return redirect(url_for("manage_files", instructor_id=instructor.id))

            rec = InstructorFile.query.filter_by(instructor_id=instructor.id, file_id=file_id).first()
            try:
                if instructor.personal_vector_store_id:
                    delete_file_from_vector_store_http(instructor.personal_vector_store_id, file_id)
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

    files = InstructorFile.query.filter_by(instructor_id=instructor.id).order_by(InstructorFile.uploaded_at.desc()).all()
    return render_template_string(TEMPLATE_MANAGE_FILES, instructor=instructor, files=files)




@app.route("/wink/<slug>", methods=["GET", "POST"])
def wink_chat(slug):
    instructor = Instructor.query.filter_by(slug=slug).first_or_404()

    left_column_html = build_default_left_column_html(
        instructor.name or instructor.email
    )
    left_column_html = sanitize_left_column_html(left_column_html)

    session_key = f"wink_messages_{instructor.id}"
    messages = session.get(session_key, [])

    if request.method == "POST":

        if request.form.get("reset"):
            session.pop(session_key, None)
            messages = []

        else:
            user_message = request.form.get("message", "").strip()
            if user_message:
                messages.append({
                    "role": "user",
                    "text": user_message
                })

                system_prompt = {
                    "role": "system",
                    "content": (
                        "You are WINK (What I Need to Know). "
                        "Answer using the instructor syllabus and UTEP resources when available. "
                        "If the answer is not found there, use general knowledge. "
                        "Be clear, student-friendly, and accurate."
                    ),
                }

                chat_input = [
                    system_prompt,
                    *[
                        {"role": m["role"], "content": m["text"]}
                        for m in messages
                    ]
                ]

                assistant_text = ""


                vector_store_ids = []

                if instructor.personal_vector_store_id:
                    vector_store_ids.append(instructor.personal_vector_store_id)





                if COMMON_VECTOR_STORE_ID:
                    vector_store_ids.append(COMMON_VECTOR_STORE_ID)

                assistant_text = ask_with_optional_vector_store(
                    chat_input,
                    vector_store_ids if vector_store_ids else None
                )



              




  
                # 3. Plain ChatGPT fallback
                if not assistant_text.strip():
                    assistant_text = ask_with_optional_vector_store(chat_input)

                messages.append({
                    "role": "assistant",
                    "text": assistant_text
                })

                session[session_key] = messages

    return render_template_string(
        TEMPLATE_WINK_CHAT,
        instructor=instructor,
        left_column_html=left_column_html,
        messages=messages
    )















   


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
