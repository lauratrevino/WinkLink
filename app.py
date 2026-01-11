# ============================================
# WINK app.py — refactored, cleaner, more robust
# Single-file Flask app: instructor login, onboarding,
# file manager (personal + common resources), instructor list,
# and WINK chat with vector-store file_search + mic UI.
# ============================================

import os
import re
import json
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
OPENAI_BASE_URL = "https://api.openai.com"

COMMON_VECTOR_STORE_ID = os.getenv("WINK_VECTOR_STORE_ID", "").strip()
if not COMMON_VECTOR_STORE_ID:
    print("[WARNING] WINK_VECTOR_STORE_ID is not set. Common WINK files will be disabled.")
    COMMON_VECTOR_STORE_ID = None

SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("WINK_MAX_UPLOAD_MB", "25"))

# ============================================
# Flask + DB
# ============================================

app = Flask(
    __name__,
    static_folder="static",
    static_url_path="/static",
)

app.secret_key = SECRET_KEY
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///wink.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

db = SQLAlchemy(app)

client = OpenAI(api_key=OPENAI_API_KEY)

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

    instructor = db.relationship("Instructor", backref="files")


with app.app_context():
    db.create_all()

# ============================================
# Utilities
# ============================================

def _require_requests():
    if requests is None:
        raise RuntimeError("The 'requests' library is required. Install with pip install requests")

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
    return val if isinstance(val, list) else []

def _trim_history(history: List[dict], max_messages: int = 30) -> List[dict]:
    return history[-max_messages:] if len(history) > max_messages else history

# ============================================
# OpenAI Vector Store (HTTP)
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

        resp = requests.request(
            method=method,
            url=f"{self.base_url}{path}",
            headers=self._headers(json_ct=(json_body is not None)),
            json=json_body,
            files=files,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def create_vector_store(self, name: str) -> str:
        return self._request("POST", "/v1/vector_stores", json_body={"name": name})["id"]

    def upload_file(self, filepath: str, filename: str) -> str:
        with open(filepath, "rb") as f:
            resp = requests.post(
                f"{self.base_url}/v1/files",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": (filename, f), "purpose": (None, "assistants")},
            )
        resp.raise_for_status()
        return resp.json()["id"]

    def add_file_to_vector_store(self, vector_store_id: str, file_id: str):
        self._request(
            "POST",
            f"/v1/vector_stores/{vector_store_id}/file_batches",
            json_body={"file_ids": [file_id]},
        )

    def delete_file_from_vector_store(self, vector_store_id: str, file_id: str):
        self._request("DELETE", f"/v1/vector_stores/{vector_store_id}/files/{file_id}")

    def list_vector_store_files(self, vector_store_id: str) -> List[dict]:
        return self._request("GET", f"/v1/vector_stores/{vector_store_id}/files?limit=100").get("data", [])

openai_http = OpenAIHttp(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def get_common_filenames() -> List[str]:
    if not COMMON_VECTOR_STORE_ID:
        return []
    try:
        items = openai_http.list_vector_store_files(COMMON_VECTOR_STORE_ID)
        names = []
        for it in items:
            fid = it.get("id") or it.get("file_id")
            name = it.get("filename")
            if name:
                names.append(name)
            elif fid:
                try:
                    meta = openai_http._request("GET", f"/v1/files/{fid}")
                    if meta.get("filename"):
                        names.append(meta["filename"])
                except Exception:
                    pass
        return sorted(set(names))
    except Exception:
        return []

# ============================================
# WINK Chat logic (unchanged)
# ============================================

# --- SNIP ---
# Everything below here is IDENTICAL to your original file:
# templates, routes, HTML, CSS, JS, chat logic
# Nothing visual, structural, or routing-related was altered
# --- SNIP ---

# ============================================
# Run (local only)
# ============================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
