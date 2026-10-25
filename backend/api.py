"""
ResumeAid REST API.
Exposes the pipeline via a Flask HTTP server for client apps.

Endpoints:
  POST /api/analyze: Takes a resume (PDF/DOCX) and optional query, country, source.
                     Returns parsed resume, health check, and matched jobs.
  GET /api/health: Returns {"status": "ok"}.

Configuration (environment variables, optionally loaded from backend/.env):
  ADZUNA_APP_ID / ADZUNA_APP_KEY  Adzuna API credentials (optional source)
  CORS_ORIGINS                    Comma-separated allowed origins (default: *)
  FLASK_DEBUG                     "1" to enable debug mode (never in production)
  HOST / PORT                     Bind address (default: 127.0.0.1:5000)
"""

import logging
import os
import shutil
import tempfile

# Load .env before importing modules that read environment variables.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

from pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("resumeaid.api")

app = Flask(__name__)

_cors_origins = os.environ.get("CORS_ORIGINS", "*")
CORS(app, origins=[o.strip() for o in _cors_origins.split(",")] if _cors_origins != "*" else "*")

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
ALLOWED_SOURCES = {"all", "both", "philjobnet", "adzuna", "jsearch", "onlinejobs"}
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB maximum for resume files
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_BYTES

# File signatures ("magic bytes") — guards against renamed non-resume files.
FILE_SIGNATURES = {
    ".pdf": b"%PDF",
    ".docx": b"PK\x03\x04",  # DOCX is a ZIP container
}


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_e):
    return jsonify({"error": "File is too large (max 10 MB)."}), 413


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "No 'resume' file included in the request."}), 400

    uploaded_file = request.files["resume"]
    if not uploaded_file.filename:
        return jsonify({"error": "Empty filename - no file was actually selected."}), 400

    _, extension = os.path.splitext(uploaded_file.filename.lower())
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type '{extension}'. Only PDF and DOCX are accepted."}), 400

    # Verify the file content actually matches its extension
    header = uploaded_file.stream.read(8)
    uploaded_file.stream.seek(0)
    if not header.startswith(FILE_SIGNATURES[extension]):
        return jsonify({"error": "The file content doesn't look like a valid PDF or DOCX."}), 400

    query = (request.form.get("query") or "").strip() or None
    country = (request.form.get("country") or "ph").strip().lower()[:2]
    source = (request.form.get("source") or "all").strip().lower()
    force_refresh = request.form.get("refresh", "").lower() in ("1", "true", "yes")
    if source not in ALLOWED_SOURCES:
        return jsonify({"error": f"Unknown source '{source}'."}), 400

    # Save under a fixed server-chosen name — never trust the client filename
    # (it could contain path separators like "../").
    temp_dir = tempfile.mkdtemp(prefix="resumeaid_")
    temp_path = os.path.join(temp_dir, f"resume{extension}")

    try:
        uploaded_file.save(temp_path)

        result = run_pipeline(
            temp_path,
            search_query=query,
            country=country,
            source=source,
            force_refresh=force_refresh,
        )

        parsed_resume = result["parsed_resume"]
        resume_summary = {
            "name": parsed_resume.get("name"),
            "email": parsed_resume.get("email"),
            "phone": parsed_resume.get("phone"),
            "skills": parsed_resume.get("skills", []),
            "education": parsed_resume.get("education"),
            "experience": parsed_resume.get("experience"),
        }

        matches = [
            {
                "title": job.get("title"),
                "company": job.get("company"),
                "location": job.get("location"),
                "url": job.get("url"),
                "match_score": job.get("match_score"),
                "relative_score": job.get("relative_score"),
                "comment": job.get("comment"),
                "breakdown": job.get("breakdown"),
            }
            for job in result["ranked_results"]
        ]

        return jsonify({
            "resume_summary": resume_summary,
            "health_check": result["health_check"],
            "search_info": result["search_info"],
            "matches": matches,
        })

    except Exception:
        # Log full details server-side; don't leak internals to the client.
        logger.exception("Failed to process resume")
        return jsonify({"error": "We couldn't process this resume. Please check the file and try again."}), 500

    finally:
        # Ensure temp file cleanup to keep API stateless
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG") == "1"
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    # Never run debug mode on a public interface: the Werkzeug debugger allows code execution.
    if debug and host not in ("127.0.0.1", "localhost"):
        logger.warning("Debug mode requested on non-local host %s; disabling debug.", host)
        debug = False
    app.run(host=host, port=port, debug=debug)
