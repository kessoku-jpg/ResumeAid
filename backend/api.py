"""
ResumeAid REST API.
Exposes the pipeline via a Flask HTTP server for client apps.

Endpoints:
  POST /api/analyze: Takes a resume (PDF/DOCX) and optional query, country, source.
                     Returns parsed resume, health check, and matched jobs.
  GET /api/health: Returns {"status": "ok"}.
"""

import os
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS

from pipeline import run_pipeline

app = Flask(__name__)
CORS(app)  # Allow requests from browser / file://

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB maximum for resume files
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_BYTES


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "No 'resume' file included in the request."}), 400

    uploaded_file = request.files["resume"]
    if uploaded_file.filename == "":
        return jsonify({"error": "Empty filename - no file was actually selected."}), 400

    _, extension = os.path.splitext(uploaded_file.filename.lower())
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type '{extension}'. Only PDF and DOCX are accepted."}), 400

    query = request.form.get("query") or None
    country = request.form.get("country", "ph")
    source = request.form.get("source", "all")

    # Save uploaded file to temp path so extractors (pdfplumber/docx) can read it
    temp_dir = tempfile.mkdtemp(prefix="resumeaid_")
    temp_path = os.path.join(temp_dir, uploaded_file.filename)

    try:
        uploaded_file.save(temp_path)

        result = run_pipeline(
            temp_path,
            search_query=query,
            country=country,
            source=source,
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

    except Exception as e:
        # Catch-all to ensure a clean JSON error response for clients
        return jsonify({"error": f"Failed to process resume: {e}"}), 500

    finally:
        # Ensure temp file cleanup to keep API stateless
        try:
            os.remove(temp_path)
            os.rmdir(temp_dir)
        except OSError:
            pass


if __name__ == "__main__":
    # Bind to 0.0.0.0 to allow network access for testing
    app.run(host="0.0.0.0", port=5000, debug=True)


