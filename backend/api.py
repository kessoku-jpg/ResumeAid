"""
REST API
--------
Wraps pipeline.py in a Flask HTTP server so the Android app can
actually call it, instead of only being runnable from the command
line.

ONE ENDPOINT does the real work:
    POST /api/analyze
        multipart/form-data with:
          - resume: the PDF/DOCX file (required)
          - query: custom search text (optional â€” auto-derived from
            the resume's skills if omitted, same as pipeline.py)
          - country: Adzuna country code (optional, default "us")
          - source: "all" | "philjobnet" | "adzuna" | "onlinejobs"
            (optional, default "all")

    Returns JSON:
        {
          "resume_summary": { name, email, phone, skills, education, experience },
          "health_check": { issues, suggestions, suggested_skills },
          "search_info": { query, confidence, source },
          "matches": [
            { title, company, location, url, match_score, relative_score, comment },
            ...
          ]
        }

Also:
    GET /api/health   -> {"status": "ok"}   (simple connectivity check)

NOTE ON RESPONSE SHAPE vs. the Android project's existing Job.java /
Resume.java POJOs: those were written earlier as UI-only placeholders
(before the real backend existed) and include fields like "TF-IDF
score" and "tags" that this backend doesn't actually produce. The
fields above are what the real pipeline actually computes â€” the
Android models will need updating to match these real field names
(match_score, relative_score, comment) rather than the placeholder
ones.

Install dependencies (on top of everything pipeline.py already needs):
    pip install flask

Run it:
    python api.py
Then it's reachable at http://<your-computer's-LAN-IP>:5000 from a
phone on the same network (use your Android emulator's special host
alias 10.0.2.2 instead of localhost/127.0.0.1 if testing on an
emulator rather than a real device).

NOTE ON MODEL LOADING: importing pipeline.py (which imports
matching_engine.py) loads the Sentence-BERT model once, at server
startup â€” not per-request. That's why the first request after
starting the server is instant, not slow.
"""

import os
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS

from pipeline import run_pipeline

app = Flask(__name__)
CORS(app)  # Allow requests from browser / file://

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB â€” generous for a resume file
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
        return jsonify({"error": "Empty filename â€” no file was actually selected."}), 400

    _, extension = os.path.splitext(uploaded_file.filename.lower())
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type '{extension}'. Only PDF and DOCX are accepted."}), 400

    query = request.form.get("query") or None
    country = request.form.get("country", "us")
    source = request.form.get("source", "all")

    # Save the upload to a real temp file â€” resume_parser.py's
    # extractors need an actual file path on disk, not an in-memory
    # stream, since pdfplumber/python-docx both expect a path or
    # file-like object opened in a specific way.
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
        # Catch-all so a parsing/matching failure returns a clean JSON
        # error instead of Flask's default HTML error page â€” an
        # Android app expects JSON back, not HTML.
        return jsonify({"error": f"Failed to process resume: {e}"}), 500

    finally:
        # Clean up the temp file/dir regardless of success or failure â€”
        # this is a stateless API, nothing should linger on disk.
        try:
            os.remove(temp_path)
            os.rmdir(temp_dir)
        except OSError:
            pass


if __name__ == "__main__":
    # host="0.0.0.0" makes this reachable from other devices on the
    # same network (like a phone), not just this computer itself.
    app.run(host="0.0.0.0", port=5000, debug=True)


