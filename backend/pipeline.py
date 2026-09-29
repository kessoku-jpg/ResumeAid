"""
Pipeline
--------
Connects resume_parser.py and matching_engine.py into one real flow:

    resume file (PDF/DOCX)  -->  parse_resume()  -->  parsed dict
                                                            |
                                                            v
    job description(s)     -->  rank_jobs_for_resume()  -->  ranked matches

Job listings come from ALL THREE scrapers by default — philjobnet_scraper.py
(PH postings), job_scraper.py (Adzuna API, international/remote), and
onlinejobs_scraper.py (PH remote/VA postings) — 10 results from each, merged
into one combined list, then ranked together by match percentage, so the
final results freely mix all three sources based on actual fit rather than
which source they came from. Use --source philjobnet, --source adzuna, or
--source onlinejobs to fetch from just one instead.

The search query is derived AUTOMATICALLY from the resume's own
detected skills — no need to type it manually. You can still override
it with an explicit query if you want a specific search instead.
Falls back to a few hardcoded placeholder listings if none of the
sources are reachable, so this still runs at every stage of
development.

Before matching, resume_improver.py's diagnostics run automatically:
if the resume has no dedicated Skills section, real O*NET-vocabulary
matches found in the resume's body text get auto-applied as
explicit_skills for this run (the original resume file is never
modified) — this is what keeps a resume with no Skills section from
degrading match quality the way it used to.

Usage:
    python pipeline.py path/to/resume.pdf
    python pipeline.py path/to/resume.pdf --query "cashier"
    python pipeline.py path/to/resume.pdf --source adzuna --query "python developer" --country gb

Make sure resume_parser.py, matching_engine.py, philjobnet_scraper.py,
job_scraper.py, onlinejobs_scraper.py, and resume_improver.py are all
in the same folder as this file.

Install dependencies first (covers every file's requirements):
    pip install pdfplumber python-docx pytesseract pdf2image spacy sentence-transformers requests beautifulsoup4
    python -m spacy download en_core_web_sm
"""

import sys

from resume_parser import parse_resume
from matching_engine import rank_jobs_for_resume, infer_job_title

try:
    from philjobnet_scraper import search_jobs as search_philjobnet
    PHILJOBNET_AVAILABLE = True
except ImportError:
    PHILJOBNET_AVAILABLE = False

try:
    from job_scraper import search_jobs as search_jsearch
    JSEARCH_AVAILABLE = True
except ImportError:
    JSEARCH_AVAILABLE = False

try:
    from onlinejobs_scraper import search_jobs as search_onlinejobs
    ONLINEJOBS_AVAILABLE = True
except ImportError:
    ONLINEJOBS_AVAILABLE = False

try:
    from resume_improver import analyze_parsed_resume, print_report
    IMPROVER_AVAILABLE = True
except ImportError:
    IMPROVER_AVAILABLE = False


# Fallback placeholder listings — used only when no search query
# could be derived (e.g. no skills detected at all), or when the real
# job_scraper.py call fails (e.g. RapidAPI key isn't set up
# yet). Keeps this runnable at every stage of development instead of
# hard-failing.
SAMPLE_JOB_LISTINGS = [
    {
        "title": "Frontend Developer",
        "company": "Example Tech Co.",
        "description": (
            "Looking for a front-end developer experienced in React and "
            "modern JavaScript frameworks. Comfortable working in an "
            "Agile team environment with sprint planning."
        ),
        "url": "",
    },
    {
        "title": "Data Entry Clerk",
        "company": "Example Admin Services",
        "description": (
            "Seeking a detail-oriented data entry clerk for manual "
            "spreadsheet work. No programming experience required."
        ),
        "url": "",
    },
    {
        "title": "Backend / Full-Stack Developer",
        "company": "Example Software Inc.",
        "description": (
            "Full-stack role requiring Python backend experience, SQL "
            "databases, and containerization with Docker. Bonus if "
            "you've used REST APIs."
        ),
        "url": "",
    },
]


def build_search_query_from_resume(parsed_resume, max_skills=4):
    """
    Auto-derives a search query from the resume itself — this
    is what removes the need to type a query manually.

    Tries TWO approaches, in order:
      1. Semantic occupation inference (infer_job_title, in
         matching_engine.py) — embeds the resume's skills/experience
         and finds the closest-matching real O*NET occupation title
         (e.g. "Software Developers"). This produces an actual job
         title, which is what a search engine expects, rather than a
         raw keyword dump.
      2. Fallback: if that's not confident enough (or
         occupation_profiles.json isn't set up), just joins the
         resume's top listed skills as keywords instead (e.g. "Python
         React SQL Docker") — cruder, but still automatic.

    Returns (query_string, confidence_or_None) — confidence is the
    inference similarity score when approach 1 was used, or None when
    the skill-join fallback was used instead (there's no comparable
    score for that). Returns (None, None) if neither approach found
    anything usable.
    """
    inferred = infer_job_title(parsed_resume)
    if inferred:
        title, score = inferred
        return title, score

    skills = parsed_resume.get("explicit_skills") or parsed_resume.get("skills") or []
    top_skills = [s for s in skills if s][:max_skills]
    if not top_skills:
        return None, None
    return " ".join(top_skills), None


def _try_fetch_philjobnet(query, max_results):
    """Fetches from PhilJobNet, returning [] (with a printed reason) on any failure — never raises."""
    if not PHILJOBNET_AVAILABLE:
        print("philjobnet_scraper.py not found or 'beautifulsoup4' not installed — skipping PhilJobNet")
        return []
    try:
        return search_philjobnet(query, max_results=max_results)
    except Exception as e:
        print(f"Couldn't fetch live PhilJobNet listings ({e})")
        return []


def _try_fetch_adzuna(query, country, results_per_page):
    """Fetches from Adzuna API, returning [] (with a printed reason) on any failure — never raises.
    Note: Adzuna does not support Philippines (ph); job_scraper.py automatically falls back to
    'gb' (UK) for international/remote results when ph is passed.
    """
    if not JSEARCH_AVAILABLE:
        print("job_scraper.py not found or 'requests' not installed — skipping Adzuna")
        return []
    try:
        return search_jsearch(query, country=country, results_per_page=results_per_page)
    except Exception as e:
        print(f"Couldn't fetch live Adzuna listings ({e})")
        return []


def _try_fetch_onlinejobs(query, max_results):
    """Fetches from OnlineJobs.ph, returning [] (with a printed reason) on any failure — never raises."""
    if not ONLINEJOBS_AVAILABLE:
        print("onlinejobs_scraper.py not found or 'beautifulsoup4' not installed — skipping OnlineJobs.ph")
        return []
    try:
        return search_onlinejobs(query, max_results=max_results)
    except Exception as e:
        print(f"Couldn't fetch live OnlineJobs.ph listings ({e})")
        return []


def get_job_listings(search_query=None, country="ph", source="all", results_per_source=10):
    """
    Returns real job listings from the requested source:
      - "all" / "both" (default): fetches results_per_source from EACH
        of PhilJobNet, Adzuna, and OnlineJobs.ph, MERGES them into one
        combined list. rank_jobs_for_resume then scores and ranks the
        whole merged set together by match percentage — so the final
        ranking can freely mix all three sources based on actual fit,
        not on which source they came from.
          • PhilJobNet  — PH government job board (PH-specific)
          • Adzuna      — international/remote jobs (ph falls back to gb)
          • OnlineJobs.ph — PH remote/VA postings (PH-specific)
      - "philjobnet": PhilJobNet only
      - "adzuna" / "jsearch": Adzuna only (international/remote)
      - "onlinejobs": OnlineJobs.ph only

    Each source fails independently — if Adzuna credentials aren't
    set up, or one scraper's request fails, that source just
    contributes zero results (with a printed reason) rather than
    breaking the whole run. Falls back to placeholder listings only if
    NO source returns anything at all (or the query is empty).
    """
    if not search_query:
        return SAMPLE_JOB_LISTINGS

    if source in ("all", "both"):
        philjobnet_jobs = _try_fetch_philjobnet(search_query, results_per_source)
        adzuna_jobs = _try_fetch_adzuna(search_query, country, results_per_source)
        onlinejobs_jobs = _try_fetch_onlinejobs(search_query, results_per_source)
        jobs = philjobnet_jobs + adzuna_jobs + onlinejobs_jobs
        if jobs:
            print(f"Combined {len(philjobnet_jobs)} PhilJobNet + {len(adzuna_jobs)} Adzuna + "
                  f"{len(onlinejobs_jobs)} OnlineJobs.ph listings ({len(jobs)} total)")
    elif source == "onlinejobs":
        jobs = _try_fetch_onlinejobs(search_query, results_per_source)
    elif source in ("adzuna", "jsearch"):
        # "jsearch" kept as alias so existing Android clients don't break
        jobs = _try_fetch_adzuna(search_query, country, results_per_source)
    else:
        jobs = _try_fetch_philjobnet(search_query, results_per_source)

    if not jobs:
        print(f"No live results for '{search_query}' — using placeholder listings")
        return SAMPLE_JOB_LISTINGS
    return jobs


def run_pipeline(resume_filepath, job_listings=None, search_query=None, country="us", source="all"):
    """
    Full flow: parse the resume file, then rank it against a list of
    job listings.

    If job_listings is given directly, uses that. Otherwise, if
    search_query is given, searches the chosen `source` with that
    exact query. Otherwise (the default case), AUTOMATICALLY builds a
    search query from the resume's own detected skills — no manual
    typing needed.
    """
    print(f"Parsing resume: {resume_filepath}")
    parsed_resume = parse_resume(resume_filepath)

    print("\n--- Parsed resume summary ---")
    print(f"Name: {parsed_resume.get('name')}")
    print(f"Email: {parsed_resume.get('email')}")
    print(f"Detected skills: {parsed_resume.get('skills')}")

    # Run resume diagnostics BEFORE matching, and auto-apply suggested
    # skills into explicit_skills if the resume has none — this is
    # what makes the "no Skills section" problem stop hurting match
    # quality for THIS run, without requiring the person to go edit
    # and re-upload their actual resume file first. The original file
    # on disk is never touched — this only affects parsed_resume in
    # memory for this pipeline run.
    health_check = {"issues": [], "suggestions": [], "suggested_skills": []}
    if IMPROVER_AVAILABLE:
        issues, suggestions, suggested_skills = analyze_parsed_resume(parsed_resume, resume_filepath)
        health_check = {"issues": issues, "suggestions": suggestions, "suggested_skills": suggested_skills}
        print()
        print_report(resume_filepath, issues, suggestions, suggested_skills)

        if not parsed_resume.get("explicit_skills") and suggested_skills:
            parsed_resume["explicit_skills"] = suggested_skills
            # Keep the display "skills" list consistent too, without duplicating
            existing_lower = {s.lower() for s in parsed_resume.get("skills", [])}
            for skill in suggested_skills:
                if skill.lower() not in existing_lower:
                    parsed_resume.setdefault("skills", []).append(skill)
            print(f"Auto-applied {len(suggested_skills)} suggested skill(s) for this matching run "
                  f"(your resume file itself is unchanged).\n")

    search_info = {"query": search_query, "confidence": None, "source": source}
    if job_listings is None:
        query = search_query
        confidence = None
        if query is None:
            query, confidence = build_search_query_from_resume(parsed_resume)
            if query and confidence is not None:
                print(f'\nInferred likely occupation from resume: "{query}" (confidence: {confidence:.0%})')
            elif query:
                print(f'\nAuto-detected search query from resume\'s skills: "{query}"')
            else:
                print("\nNo usable skills detected to build a search query from")
        search_info = {"query": query, "confidence": confidence, "source": source}
        job_listings = get_job_listings(query, country, source)

    print(f"\nMatching against {len(job_listings)} job listings...")
    ranked_results = rank_jobs_for_resume(parsed_resume, job_listings)

    print("\n--- Ranked matches ---")
    for job in ranked_results:
        print(f"\n{job['match_score']}% raw / {job['relative_score']}% relative  —  {job.get('title', 'Untitled listing')}"
              f"{' at ' + job['company'] if job.get('company') else ''}")
        print(f"   {job['comment']}")
        if job.get("url"):
            print(f"   {job['url']}")

    return {
        "parsed_resume": parsed_resume,
        "health_check": health_check,
        "search_info": search_info,
        "ranked_results": ranked_results,
    }


def _parse_args(argv):
    """Minimal argument parser: positional resume path, optional --query, --country, --source flags."""
    if len(argv) < 1:
        return None
    resume_path = argv[0]
    query = None
    country = "us"
    source = "all"

    i = 1
    while i < len(argv):
        if argv[i] == "--query" and i + 1 < len(argv):
            query = argv[i + 1]
            i += 2
        elif argv[i] == "--country" and i + 1 < len(argv):
            country = argv[i + 1]
            i += 2
        elif argv[i] == "--source" and i + 1 < len(argv):
            source = argv[i + 1]
            i += 2
        else:
            i += 1

    return resume_path, query, country, source


if __name__ == "__main__":
    parsed_args = _parse_args(sys.argv[1:])
    if parsed_args is None:
        print("Usage: python pipeline.py <path_to_resume.pdf_or_docx> [--query \"custom search\"] [--source all|philjobnet|adzuna|onlinejobs] [--country gb]")
        print("Without --query, the search term is auto-derived from the resume's own detected skills.")
        print("Default source is 'all' (PhilJobNet + Adzuna + OnlineJobs.ph, merged and ranked together).")
        print("--country applies to the Adzuna source; 'ph' is not supported by Adzuna and auto-falls back to 'gb'.")
        sys.exit(1)

    resume_path, query, country_code, source = parsed_args
    run_pipeline(resume_path, search_query=query, country=country_code, source=source)
