"""
Connects resume parsing and job matching into a single flow:
resume -> parse_resume() -> rank_jobs_for_resume() -> matches.

Defaults to fetching from all scrapers (PhilJobNet, Adzuna, OnlineJobs)
and automatically derives search queries from resume skills.
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


# Fallback placeholder listings if APIs fail or return nothing.
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
    Auto-derives search query using semantic occupation inference, 
    falling back to joining top skills as keywords.
    Returns (query_string, confidence_score).
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
    """Fetches PhilJobNet listings, returning [] on failure."""
    if not PHILJOBNET_AVAILABLE:
        print("philjobnet_scraper.py not found or 'beautifulsoup4' not installed — skipping PhilJobNet")
        return []
    try:
        return search_philjobnet(query, max_results=max_results)
    except Exception as e:
        print(f"Couldn't fetch live PhilJobNet listings ({e})")
        return []


def _try_fetch_adzuna(query, country, results_per_page):
    """Fetches Adzuna API listings, returning [] on failure."""
    if not JSEARCH_AVAILABLE:
        print("job_scraper.py not found or 'requests' not installed — skipping Adzuna")
        return []
    try:
        return search_jsearch(query, country=country, results_per_page=results_per_page)
    except Exception as e:
        print(f"Couldn't fetch live Adzuna listings ({e})")
        return []


def _try_fetch_onlinejobs(query, max_results):
    """Fetches OnlineJobs.ph listings, returning [] on failure."""
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
    Returns aggregated job listings from the requested sources.
    Defaults to returning results from all available scrapers.
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
    Full pipeline: parses resume and ranks it against job listings.
    Automatically derives a search query if none is provided.
    """
    print(f"Parsing resume: {resume_filepath}")
    parsed_resume = parse_resume(resume_filepath)

    print("\n--- Parsed resume summary ---")
    print(f"Name: {parsed_resume.get('name')}")
    print(f"Email: {parsed_resume.get('email')}")
    print(f"Detected skills: {parsed_resume.get('skills')}")

    # Run diagnostics and temporarily apply suggested skills for matching
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
    """Minimal CLI argument parser."""
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
