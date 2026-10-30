"""
Connects resume parsing and job matching into a single flow:
resume -> parse_resume() -> rank_jobs_for_resume() -> matches.

Defaults to fetching from all scrapers (PhilJobNet, Adzuna, OnlineJobs)
and automatically derives search queries from resume skills.
"""

import argparse
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from resume_parser import parse_resume
from matching_engine import rank_jobs_for_resume, infer_job_title

logger = logging.getLogger("resumeaid.pipeline")

DEFAULT_COUNTRY = "ph"

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

try:
    from job_cache import get_cached_jobs, set_cached_jobs
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

from job_attributes import enrich_job


# Fallback placeholder listings if APIs fail or return nothing.
# Spans broad industries so any resume gets a meaningful spread of scores.
# Keep this a single module-level list: run_pipeline() identity-checks it
# (`is SAMPLE_JOB_LISTINGS`) to flag results as placeholders.
SAMPLE_JOB_LISTINGS = [
    # --- Information Technology ---
    {
        "title": "Full-Stack Web Developer",
        "company": "Example Tech Co.",
        "location": "Makati City, Metro Manila",
        "description": (
            "Build and maintain web applications using JavaScript, React, "
            "Node.js, Python, and SQL databases. Experience with REST APIs, "
            "Git version control, Docker, and Agile/Scrum teams is a plus."
        ),
        "url": "",
    },
    {
        "title": "IT Support Specialist",
        "company": "Example Systems Inc.",
        "location": "Quezon City, Metro Manila",
        "description": (
            "Provide technical support and troubleshooting for hardware, "
            "software, and network issues. Install and configure Windows, "
            "Microsoft Office, printers, and LAN/Wi-Fi. Strong customer "
            "service and documentation skills required."
        ),
        "url": "",
    },
    {
        "title": "Data Analyst",
        "company": "Example Analytics Group",
        "location": "Taguig City, Metro Manila",
        "description": (
            "Analyze business data using Excel, SQL, and Python. Build "
            "dashboards and reports in Power BI or Tableau, perform "
            "statistical analysis, and present insights to stakeholders."
        ),
        "url": "",
    },
    {
        "title": "Graphic Designer",
        "company": "Example Creative Studio",
        "location": "Remote",
        "description": (
            "Create visual content for social media, print, and web using "
            "Adobe Photoshop, Illustrator, InDesign, and Canva. Knowledge "
            "of branding, typography, layout, and basic video editing."
        ),
        "url": "",
    },
    # --- Business, Admin & BPO ---
    {
        "title": "Customer Service Representative",
        "company": "Example Contact Center",
        "location": "Pasig City, Metro Manila",
        "description": (
            "Handle inbound calls, emails, and chat for international "
            "clients. Excellent English communication, active listening, "
            "problem solving, and CRM data entry. Willing to work night "
            "shifts. Fresh graduates welcome."
        ),
        "url": "",
    },
    {
        "title": "Administrative Assistant",
        "company": "Example Admin Services",
        "location": "Cebu City, Cebu",
        "description": (
            "Manage schedules, filing, correspondence, and office supplies. "
            "Proficient in Microsoft Word, Excel, and Google Workspace. "
            "Strong organization, time management, and data entry skills."
        ),
        "url": "",
    },
    {
        "title": "Virtual Assistant",
        "company": "Example Remote Staffing",
        "location": "Remote",
        "description": (
            "Support overseas clients with email management, calendar "
            "scheduling, research, social media posting, and bookkeeping "
            "tasks. Self-motivated with reliable internet and good written "
            "English."
        ),
        "url": "",
    },
    {
        "title": "Human Resources Assistant",
        "company": "Example People Solutions",
        "location": "Mandaluyong City, Metro Manila",
        "description": (
            "Assist with recruitment, onboarding, payroll, timekeeping, and "
            "employee records. Knowledge of Philippine labor law, SSS, "
            "PhilHealth, and Pag-IBIG processes. Good interpersonal skills."
        ),
        "url": "",
    },
    {
        "title": "Sales and Marketing Associate",
        "company": "Example Retail Brands",
        "location": "Davao City, Davao del Sur",
        "description": (
            "Generate leads, meet sales targets, and build client "
            "relationships. Plan digital marketing campaigns, social media "
            "content, and product promotions. Strong negotiation and "
            "presentation skills."
        ),
        "url": "",
    },
    # --- Finance & Accounting ---
    {
        "title": "Accounting Staff / Bookkeeper",
        "company": "Example Finance Corp.",
        "location": "Makati City, Metro Manila",
        "description": (
            "Prepare journal entries, bank reconciliations, accounts "
            "payable and receivable, and financial statements. Familiar "
            "with QuickBooks or Xero, Excel, and BIR tax filing. CPA or "
            "accountancy graduate preferred."
        ),
        "url": "",
    },
    # --- Healthcare ---
    {
        "title": "Staff Nurse",
        "company": "Example Medical Center",
        "location": "Manila, Metro Manila",
        "description": (
            "Provide patient care, administer medication, monitor vital "
            "signs, and maintain medical records. Licensed Registered Nurse "
            "with BLS/ACLS certification. Strong communication and "
            "teamwork in a hospital setting."
        ),
        "url": "",
    },
    {
        "title": "Medical Technologist",
        "company": "Example Diagnostics Laboratory",
        "location": "Iloilo City, Iloilo",
        "description": (
            "Perform laboratory tests including hematology, clinical "
            "chemistry, and microbiology. Operate and calibrate lab "
            "equipment, ensure quality control, and record results "
            "accurately. Licensed RMT required."
        ),
        "url": "",
    },
    # --- Education ---
    {
        "title": "Elementary / High School Teacher",
        "company": "Example Learning Academy",
        "location": "Baguio City, Benguet",
        "description": (
            "Plan lessons, deliver classroom instruction, assess student "
            "progress, and manage classroom behavior. LET passer with "
            "strong communication, curriculum development, and "
            "online teaching experience preferred."
        ),
        "url": "",
    },
    # --- Engineering & Technical ---
    {
        "title": "Civil Engineer",
        "company": "Example Construction & Development",
        "location": "Cavite",
        "description": (
            "Prepare structural plans, cost estimates, and project "
            "schedules. Supervise construction sites, ensure safety "
            "compliance, and coordinate with contractors. AutoCAD and "
            "licensed Civil Engineer required."
        ),
        "url": "",
    },
    {
        "title": "Electrician",
        "company": "Example Building Services",
        "location": "Laguna",
        "description": (
            "Install, maintain, and repair electrical wiring, panels, and "
            "lighting systems in residential and commercial buildings. "
            "Read blueprints and follow safety codes. TESDA NC II "
            "certification preferred."
        ),
        "url": "",
    },
    {
        "title": "Production Operator",
        "company": "Example Manufacturing Inc.",
        "location": "Batangas",
        "description": (
            "Operate production machines, perform quality inspections, and "
            "follow standard operating procedures and 5S. Able to work "
            "shifting schedules. High school or vocational graduate."
        ),
        "url": "",
    },
    # --- Logistics & Transport ---
    {
        "title": "Warehouse and Logistics Coordinator",
        "company": "Example Logistics Hub",
        "location": "Valenzuela City, Metro Manila",
        "description": (
            "Manage inventory, receiving, dispatching, and shipment "
            "tracking. Use Excel and inventory management systems, "
            "coordinate with delivery drivers and suppliers. Forklift "
            "experience is a plus."
        ),
        "url": "",
    },
    {
        "title": "Delivery Driver",
        "company": "Example Express Couriers",
        "location": "Cagayan de Oro, Misamis Oriental",
        "description": (
            "Deliver parcels safely and on time, plan routes, and maintain "
            "vehicle condition. Valid professional driver's license with "
            "knowledge of local roads and good customer service."
        ),
        "url": "",
    },
    # --- Hospitality & Food Service ---
    {
        "title": "Hotel Front Desk Officer",
        "company": "Example Resort & Hotel",
        "location": "Boracay, Aklan",
        "description": (
            "Handle guest check-in and check-out, reservations, inquiries, "
            "and complaints. Strong customer service, cash handling, and "
            "communication skills. Hospitality or tourism graduate preferred."
        ),
        "url": "",
    },
    {
        "title": "Cook / Kitchen Staff",
        "company": "Example Restaurant Group",
        "location": "Pampanga",
        "description": (
            "Prepare and cook menu items, maintain food safety and "
            "sanitation standards, manage kitchen inventory, and work "
            "efficiently in a fast-paced team. Culinary training or TESDA "
            "Cookery NC II is an advantage."
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
        logger.warning("philjobnet_scraper.py not found or 'beautifulsoup4' not installed — skipping PhilJobNet")
        return []
    try:
        return search_philjobnet(query, max_results=max_results)
    except Exception as e:
        logger.warning("Couldn't fetch live PhilJobNet listings (%s)", e)
        return []


def _try_fetch_adzuna(query, country, results_per_page):
    """Fetches Adzuna API listings, returning [] on failure."""
    if not JSEARCH_AVAILABLE:
        logger.warning("job_scraper.py not found or 'requests' not installed — skipping Adzuna")
        return []
    try:
        return search_jsearch(query, country=country, results_per_page=results_per_page)
    except Exception as e:
        logger.warning("Couldn't fetch live Adzuna listings (%s)", e)
        return []


def _try_fetch_onlinejobs(query, max_results):
    """Fetches OnlineJobs.ph listings, returning [] on failure."""
    if not ONLINEJOBS_AVAILABLE:
        logger.warning("onlinejobs_scraper.py not found or 'beautifulsoup4' not installed — skipping OnlineJobs.ph")
        return []
    try:
        return search_onlinejobs(query, max_results=max_results)
    except Exception as e:
        logger.warning("Couldn't fetch live OnlineJobs.ph listings (%s)", e)
        return []


def _normalize_key(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def dedupe_jobs(jobs):
    """Removes duplicate listings (same URL, or same title + company) across sources."""
    seen_urls, seen_pairs, unique = set(), set(), []
    for job in jobs:
        url = (job.get("url") or "").strip().lower()
        pair = (_normalize_key(job.get("title")), _normalize_key(job.get("company")))
        if (url and url in seen_urls) or (pair[0] and pair in seen_pairs):
            continue
        if url:
            seen_urls.add(url)
        if pair[0]:
            seen_pairs.add(pair)
        unique.append(job)
    return unique


def get_job_listings(search_query=None, country=DEFAULT_COUNTRY, source="all", results_per_source=10, force_refresh=False, return_meta=False):
    """
    Returns aggregated job listings from the requested sources.
    Uses SQLite job cache to return instant results (<10ms) when available.
    Only scrapes sources that are not in cache or when force_refresh is True.
    Returns SAMPLE_JOB_LISTINGS (identity-checkable) when nothing live or cached is found.
    """
    if not search_query:
        meta = {"from_cache": False, "cached_sources": [], "live_sources": [], "is_placeholder": True}
        return (SAMPLE_JOB_LISTINGS, meta) if return_meta else SAMPLE_JOB_LISTINGS

    if source in ("all", "both"):
        requested_sources = ["philjobnet", "onlinejobs", "adzuna"]
    elif source == "onlinejobs":
        requested_sources = ["onlinejobs"]
    elif source in ("adzuna", "jsearch"):
        requested_sources = ["adzuna"]
    else:
        requested_sources = ["philjobnet"]

    cached_jobs_map = {}
    cached_sources = []
    live_sources = []
    sources_to_scrape = []

    if CACHE_AVAILABLE and not force_refresh:
        for src in requested_sources:
            cached_res = get_cached_jobs(src, search_query, country=(country if src == "adzuna" else None))
            if cached_res and cached_res.get("jobs"):
                cached_jobs_map[src] = cached_res["jobs"]
                cached_sources.append(src)
                logger.info("Retrieved %d cached listings for %s ('%s', age: %.1fm)",
                            len(cached_res["jobs"]), src, search_query, cached_res["age_seconds"] / 60.0)
            else:
                sources_to_scrape.append(src)
    else:
        sources_to_scrape = list(requested_sources)

    if sources_to_scrape:
        logger.info("Scraping live listings for: %s", ", ".join(sources_to_scrape))

        def _fetch_source(src):
            if src == "philjobnet":
                return src, _try_fetch_philjobnet(search_query, results_per_source)
            elif src == "adzuna":
                return src, _try_fetch_adzuna(search_query, country, results_per_source)
            elif src == "onlinejobs":
                # Max 6 results to respect crawl rate and keep latency reasonable
                return src, _try_fetch_onlinejobs(search_query, min(results_per_source, 6))
            return src, []

        with ThreadPoolExecutor(max_workers=max(1, len(sources_to_scrape))) as executor:
            futures = [executor.submit(_fetch_source, src) for src in sources_to_scrape]
            for f in futures:
                src, scraped_jobs = f.result()
                if scraped_jobs:
                    live_sources.append(src)
                    cached_jobs_map[src] = scraped_jobs
                    if CACHE_AVAILABLE:
                        set_cached_jobs(src, search_query, scraped_jobs, country=(country if src == "adzuna" else None))
                else:
                    # Fallback to stale cached data if live scrape returned nothing
                    if CACHE_AVAILABLE:
                        stale_res = get_cached_jobs(src, search_query, country=(country if src == "adzuna" else None), allow_stale=True)
                        if stale_res and stale_res.get("jobs"):
                            logger.warning("Using stale cache (%d jobs) for '%s' after live scrape returned empty",
                                           len(stale_res["jobs"]), src)
                            cached_jobs_map[src] = stale_res["jobs"]
                            cached_sources.append(src)

    all_jobs = []
    for src in requested_sources:
        # Tag each listing with the board it came from (used by the source filter)
        all_jobs.extend({**job, "source": src} for job in cached_jobs_map.get(src, []))

    jobs = dedupe_jobs(all_jobs)
    if jobs:
        logger.info("Combined %d listings across %s (%d unique)",
                    len(all_jobs), ", ".join(requested_sources), len(jobs))
    else:
        logger.warning("No live or cached results for '%s' — using placeholder listings", search_query)
        jobs = SAMPLE_JOB_LISTINGS

    is_placeholder = jobs is SAMPLE_JOB_LISTINGS
    meta = {
        "from_cache": len(cached_sources) > 0 and len(live_sources) == 0,
        "cached_sources": cached_sources,
        "live_sources": live_sources,
        "is_placeholder": is_placeholder,
    }

    if return_meta:
        return jobs, meta
    return jobs


def run_pipeline(resume_filepath, job_listings=None, search_query=None, country=DEFAULT_COUNTRY, source="all", force_refresh=False):
    """
    Full pipeline: parses resume and ranks it against job listings.
    Automatically derives a search query if none is provided.
    Supports force_refresh to bypass the job cache when requested.
    """
    logger.info("Parsing resume")
    parsed_resume = parse_resume(resume_filepath)

    # Don't log name/email/phone — resumes contain personal data.
    logger.info("Detected %d skill(s)", len(parsed_resume.get("skills") or []))

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
            logger.info("Auto-applied %d suggested skill(s) for this matching run "
                        "(the resume file itself is unchanged).", len(suggested_skills))

    search_info = {
        "query": search_query,
        "confidence": None,
        "source": source,
        "is_placeholder": False,
        "from_cache": False,
        "cached_sources": [],
        "live_sources": [],
    }
    if job_listings is None:
        query = search_query
        confidence = None
        if query is None:
            query, confidence = build_search_query_from_resume(parsed_resume)
            if query and confidence is not None:
                logger.info('Inferred likely occupation: "%s" (confidence: %.0f%%)', query, confidence * 100)
            elif query:
                logger.info('Auto-detected search query from skills: "%s"', query)
            else:
                logger.info("No usable skills detected to build a search query from")

        job_listings, cache_meta = get_job_listings(
            query, country=country, source=source, force_refresh=force_refresh, return_meta=True
        )
        search_info = {
            "query": query,
            "confidence": confidence,
            "source": source,
            "is_placeholder": job_listings is SAMPLE_JOB_LISTINGS,
            "from_cache": cache_meta.get("from_cache", False),
            "cached_sources": cache_meta.get("cached_sources", []),
            "live_sources": cache_meta.get("live_sources", []),
        }

    # Normalize salary / work setup / job type / posting date for filtering.
    # Done after the cache so older cached entries get the same fields.
    default_source = "sample" if search_info["is_placeholder"] else None
    job_listings = [enrich_job(job, default_source=default_source) for job in job_listings]

    logger.info("Matching against %d job listings...", len(job_listings))
    ranked_results = rank_jobs_for_resume(parsed_resume, job_listings)

    for job in ranked_results:
        logger.info("%s%% raw / %s%% relative — %s%s", job['match_score'], job['relative_score'],
                    job.get('title', 'Untitled listing'),
                    f" at {job['company']}" if job.get('company') else '')

    return {
        "parsed_resume": parsed_resume,
        "health_check": health_check,
        "search_info": search_info,
        "ranked_results": ranked_results,
    }



def _parse_args(argv=None):
    """CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Parse a resume and rank live job listings against it.",
        epilog="--country applies to the Adzuna source; 'ph' is not supported by Adzuna and auto-falls back to 'gb'.",
    )
    parser.add_argument("resume_path", help="Path to a .pdf or .docx resume")
    parser.add_argument("--query", default=None,
                        help="Custom search query (default: auto-derived from the resume)")
    parser.add_argument("--country", default=DEFAULT_COUNTRY, help=f"Country code (default: {DEFAULT_COUNTRY})")
    parser.add_argument("--source", default="all", choices=["all", "philjobnet", "adzuna", "onlinejobs"],
                        help="Job source (default: all, merged and ranked together)")
    parser.add_argument("--refresh", "--force-refresh", action="store_true",
                        help="Force live scraping, bypassing the local job cache")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = _parse_args()
    result = run_pipeline(
        args.resume_path,
        search_query=args.query,
        country=args.country,
        source=args.source,
        force_refresh=args.refresh,
    )

    print("\n--- Ranked matches ---")
    for job in result["ranked_results"]:
        print(f"\n{job['match_score']}% raw / {job['relative_score']}% relative  —  {job.get('title', 'Untitled listing')}"
              f"{' at ' + job['company'] if job.get('company') else ''}")
        print(f"   {job['comment']}")
        if job.get("url"):
            print(f"   {job['url']}")
