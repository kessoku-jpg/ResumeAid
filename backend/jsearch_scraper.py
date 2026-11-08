"""Fetches job listings via the JSearch API (OpenWeb Ninja).

JSearch aggregates Google for Jobs, so it covers the Philippines, which Adzuna
does not. The pipeline only uses this source when the search country is "ph"
(or when source="jsearch" is requested explicitly).

Docs: https://www.openwebninja.com/api/jsearch/docs

Setup: create an account at https://app.openwebninja.com, subscribe to JSearch
(free tier available) and put your key in backend/.env as JSEARCH_API_KEY.
(This is NOT a RapidAPI key; JSearch is no longer served from RapidAPI.)
"""

import os
import sys

import requests

# Load backend/.env so the key is found when this file is run standalone too.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from job_attributes import build_salary

JSEARCH_URL = "https://api.openwebninja.com/jsearch/search-v2"

COUNTRY_NAMES = {"ph": "Philippines"}

# Fallback currency when JSearch returns a salary range without a currency.
COUNTRY_CURRENCY = {"ph": "PHP"}

# JSearch employment types -> the project's normalized employment_type values.
EMPLOYMENT_TYPES = {
    "FULLTIME": "full_time",
    "PARTTIME": "part_time",
    "CONTRACTOR": "contract",
    "INTERN": "internship",
}

# JSearch salary periods -> job_attributes pay periods.
SALARY_PERIODS = {
    "YEAR": "year", "MONTH": "month", "WEEK": "week", "DAY": "day", "HOUR": "hour",
}


def _location(result):
    """Prefers v2's single 'job_location' string, else joins city/state/country."""
    direct = str(result.get("job_location") or "").strip()
    if direct:
        return direct
    parts = [result.get("job_city"), result.get("job_state"), result.get("job_country")]
    return ", ".join(str(p).strip() for p in parts if p and str(p).strip())


def _employment_type(result):
    """Maps JSearch's job_employment_types list (v2) or job_employment_type string."""
    candidates = list(result.get("job_employment_types") or [])
    candidates.append(result.get("job_employment_type"))
    for raw in candidates:
        key = str(raw or "").upper().replace("-", "").replace("_", "").replace(" ", "")
        if key in EMPLOYMENT_TYPES:
            return EMPLOYMENT_TYPES[key]
    return None


def _to_float(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _salary(result, country="ph"):
    low = _to_float(result.get("job_min_salary"))
    high = _to_float(result.get("job_max_salary"))
    period = SALARY_PERIODS.get(str(result.get("job_salary_period") or "").upper())
    currency = result.get("job_salary_currency") or COUNTRY_CURRENCY.get((country or "").lower())
    if (low is None and high is None) or not period or not currency:
        return None
    return build_salary(low, high, currency, period, text=result.get("job_salary_string") or None)


def _apply_url(result):
    """Best application link: the main one, else the first of apply_options."""
    link = result.get("job_apply_link")
    if link:
        return link
    for option in result.get("apply_options") or []:
        if option.get("apply_link"):
            return option["apply_link"]
    return result.get("job_google_link") or ""


def _description(result):
    """Description plus highlights (qualifications/responsibilities/benefits) when present."""
    text = (result.get("job_description") or "").strip()
    highlights = result.get("job_highlights") or {}
    extras = []
    if isinstance(highlights, dict):
        for section, items in highlights.items():
            if isinstance(items, list) and items:
                extras.append(f"{section}:\n" + "\n".join(f"- {i}" for i in items))
    if extras and not any(e.split("\n", 1)[1][:40] in text for e in extras):
        text = (text + "\n\n" + "\n\n".join(extras)).strip()
    return text


def _query_candidates(query):
    """Full query first, then shorter fallbacks.

    The pipeline sometimes sends a keyword list built from resume skills
    (e.g. "Python SQL Excel Java Communication"). Google Jobs treats that as
    one very narrow search and JSearch returns nothing, so we retry with the
    first two terms and then the first term alone.
    """
    words = (query or "").split()
    candidates = [" ".join(words)]
    if len(words) > 2:
        candidates.append(" ".join(words[:2]))
    if len(words) > 1:
        candidates.append(words[0])
    return [c for i, c in enumerate(candidates) if c and c not in candidates[:i]]


def _fetch_raw(query, code, api_key):
    """Calls search-v2 once and returns the raw job dicts."""
    place = COUNTRY_NAMES.get(code)
    params = {
        "query": f"{query} in {place}" if place else query,
        "num_pages": 1,
        "country": code,
        "date_posted": "all",
    }
    headers = {"x-api-key": api_key}

    response = requests.get(JSEARCH_URL, params=params, headers=headers, timeout=30)
    if response.status_code in (401, 403):
        try:
            detail = response.json().get("message") or str(response.text)[:200]
        except Exception:
            detail = str(getattr(response, "text", ""))[:200]
        raise RuntimeError(
            f"JSearch rejected the API key ({response.status_code}: {detail}). "
            "Use the key from app.openwebninja.com (not a RapidAPI key) and make "
            "sure your JSearch subscription is active."
        )
    if response.status_code == 429:
        raise RuntimeError("JSearch rate limit or monthly quota reached (429).")
    response.raise_for_status()

    data = response.json().get("data") or []
    # search-v2 nests the results as data.jobs (plus a pagination cursor).
    return (data.get("jobs") or []) if isinstance(data, dict) else data


def search_jobs(query, country="ph", results_per_page=10, page=1):
    """Searches JSearch and returns jobs in the project's standard schema."""
    # Strip stray whitespace/quotes that commonly sneak in when pasting a key.
    api_key = (os.environ.get("JSEARCH_API_KEY") or "").strip().strip("\"'").strip()
    if not api_key:
        raise RuntimeError(
            "JSEARCH_API_KEY is not set. Create a key at https://app.openwebninja.com "
            "(JSearch API) and add it to backend/.env."
        )

    code = (country or "ph").lower().strip()

    raw_jobs = []
    for candidate in _query_candidates(query):
        raw_jobs = _fetch_raw(candidate, code, api_key)
        if raw_jobs:
            break

    jobs = []
    for result in raw_jobs[:results_per_page]:
        title = (result.get("job_title") or "").strip()
        if not title:
            continue
        job = {
            "title": title,
            "company": (result.get("employer_name") or "").strip() or "Unknown",
            "description": _description(result),
            "url": _apply_url(result),
            "location": _location(result),
            "salary": _salary(result, code),
            "employment_type": _employment_type(result),
            "posted_raw": (result.get("job_posted_at_datetime_utc")
                           or result.get("job_posted_at") or ""),
        }
        if result.get("job_is_remote"):
            job["work_mode"] = "remote"
        jobs.append(job)
    return jobs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python jsearch_scraper.py "job search query" [country_code]')
        sys.exit(1)
    try:
        found = search_jobs(sys.argv[1], country=sys.argv[2] if len(sys.argv) > 2 else "ph")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"Found {len(found)} jobs")
    for j in found:
        sal = j["salary"]["text"] if j.get("salary") and j["salary"].get("text") else "-"
        print(f"- {j['title']} at {j['company']} ({j['location']}) | {j.get('employment_type')} | "
              f"{j.get('work_mode', '-')} | salary: {sal} | posted: {j['posted_raw']}\n  {j['url']}")
