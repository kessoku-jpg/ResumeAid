"""
Job Listing Fetcher (Adzuna API)
---------------------------------
Replaces the hardcoded SAMPLE_JOB_LISTINGS placeholder in pipeline.py
with real, live job postings from the Adzuna API.

WHY ADZUNA: free tier (1,000 calls/month, no cost), instant signup (no
approval wait), broad country coverage, and returns clean structured
JSON — no scraping, no HTML parsing, no risk of breaking when a job
site redesigns its page.

SETUP (one-time):
    1. Go to https://developer.adzuna.com/ and register — instant,
       free, gives you an App ID and App Key immediately.
    2. Set them as environment variables (don't hardcode secrets in
       code that might end up in a public repo):

       Windows (PowerShell):
           $env:ADZUNA_APP_ID = "your_app_id"
           $env:ADZUNA_APP_KEY = "your_app_key"
       Mac/Linux:
           export ADZUNA_APP_ID="your_app_id"
           export ADZUNA_APP_KEY="your_app_key"

       (To make these permanent rather than per-terminal-session, add
       them to your system environment variables or your shell's
       profile file.)

    3. Install the one dependency:
           pip install requests

Install dependencies:
    pip install requests

Usage:
    python job_scraper.py "python developer" us
    python job_scraper.py "hr assistant" ph      (see country-code note below)
"""

import os
import sys

import requests

BASE_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"

# Country codes Adzuna supports as of this writing. The Philippines is
# NOT among them — Adzuna doesn't operate there. For this project,
# "sg" (Singapore) or "au" (Australia) are the closest regional
# options if you want APAC-relevant listings; "us" or "gb" give the
# broadest job volume for a demo. Pick whichever fits your test case.
SUPPORTED_COUNTRIES = {
    "au", "at", "br", "ca", "de", "fr", "in", "it", "mx",
    "nl", "nz", "pl", "ru", "sg", "za", "us", "gb",
}


def search_jobs(query, country="us", results_per_page=10, page=1):
    """
    Searches Adzuna for jobs matching `query` in `country`, returns a
    list of job dicts in the SAME schema pipeline.py's
    SAMPLE_JOB_LISTINGS already uses — so this is a drop-in
    replacement, nothing else needs to change:
        {"title": ..., "company": ..., "description": ..., "url": ..., "location": ...}

    Raises a clear error if credentials aren't set, rather than a
    confusing HTTP failure.
    """
    app_id = os.environ.get("ADZUNA_APP_ID")
    app_key = os.environ.get("ADZUNA_APP_KEY")

    if not app_id or not app_key:
        raise RuntimeError(
            "ADZUNA_APP_ID and/or ADZUNA_APP_KEY environment variables "
            "aren't set. Register for free at https://developer.adzuna.com/ "
            "and set both — see this file's setup instructions at the top."
        )

    if country not in SUPPORTED_COUNTRIES:
        raise ValueError(
            f"'{country}' isn't a country Adzuna supports. Choose from: "
            f"{sorted(SUPPORTED_COUNTRIES)}"
        )

    url = BASE_URL.format(country=country, page=page)
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": query,
        "results_per_page": results_per_page,
        "content-type": "application/json",
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()  # raises an exception on 4xx/5xx errors
    data = response.json()

    jobs = []
    for result in data.get("results", []):
        jobs.append({
            "title": result.get("title", "").strip(),
            "company": result.get("company", {}).get("display_name", "Unknown"),
            "description": result.get("description", "").strip(),
            "url": result.get("redirect_url", ""),
            "location": result.get("location", {}).get("display_name", ""),
        })

    return jobs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python job_scraper.py "job search query" [country_code]')
        print(f"Supported country codes: {sorted(SUPPORTED_COUNTRIES)}")
        sys.exit(1)

    query = sys.argv[1]
    country = sys.argv[2] if len(sys.argv) > 2 else "us"

    try:
        jobs = search_jobs(query, country=country)
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Found {len(jobs)} jobs for '{query}' in {country}:\n")
    for job in jobs:
        print(f"- {job['title']} at {job['company']} ({job['location']})")
        print(f"  {job['description'][:150]}...")
        print(f"  {job['url']}\n")
