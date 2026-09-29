"""
Job Listing Fetcher (Adzuna API)
---------------------------------
Replaces the old JSearch/RapidAPI integration with Adzuna — a free,
official REST API that doesn't require a middleman service.

WHY ADZUNA: Real API with proper developer access, generous free tier
(250 calls/day, 2,500/month), no scrapers, clean structured JSON, and
worldwide coverage including remote-friendly roles. Adzuna does NOT
support Philippines as a country, so for PH-specific listings use the
philjobnet_scraper.py and onlinejobs_scraper.py sources instead —
this source covers international / remote jobs.

SUPPORTED COUNTRY CODES (Adzuna):
    gb  United Kingdom     us  United States
    au  Australia          ca  Canada
    de  Germany            fr  France
    in  India              br  Brazil
    nz  New Zealand        za  South Africa
    at  Austria            pl  Poland

    ⚠  Philippines (ph) is NOT supported by Adzuna.
       For PH jobs, the pipeline already uses PhilJobNet + OnlineJobs.ph.

SETUP (one-time):
    1. Register for a FREE developer account at:
           https://developer.adzuna.com/
    2. Create an app — you'll get an app_id and app_key immediately.
    3. Set them as environment variables:

       Windows (PowerShell):
           $env:ADZUNA_APP_ID  = "your_app_id"
           $env:ADZUNA_APP_KEY = "your_app_key"
       Mac/Linux:
           export ADZUNA_APP_ID="your_app_id"
           export ADZUNA_APP_KEY="your_app_key"

    4. Install the one dependency:
           pip install requests

Usage:
    python job_scraper.py "python developer"
    python job_scraper.py "data analyst" gb
    python job_scraper.py "project manager" us
"""

import os
import sys

import requests

# Adzuna endpoint: /v1/api/jobs/{country}/search/{page}
ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"

# Countries Adzuna supports. If a country code isn't in this list,
# we fall back to "gb" (the widest English-language market) rather
# than sending an unsupported code that would return a bad response.
SUPPORTED_COUNTRIES = {
    "gb", "us", "au", "ca", "de", "fr", "in", "br",
    "nz", "za", "at", "pl", "nl", "sg", "it", "ru",
}

# Country codes that map to Philippines or aren't supported — all
# fall back to "gb" for international/remote results.
UNSUPPORTED_FALLBACK = "gb"


def _resolve_country(country_code):
    """
    Maps a country code to a valid Adzuna country. Philippines (ph) and
    any other unsupported code fall back to 'gb' with a printed notice.
    """
    code = (country_code or "gb").lower().strip()

    if code == "ph":
        print(
            "  [Adzuna] Philippines (ph) is not supported by Adzuna — "
            "searching 'gb' (UK) for remote/international results instead. "
            "PH-specific jobs come from PhilJobNet and OnlineJobs.ph."
        )
        return UNSUPPORTED_FALLBACK

    if code not in SUPPORTED_COUNTRIES:
        print(
            f"  [Adzuna] Country '{code}' is not in Adzuna's supported list — "
            f"falling back to '{UNSUPPORTED_FALLBACK}'."
        )
        return UNSUPPORTED_FALLBACK

    return code


def search_jobs(query, country="gb", results_per_page=10, page=1):
    """
    Searches Adzuna for jobs matching `query`, returns a list of job
    dicts in the SAME schema the rest of the pipeline uses — drop-in
    replacement for the old JSearch scraper:
        {"title": ..., "company": ..., "description": ..., "url": ..., "location": ...}

    `country` should be an ISO 3166-1 alpha-2 code supported by Adzuna
    (see SUPPORTED_COUNTRIES above). Philippines (ph) is not supported;
    passing it will fall back to 'gb' automatically.

    Raises RuntimeError if ADZUNA_APP_ID or ADZUNA_APP_KEY aren't set.
    """
    app_id = os.environ.get("ADZUNA_APP_ID")
    app_key = os.environ.get("ADZUNA_APP_KEY")

    if not app_id or not app_key:
        raise RuntimeError(
            "ADZUNA_APP_ID and ADZUNA_APP_KEY environment variables are not set. "
            "Register for a free account at https://developer.adzuna.com/ "
            "and set both variables — see this file's setup instructions at the top."
        )

    resolved_country = _resolve_country(country)
    url = ADZUNA_BASE_URL.format(country=resolved_country, page=page)

    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": query,          # keyword search (job title / skills)
        "results_per_page": results_per_page,
        "content-type": "application/json",
        "sort_by": "relevance",
    }

    response = requests.get(url, params=params, timeout=15)

    # Clear, actionable messages for common failure modes
    if response.status_code == 401:
        raise RuntimeError(
            "Adzuna returned 401 Unauthorized — check that ADZUNA_APP_ID and "
            "ADZUNA_APP_KEY are set correctly and match your account at "
            "https://developer.adzuna.com/"
        )
    if response.status_code == 429:
        raise RuntimeError(
            "Adzuna rate limit hit (429). You've exceeded your daily/monthly "
            "request quota. Free tier allows 250 calls/day, 2,500/month."
        )
    if response.status_code == 404:
        raise RuntimeError(
            f"Adzuna returned 404 — country code '{resolved_country}' may not "
            "be supported. Check the list at https://developer.adzuna.com/docs/search"
        )

    response.raise_for_status()
    data = response.json()

    jobs = []
    for result in data.get("results", []):
        title = (result.get("title") or "").strip()
        company = (result.get("company") or {}).get("display_name") or "Unknown"
        description = (result.get("description") or "").strip()
        url_apply = result.get("redirect_url") or ""

        # Location: combine area + location_name if available
        location_obj = result.get("location") or {}
        area_list = location_obj.get("area") or []
        location_str = ", ".join(str(a) for a in area_list if a) if area_list else ""

        jobs.append({
            "title": title,
            "company": company,
            "description": description,
            "url": url_apply,
            "location": location_str,
        })

    return jobs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python job_scraper.py "job search query" [country_code]')
        print('Example: python job_scraper.py "python developer" gb')
        print('         python job_scraper.py "project manager" us')
        print('Supported countries: gb us au ca de fr in br nz za at pl')
        sys.exit(1)

    query = sys.argv[1]
    country = sys.argv[2] if len(sys.argv) > 2 else "gb"

    try:
        jobs = search_jobs(query, country=country)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Found {len(jobs)} jobs for '{query}' (country: {country}):\n")
    for job in jobs:
        print(f"- {job['title']} at {job['company']} ({job['location']})")
        print(f"  {job['description'][:150]}...")
        print(f"  {job['url']}\n")
