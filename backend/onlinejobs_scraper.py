"""
OnlineJobs.ph Scraper
-----------------------
Fetches real, live job listings from OnlineJobs.ph — the largest
Philippine marketplace for remote/virtual-assistant work. Chosen
because it's confirmed genuinely scrapable: a real robots.txt with a
Crawl-delay directive (meaning crawling is permitted, just paced),
no login wall on public job listings, and other developers already
scrape it openly (several public tools/repos do this).

TWO HONEST LIMITATIONS, confirmed before writing this:
  1. Employer/company names are BLURRED and require a login to view
     on this site — not something a scraper can get around. This
     script honestly returns "Unknown" for company rather than
     guessing. Title, description, salary, work type, and required
     skills are all openly visible without an account.
  2. This site is known (per a 2026 developer report) to BLOCK
     requests coming from cloud server IP addresses — it works fine
     from a normal residential connection (your own PC), but would
     likely get blocked if this script runs on a cloud-hosted server
     later (e.g. if you deploy your REST API to one). Keep that in
     mind down the line.

VERIFIED BUT NOT YET TEST-RUN: URL patterns and the real section
labels below ("JOB OVERVIEW", "SKILL REQUIREMENT", etc.) were
confirmed against real, live pages before writing this. What
couldn't be verified: the exact HTML tags/CSS classes, since this
environment only sees converted text, not raw HTML, and can't run
this script network-side. Run it and tell me what happens — same
approach as the other scrapers in this project.

Install dependencies:
    pip install requests beautifulsoup4

Usage:
    python onlinejobs_scraper.py "virtual assistant"
    python onlinejobs_scraper.py "video editor" 5
"""

import re
import sys
import time
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.onlinejobs.ph"
SEARCH_URL_TEMPLATE = BASE_URL + "/jobseekers/jobsearch?jobkeyword={query}"

HEADERS = {
    "User-Agent": "ResumeAid-CourseProject/1.0 (educational resume-matching tool; low request rate)"
}

# Text labels confirmed to exist verbatim on real OnlineJobs.ph job
# detail pages — used as section boundaries the same way
# philjobnet_scraper.py does (see that file for the full reasoning).
KNOWN_SECTION_LABELS = [
    "type of work", "wage / salary", "hours per week", "date updated",
    "job overview", "skill requirement", "view other job posts from",
    "share this post", "report",
]

# OnlineJobs.ph's own robots.txt specifies Crawl-delay: 5 (confirmed
# via multiple independent developer reports) — respected here.
REQUEST_DELAY_SECONDS = 5.0


def _get_soup(url):
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def _find_job_detail_links(search_results_soup, max_results):
    """
    Finds job detail page URLs by matching the URL PATTERN (confirmed
    real: /jobseekers/job/{slug}-{id}) rather than any CSS class.
    """
    pattern = re.compile(r"/jobseekers/job/[\w-]+-\d+")
    seen = set()
    urls = []

    for a_tag in search_results_soup.find_all("a", href=True):
        href = a_tag["href"]
        if pattern.search(href):
            full_url = href if href.startswith("http") else BASE_URL + href
            if full_url not in seen:
                seen.add(full_url)
                urls.append(full_url)
        if len(urls) >= max_results:
            break

    return urls


def _extract_section_text(soup, heading_text, max_chars=2000):
    """
    Finds a tag whose own text matches heading_text (case-
    insensitive, trailing colon ignored — some real headings on this
    site end in ":"), collects text from subsequent tags in document
    order until hitting any OTHER known section label. Returns ""
    if the heading isn't found.
    """
    heading_lower = heading_text.strip().rstrip(":").lower()
    other_labels = {lbl.rstrip(":") for lbl in KNOWN_SECTION_LABELS if lbl != heading_lower}

    heading_tag = None
    for tag in soup.find_all(True):
        if tag.get_text(strip=True).rstrip(":").lower() == heading_lower:
            heading_tag = tag
            break

    if heading_tag is None:
        return ""

    collected = []
    total_len = 0
    for sibling in heading_tag.find_all_next():
        sibling_text = sibling.get_text(strip=True)
        if not sibling_text:
            continue
        normalized = sibling_text.rstrip(":").lower()
        if normalized == heading_lower:
            continue  # the heading's own text (or a duplicate node of it) — not content
        if normalized in other_labels:
            break  # hit the next section — stop collecting
        collected.append(sibling.get_text(" ", strip=True))
        total_len += len(collected[-1])
        if total_len >= max_chars:
            break

    return " ".join(collected)[:max_chars].strip()


def _parse_job_detail(url):
    """
    Fetches one job detail page and extracts a job dict matching the
    project-wide schema: title, company, description, url, location.
    """
    soup = _get_soup(url)

    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)
    else:
        slug_match = re.search(r"/job/([\w-]+)-\d+$", url)
        title = slug_match.group(1).replace("-", " ").title() if slug_match else "Untitled listing"

    job_overview = _extract_section_text(soup, "Job Overview")
    work_type = _extract_section_text(soup, "Type Of Work", max_chars=100)
    salary = _extract_section_text(soup, "Wage / Salary", max_chars=100)
    skills = _extract_section_text(soup, "Skill Requirement", max_chars=300)

    description_parts = [p for p in [work_type and f"Type: {work_type}", salary and f"Pay: {salary}", job_overview, skills and f"Skills: {skills}"] if p]
    description = "\n".join(description_parts).strip()

    # Company name is blurred/login-gated on this site — see module
    # docstring. Honest "Unknown" rather than a guess.
    company = "Unknown"

    # This entire platform is remote work for Filipino workers — a
    # per-job city is rarely specified, so this default is accurate
    # for virtually every listing here.
    location = "Remote — Philippines"

    return {
        "title": title,
        "company": company,
        "description": description,
        "url": url,
        "location": location,
    }


def search_jobs(query, max_results=10):
    """
    Searches OnlineJobs.ph for `query`, returns a list of job dicts
    in the SAME schema used throughout this project — drop-in
    compatible with pipeline.py, matching_engine.py, etc.
    """
    search_url = SEARCH_URL_TEMPLATE.format(query=quote(query))
    search_soup = _get_soup(search_url)

    detail_urls = _find_job_detail_links(search_soup, max_results)
    if not detail_urls:
        return []

    jobs = []
    for i, url in enumerate(detail_urls):
        if i > 0:
            time.sleep(REQUEST_DELAY_SECONDS)  # respects the site's Crawl-delay: 5
        try:
            jobs.append(_parse_job_detail(url))
        except requests.RequestException as e:
            print(f"Skipping {url} (request failed: {e})")
            continue

    return jobs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python onlinejobs_scraper.py "search query" [max_results]')
        sys.exit(1)

    query = sys.argv[1]
    max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    jobs = search_jobs(query, max_results=max_results)

    print(f"Found {len(jobs)} jobs for '{query}':\n")
    for job in jobs:
        print(f"- {job['title']} ({job['location']})")
        print(f"  {job['description'][:150]}...")
        print(f"  {job['url']}\n")
