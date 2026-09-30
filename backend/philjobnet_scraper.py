"""Scrapes real job listings from PhilJobNet (philjobnet.gov.ph)."""

import re
import sys
import time
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://philjobnet.gov.ph"
SEARCH_URL_TEMPLATE = BASE_URL + "/job-vacancies/0/{query}/0"

# Identifies this tool honestly rather than pretending to be a browser —
# standard courtesy for a small, occasional, non-commercial scraper
# hitting a public government server.
HEADERS = {
    "User-Agent": "ResumeAid-CourseProject/1.0 (educational resume-matching tool; low request rate)"
}

# Text labels confirmed to exist verbatim on real PhilJobNet job detail
# pages. Used as SECTION BOUNDARIES for extraction — when collecting
# text under one heading, stop as soon as any of the other labels in
# this list is encountered, rather than guessing where a <div> ends.
KNOWN_SECTION_LABELS = [
    "job description", "qualifications/requirements", "work location",
    "remarks", "about the company", "industry", "employment size",
    "share job",
]

# Politeness delay between requests to the job detail pages (seconds).
# This is a public government server, not a rate-limited API — keep
# this reasonable rather than hitting it as fast as possible.
REQUEST_DELAY_SECONDS = 1.0


def _get_soup(url):
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def _find_job_detail_links(search_results_soup, max_results):
    """Extracts job detail URLs from the search results page."""
    pattern = re.compile(r"/job-vacancies/job/[\w-]+-\d+")
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
    """Extracts text for a specific section heading from the job page."""
    heading_lower = heading_text.strip().lower()
    other_labels = {lbl for lbl in KNOWN_SECTION_LABELS if lbl != heading_lower}

    heading_tag = None
    for tag in soup.find_all(True):
        if tag.get_text(strip=True).lower() == heading_lower:
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
        if sibling_text.lower() in other_labels:
            break  # hit the next section — stop collecting
        collected.append(sibling.get_text(" ", strip=True))
        total_len += len(collected[-1])
        if total_len >= max_chars:
            break

    return " ".join(collected)[:max_chars].strip()


def _parse_job_detail(url):
    """Parses a single job detail page into the standard job schema."""
    soup = _get_soup(url)

    # Title: prefer the page's <h1>; fall back to deriving it from the
    # URL slug (e.g. ".../job/senior-cashier-1472242" -> "Senior Cashier")
    # if no h1 is found.
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)
    else:
        slug_match = re.search(r"/job/([\w-]+)-\d+$", url)
        title = slug_match.group(1).replace("-", " ").title() if slug_match else "Untitled listing"

    # Company: a link to a company profile page, confirmed real
    # pattern /job-vacancies/company/{slug}-{id}
    company_link = soup.find("a", href=re.compile(r"/job-vacancies/company/"))
    company = company_link.get_text(strip=True) if company_link else "Unknown"

    # Description: combine the "Job Description" and
    # "Qualifications/Requirements" sections — both confirmed present
    # as literal headings on real pages.
    job_description = _extract_section_text(soup, "Job Description")
    qualifications = _extract_section_text(soup, "Qualifications/Requirements")
    description = (job_description + "\n" + qualifications).strip()

    # Location: the "Work location" section specifically (there's also
    # a location line earlier on the page near the company name, but
    # this heading is the unambiguous, clearly-labeled one)
    location = _extract_section_text(soup, "Work location", max_chars=200)

    return {
        "title": title,
        "company": company,
        "description": description,
        "url": url,
        "location": location,
    }


def _sanitize_query(query):
    """Sanitizes search query for PhilJobNet URL routing."""
    query = re.sub(r",?\s*all other$", "", query, flags=re.IGNORECASE).strip()
    query = re.sub(r"[,.]", "", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query


def search_jobs(query, max_results=10):
    """Searches PhilJobNet and returns standard job dicts."""
    clean_query = _sanitize_query(query)
    search_url = SEARCH_URL_TEMPLATE.format(query=quote(clean_query))
    search_soup = _get_soup(search_url)

    detail_urls = _find_job_detail_links(search_soup, max_results)
    if not detail_urls:
        return []

    jobs = []
    for i, url in enumerate(detail_urls):
        if i > 0:
            time.sleep(REQUEST_DELAY_SECONDS)  # politeness delay between detail page fetches
        try:
            jobs.append(_parse_job_detail(url))
        except requests.RequestException as e:
            print(f"Skipping {url} (request failed: {e})")
            continue

    return jobs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python philjobnet_scraper.py "search query" [max_results]')
        sys.exit(1)

    query = sys.argv[1]
    max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    jobs = search_jobs(query, max_results=max_results)

    print(f"Found {len(jobs)} jobs for '{query}':\n")
    for job in jobs:
        print(f"- {job['title']} at {job['company']} ({job['location']})")
        print(f"  {job['description'][:150]}...")
        print(f"  {job['url']}\n")
