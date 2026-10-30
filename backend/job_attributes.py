"""
Normalizes filterable job attributes across sources.

Each scraper exposes whatever raw fields its site provides (salary text,
"Permanent"/"Full Time" labels, posting dates, Adzuna's numeric salaries).
enrich_job() turns those into one consistent shape the frontend can filter
and sort on:

    source           "philjobnet" | "onlinejobs" | "adzuna" | "sample" | None
    salary           {text, min, max, currency, period, monthly_php_min,
                      monthly_php_max, is_estimate} or None
    work_mode        "remote" | "hybrid" | "onsite" | None (unknown)
    employment_type  "full_time" | "part_time" | "contract" | "freelance"
                     | "internship" | None (unknown)
    posted_at        "YYYY-MM-DD" or None

Salary figures are only parsed from fields that are explicitly salary fields
(never guessed from free-form descriptions), and anything unclear becomes
None rather than a wrong number.
"""

import re
from datetime import datetime

# Approximate PHP exchange rates, used ONLY to make salaries in different
# currencies comparable for sorting/filtering. Displayed values keep the
# original currency; converted values are always shown as "≈".
PHP_PER_UNIT = {
    "PHP": 1.0,
    "USD": 58.0,
    "GBP": 75.0,
    "EUR": 63.0,
    "AUD": 38.0,
    "CAD": 42.0,
    "SGD": 44.0,
    "NZD": 34.0,
    "INR": 0.68,
    "ZAR": 3.2,
    "PLN": 14.5,
    "BRL": 10.5,
    "RUB": 0.65,
}

# Adzuna reports salaries in the local currency of the market searched.
ADZUNA_CURRENCY = {
    "gb": "GBP", "us": "USD", "au": "AUD", "ca": "CAD", "sg": "SGD",
    "nz": "NZD", "in": "INR", "za": "ZAR", "pl": "PLN", "br": "BRL",
    "ru": "RUB", "de": "EUR", "fr": "EUR", "nl": "EUR", "it": "EUR",
    "at": "EUR",
}

# Multipliers to convert a pay period into a monthly figure.
HOURS_PER_MONTH = 40 * 52 / 12      # full-time week
WORKDAYS_PER_MONTH = 22
TO_MONTHLY = {
    "hour": HOURS_PER_MONTH,
    "day": WORKDAYS_PER_MONTH,
    "week": 52 / 12,
    "month": 1.0,
    "year": 1 / 12,
}

_NO_SALARY_RE = re.compile(
    r"not\s+specified|negotiable|\btbd\b|\btba\b|to\s+be\s+discussed|depends|"
    r"commission\s+only|unpaid|n/?a\b",
    re.IGNORECASE,
)

_CURRENCY_PATTERNS = [
    ("PHP", re.compile(r"₱|\bphp\b|\bpesos?\b|(?<![a-z])p\s?(?=\d)", re.IGNORECASE)),
    ("GBP", re.compile(r"£|\bgbp\b", re.IGNORECASE)),
    ("EUR", re.compile(r"€|\beur\b|\beuros?\b", re.IGNORECASE)),
    ("AUD", re.compile(r"\baud\b|a\$", re.IGNORECASE)),
    ("CAD", re.compile(r"\bcad\b|c\$", re.IGNORECASE)),
    ("SGD", re.compile(r"\bsgd\b|s\$", re.IGNORECASE)),
    ("NZD", re.compile(r"\bnzd\b|nz\$", re.IGNORECASE)),
    ("INR", re.compile(r"₹|\binr\b", re.IGNORECASE)),
    ("USD", re.compile(r"\$|\busd\b|\bdollars?\b", re.IGNORECASE)),
]

_PERIOD_PATTERNS = [
    ("hour", re.compile(r"/\s*h(ou)?r\b|per\s+hour|hourly|an\s+hour|/\s*h\b", re.IGNORECASE)),
    ("day", re.compile(r"/\s*day\b|per\s+day|daily|a\s+day", re.IGNORECASE)),
    ("week", re.compile(r"/\s*w(ee)?k\b|per\s+week|weekly|a\s+week", re.IGNORECASE)),
    ("month", re.compile(r"/\s*mo(nth)?\b|per\s+month|monthly|a\s+month", re.IGNORECASE)),
    ("year", re.compile(r"/\s*y(ea)?r\b|per\s+(year|annum)|annual(ly)?|yearly|a\s+year|p\.?a\.?\b", re.IGNORECASE)),
]

_NUMBER_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(k\b)?", re.IGNORECASE)

_REMOTE_RE = re.compile(
    r"\bremote(ly)?\b|work[\s-]*from[\s-]*home|\bwfh\b|home[\s-]*based|"
    r"telecommut|anywhere\s+in\s+the\s+(world|philippines)|virtual\s+assistant",
    re.IGNORECASE,
)
_HYBRID_RE = re.compile(r"\bhybrid\b", re.IGNORECASE)
_ONSITE_RE = re.compile(r"\bon[\s-]?site\b|\bonsite\b|in[\s-]office|office[\s-]based|skeleton\s+work", re.IGNORECASE)

_TYPE_RULES = [
    ("internship", re.compile(r"\bintern(ship)?s?\b|\bojt\b|on[\s-]the[\s-]job\s+train(ing|ee)|apprentice", re.IGNORECASE)),
    ("part_time", re.compile(r"part[\s_-]?time", re.IGNORECASE)),
    ("freelance", re.compile(r"freelance|\bgig\b|project[\s-]?based\s+freelance|per\s+project", re.IGNORECASE)),
    ("contract", re.compile(r"contract(ual)?|project[\s_-]?based|seasonal|temporary|\btemp\b|fixed[\s-]term|probationary|casual", re.IGNORECASE)),
    ("full_time", re.compile(r"full[\s_-]?time|permanent|regular", re.IGNORECASE)),
]

_DATE_FORMATS = ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y")


def _to_number(raw, has_k):
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return None
    return value * 1000 if has_k else value


def _detect_currency(text, default):
    for code, pattern in _CURRENCY_PATTERNS:
        if pattern.search(text):
            return code
    return default


def _detect_period(text):
    for period, pattern in _PERIOD_PATTERNS:
        if pattern.search(text):
            return period
    return None


def _guess_period(amount, currency):
    """Infers the pay period from the size of the number when the text doesn't say."""
    if currency == "PHP":
        if amount < 1000:
            return "hour" if amount < 150 else "day"
        return "month" if amount < 1_000_000 else "year"
    if amount < 150:
        return "hour"
    if amount < 20_000:
        return "month"
    return "year"


def _monthly_php(amount, currency, period, hours_per_week=None):
    if amount is None:
        return None
    rate = PHP_PER_UNIT.get(currency)
    factor = TO_MONTHLY.get(period)
    if period == "hour" and hours_per_week:
        factor = hours_per_week * 52 / 12
    if rate is None or factor is None:
        return None
    return round(amount * factor * rate)


def build_salary(min_amount, max_amount, currency, period, text=None, is_estimate=False, hours_per_week=None):
    """Builds the normalized salary dict from already-known numbers."""
    if min_amount is None and max_amount is None:
        return None
    if min_amount is None:
        min_amount = max_amount
    if max_amount is None:
        max_amount = min_amount
    if min_amount > max_amount:
        min_amount, max_amount = max_amount, min_amount
    if max_amount <= 0:
        return None
    return {
        "text": text,
        "min": min_amount,
        "max": max_amount,
        "currency": currency,
        "period": period,
        "monthly_php_min": _monthly_php(min_amount, currency, period, hours_per_week),
        "monthly_php_max": _monthly_php(max_amount, currency, period, hours_per_week),
        "is_estimate": bool(is_estimate),
    }


def parse_salary_text(text, default_currency=None, hours_per_week=None):
    """
    Parses free-text salary like '₱18,000.00', '$4-6/hr', 'PHP 25k - 30k monthly'.

    With no currency symbol and no default, small amounts are treated as USD
    (OnlineJobs.ph employers are mostly overseas and post '5' or '5-7' for
    hourly dollar rates) and large amounts as PHP monthly pay.
    """
    if not text:
        return None
    text = text.strip()
    if not re.search(r"\d", text) or _NO_SALARY_RE.search(text) and not re.search(r"\d{3,}", text):
        return None

    numbers = []
    for raw, k in _NUMBER_RE.findall(text):
        value = _to_number(raw, bool(k))
        if value is not None and value > 0:
            numbers.append(value)
        if len(numbers) == 2:
            break
    if not numbers:
        return None

    low, high = min(numbers), max(numbers)
    # "25-30k" style: the k only applies to the last number
    if len(numbers) == 2 and high >= 1000 and low < 100 and high / low > 500:
        low *= 1000
    currency = _detect_currency(text, default_currency)
    if currency is None:
        currency = "USD" if high < 3000 else "PHP"
    period = _detect_period(text) or _guess_period(high, currency)
    return build_salary(low, high, currency, period, text=text[:80], hours_per_week=hours_per_week)


def detect_work_mode(job):
    """Remote / hybrid / onsite from explicit labels first, then text cues."""
    explicit = (job.get("work_mode_raw") or "").lower()
    location = job.get("location") or ""
    haystack = " ".join([explicit, job.get("title") or "", location, (job.get("description") or "")[:1500]])

    if _HYBRID_RE.search(haystack):
        return "hybrid"
    if _REMOTE_RE.search(" ".join([explicit, job.get("title") or "", location])):
        return "remote"
    if _REMOTE_RE.search(haystack) and not _ONSITE_RE.search(haystack):
        return "remote"
    if _ONSITE_RE.search(haystack) or location.strip():
        return "onsite"
    return None


def detect_employment_type(job):
    """Maps site-specific labels ('Permanent', 'Full Time', 'contract') to one vocabulary."""
    explicit = " ".join(filter(None, [job.get("employment_type_raw"), job.get("contract_time"), job.get("contract_type")]))
    for value in (explicit, job.get("title") or ""):
        if not value:
            continue
        for label, pattern in _TYPE_RULES:
            if pattern.search(value):
                return label
    # Description is the weakest signal — only trust the unambiguous labels
    description = (job.get("description") or "")[:1500]
    for label in ("internship", "part_time", "freelance"):
        if dict(_TYPE_RULES)[label].search(description):
            return label
    return None


def parse_date(text):
    """Returns 'YYYY-MM-DD' for the formats our sources use, else None."""
    if not text:
        return None
    text = re.sub(r"\s+", " ", str(text)).strip()
    iso = re.match(r"(\d{4}-\d{2}-\d{2})", text)
    if iso:
        return iso.group(1)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def enrich_job(job, default_source=None):
    """Returns a copy of the job with normalized, filterable attributes added."""
    enriched = dict(job)
    enriched.setdefault("source", default_source)

    salary = job.get("salary") if isinstance(job.get("salary"), dict) else None
    if salary is None and (job.get("salary_min") is not None or job.get("salary_max") is not None):
        # Adzuna: numeric, annual, in the market's own currency
        currency = job.get("salary_currency") or "GBP"
        salary = build_salary(job.get("salary_min"), job.get("salary_max"), currency, "year",
                              is_estimate=bool(job.get("salary_is_predicted")))
    if salary is None and job.get("salary_text"):
        hours = job.get("hours_per_week")
        hours = hours if isinstance(hours, (int, float)) and 0 < hours <= 80 else None
        salary = parse_salary_text(job["salary_text"], hours_per_week=hours)
    enriched["salary"] = salary

    enriched["work_mode"] = job.get("work_mode") or detect_work_mode(job)
    enriched["employment_type"] = job.get("employment_type") or detect_employment_type(job)
    enriched["posted_at"] = parse_date(job.get("posted_at") or job.get("posted_raw"))
    return enriched
