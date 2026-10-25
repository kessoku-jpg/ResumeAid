"""
Persistent SQLite cache for job scraper results.
Prevents repeated scraping across identical queries and sources.
"""

import json
import logging
import os
import re
import sqlite3
import time

logger = logging.getLogger("resumeaid.cache")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_THIS_DIR, ".cache")
_DEFAULT_DB_PATH = os.path.join(_CACHE_DIR, "job_cache.sqlite3")

# Default TTL: 12 hours (can be overridden via environment variable)
DEFAULT_TTL_HOURS = float(os.environ.get("JOB_CACHE_TTL_HOURS", "12.0"))


def _ensure_dir():
    """Ensures the cache directory exists."""
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _get_connection(db_path=None):
    """Returns a SQLite connection configured for concurrent safety."""
    _ensure_dir()
    path = db_path or _DEFAULT_DB_PATH
    conn = sqlite3.connect(path, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(db_path=None):
    """Creates the cache table and indices if they do not exist."""
    conn = _get_connection(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS job_cache (
                cache_key TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                query TEXT NOT NULL,
                country TEXT,
                jobs_json TEXT NOT NULL,
                result_count INTEGER NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_job_cache_lookup
            ON job_cache(source, query, country);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_job_cache_expires
            ON job_cache(expires_at);
        """)
    conn.close()


# Initialize database on module import
try:
    init_db()
except Exception as e:
    logger.warning("Could not initialize job cache database (%s)", e)


def normalize_query(query):
    """Collapses whitespace and strips punctuation for consistent caching."""
    if not query:
        return ""
    q = query.lower().strip()
    q = re.sub(r"[^\w\s-]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def make_cache_key(source, query, country=None):
    """Generates a canonical cache key."""
    norm_src = (source or "").strip().lower()
    norm_q = normalize_query(query)
    norm_country = (country or "").strip().lower()
    return f"{norm_src}:{norm_q}:{norm_country}"


def get_cached_jobs(source, query, country=None, max_age_hours=None, allow_stale=False, db_path=None):
    """
    Retrieves cached job listings for the given source and query.

    Args:
        source: Scraper identifier (e.g. 'philjobnet', 'onlinejobs', 'adzuna')
        query: Search keywords
        country: Optional country code (primarily for Adzuna)
        max_age_hours: Optional max age override in hours (defaults to TTL check)
        allow_stale: If True, returns expired cached data if nothing newer exists.

    Returns:
        dict: {
            "jobs": list,
            "created_at": float,
            "age_seconds": float,
            "is_stale": bool,
            "source": str,
        } or None if no valid cache entry exists.
    """
    key = make_cache_key(source, query, country)
    now = time.time()

    try:
        conn = _get_connection(db_path)
        with conn:
            cursor = conn.execute(
                "SELECT jobs_json, created_at, expires_at FROM job_cache WHERE cache_key = ?",
                (key,)
            )
            row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        jobs_json, created_at, expires_at = row
        age_seconds = max(0.0, now - created_at)

        # Check explicit max_age_hours if requested
        if max_age_hours is not None:
            max_age_sec = max_age_hours * 3600.0
            if age_seconds > max_age_sec and not allow_stale:
                return None

        is_stale = now > expires_at
        if is_stale and not allow_stale:
            return None

        jobs = json.loads(jobs_json)
        return {
            "jobs": jobs,
            "created_at": created_at,
            "age_seconds": age_seconds,
            "is_stale": is_stale,
            "source": source,
        }

    except Exception as e:
        logger.warning("Cache read failed for key '%s': %s", key, e)
        return None


def set_cached_jobs(source, query, jobs, country=None, ttl_hours=None, db_path=None):
    """
    Stores a list of job listings in the cache.

    Args:
        source: Scraper identifier ('philjobnet', 'onlinejobs', 'adzuna')
        query: Search keywords
        jobs: List of job dictionaries
        country: Optional country code
        ttl_hours: Time-to-live in hours (defaults to DEFAULT_TTL_HOURS)
    """
    if jobs is None:
        return

    key = make_cache_key(source, query, country)
    norm_q = normalize_query(query)
    norm_country = (country or "").strip().lower()
    ttl = float(ttl_hours if ttl_hours is not None else DEFAULT_TTL_HOURS)

    now = time.time()
    expires_at = now + (ttl * 3600.0)

    try:
        jobs_json = json.dumps(jobs, ensure_ascii=False)
        conn = _get_connection(db_path)
        with conn:
            conn.execute("""
                INSERT INTO job_cache (cache_key, source, query, country, jobs_json, result_count, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    jobs_json = excluded.jobs_json,
                    result_count = excluded.result_count,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at;
            """, (key, source, norm_q, norm_country, jobs_json, len(jobs), now, expires_at))
        conn.close()
        logger.debug("Cached %d jobs for key '%s' (TTL: %.1fh)", len(jobs), key, ttl)
    except Exception as e:
        logger.warning("Cache write failed for key '%s': %s", key, e)


def clear_expired(db_path=None):
    """Removes entries whose expires_at is in the past."""
    now = time.time()
    try:
        conn = _get_connection(db_path)
        with conn:
            cursor = conn.execute("DELETE FROM job_cache WHERE expires_at < ?", (now,))
            deleted = cursor.rowcount
        conn.close()
        return deleted
    except Exception as e:
        logger.warning("Failed to clear expired cache: %s", e)
        return 0


def clear_all(db_path=None):
    """Wipes the entire cache."""
    try:
        conn = _get_connection(db_path)
        with conn:
            cursor = conn.execute("DELETE FROM job_cache")
            deleted = cursor.rowcount
        conn.close()
        return deleted
    except Exception as e:
        logger.warning("Failed to clear cache: %s", e)
        return 0


def get_cache_stats(db_path=None):
    """Returns metadata about current cache usage."""
    try:
        conn = _get_connection(db_path)
        with conn:
            row = conn.execute("""
                SELECT COUNT(*), SUM(result_count), MIN(created_at), MAX(created_at)
                FROM job_cache
            """).fetchone()
        conn.close()
        count, total_jobs, min_t, max_t = row
        return {
            "entries_count": count or 0,
            "total_jobs_cached": total_jobs or 0,
            "oldest_timestamp": min_t,
            "newest_timestamp": max_t,
        }
    except Exception as e:
        logger.warning("Failed to get cache stats: %s", e)
        return {"entries_count": 0, "total_jobs_cached": 0}
