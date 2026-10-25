"""
Tests for backend/job_cache.py.
"""

import os
import tempfile
import time
import unittest

from job_cache import (
    init_db,
    normalize_query,
    make_cache_key,
    get_cached_jobs,
    set_cached_jobs,
    clear_expired,
    clear_all,
    get_cache_stats,
)


class TestJobCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.db_path = self.tmp.name
        self.tmp.close()
        init_db(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_normalize_query(self):
        self.assertEqual(normalize_query("  Python,   Developer! "), "python developer")
        self.assertEqual(normalize_query("Full-Stack / React"), "full-stack react")
        self.assertEqual(normalize_query(""), "")

    def test_make_cache_key(self):
        k1 = make_cache_key("PhilJobNet", "Python Developer", "PH")
        k2 = make_cache_key("philjobnet", "python  developer!", "ph")
        self.assertEqual(k1, k2)

    def test_set_and_get_cached_jobs(self):
        sample_jobs = [{"title": "Software Engineer", "company": "Acme", "url": "https://example.com"}]
        set_cached_jobs("onlinejobs", "software engineer", sample_jobs, db_path=self.db_path)

        res = get_cached_jobs("onlinejobs", "software engineer", db_path=self.db_path)
        self.assertIsNotNone(res)
        self.assertEqual(len(res["jobs"]), 1)
        self.assertEqual(res["jobs"][0]["title"], "Software Engineer")
        self.assertFalse(res["is_stale"])

    def test_ttl_expiration_and_allow_stale(self):
        sample_jobs = [{"title": "Data Analyst", "company": "Tech Corp"}]
        # Set with a very small TTL (0.0001 hours ~ 0.36 seconds)
        set_cached_jobs("philjobnet", "data analyst", sample_jobs, ttl_hours=0.0001, db_path=self.db_path)

        # Wait for expiration
        time.sleep(0.5)

        # Normal get returns None because it's expired
        res = get_cached_jobs("philjobnet", "data analyst", db_path=self.db_path)
        self.assertIsNone(res)

        # get with allow_stale=True returns the stale entry with flag
        stale_res = get_cached_jobs("philjobnet", "data analyst", allow_stale=True, db_path=self.db_path)
        self.assertIsNotNone(stale_res)
        self.assertTrue(stale_res["is_stale"])
        self.assertEqual(len(stale_res["jobs"]), 1)

    def test_clear_expired(self):
        sample_jobs = [{"title": "Designer"}]
        set_cached_jobs("adzuna", "designer", sample_jobs, ttl_hours=-1.0, db_path=self.db_path)
        cleared = clear_expired(db_path=self.db_path)
        self.assertEqual(cleared, 1)

        stats = get_cache_stats(db_path=self.db_path)
        self.assertEqual(stats["entries_count"], 0)


if __name__ == "__main__":
    unittest.main()
