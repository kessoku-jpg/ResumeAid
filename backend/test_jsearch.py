"""Tests for the JSearch source (no network, no model downloads)."""

import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pipeline imports the parser/matcher, which load heavy NLP models; stub them
# so these tests only exercise source selection.
for name, attrs in {
    "resume_parser": ["parse_resume"],
    "matching_engine": ["rank_jobs_for_resume", "infer_job_title"],
}.items():
    if name not in sys.modules:
        stub = types.ModuleType(name)
        for attr in attrs:
            setattr(stub, attr, lambda *a, **k: None)
        sys.modules[name] = stub

import jsearch_scraper  # noqa: E402
import pipeline  # noqa: E402

SAMPLE_RESPONSE = {"status": "OK", "data": [
    {
        "job_title": "Python Developer", "employer_name": "Acme PH",
        "job_description": "Build APIs.", "job_apply_link": "https://example.com/apply",
        "job_city": "Makati", "job_state": "Metro Manila", "job_country": "PH",
        "job_is_remote": False, "job_employment_type": "FULLTIME",
        "job_min_salary": 40000, "job_max_salary": 60000,
        "job_salary_currency": "PHP", "job_salary_period": "MONTH",
        "job_posted_at_datetime_utc": "2026-11-01T08:00:00.000Z",
    },
    {"job_title": "", "employer_name": "Skipped"},
    {"job_title": "VA", "job_is_remote": True, "job_employment_type": "CONTRACTOR"},
]}


def fake_response(payload, status=200):
    r = mock.Mock(status_code=status)
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


class TestJSearchScraper(unittest.TestCase):
    def test_missing_key_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                jsearch_scraper.search_jobs("python")

    def test_maps_fields_and_skips_untitled(self):
        with mock.patch.dict(os.environ, {"JSEARCH_API_KEY": "k"}), \
             mock.patch("jsearch_scraper.requests.get", return_value=fake_response(SAMPLE_RESPONSE)) as get:
            jobs = jsearch_scraper.search_jobs("python developer", country="ph")
        self.assertEqual(len(jobs), 2)
        first = jobs[0]
        self.assertEqual(first["company"], "Acme PH")
        self.assertEqual(first["location"], "Makati, Metro Manila, PH")
        self.assertEqual(first["employment_type"], "full_time")
        self.assertEqual(first["salary"]["period"], "month")
        self.assertEqual(first["salary"]["currency"], "PHP")
        self.assertEqual(jobs[1]["work_mode"], "remote")
        self.assertEqual(jobs[1]["employment_type"], "contract")
        self.assertIsNone(jobs[1]["salary"])
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["country"], "ph")
        self.assertIn("Philippines", params["query"])

    def test_bad_key_message(self):
        with mock.patch.dict(os.environ, {"JSEARCH_API_KEY": "k"}), \
             mock.patch("jsearch_scraper.requests.get", return_value=fake_response({}, 403)):
            with self.assertRaises(RuntimeError):
                jsearch_scraper.search_jobs("python")


class TestPipelineSources(unittest.TestCase):
    def _sources_used(self, country, source="all"):
        called = []

        def make(name):
            def fn(*args, **kwargs):
                called.append(name)
                return [{"title": f"{name} job", "company": name, "url": f"https://{name}/1"}]
            return fn

        with mock.patch.object(pipeline, "_try_fetch_philjobnet", make("philjobnet")), \
             mock.patch.object(pipeline, "_try_fetch_onlinejobs", make("onlinejobs")), \
             mock.patch.object(pipeline, "_try_fetch_adzuna", make("adzuna")), \
             mock.patch.object(pipeline, "_try_fetch_jsearch", make("jsearch")), \
             mock.patch.object(pipeline, "CACHE_AVAILABLE", False):
            pipeline.get_job_listings("python developer", country=country, source=source)
        return sorted(called)

    def test_ph_adds_jsearch(self):
        self.assertEqual(self._sources_used("ph"), ["adzuna", "jsearch", "onlinejobs", "philjobnet"])

    def test_other_country_has_no_jsearch(self):
        self.assertEqual(self._sources_used("us"), ["adzuna", "onlinejobs", "philjobnet"])

    def test_explicit_jsearch_source(self):
        self.assertEqual(self._sources_used("ph", source="jsearch"), ["jsearch"])

    def test_explicit_adzuna_source_unchanged(self):
        self.assertEqual(self._sources_used("ph", source="adzuna"), ["adzuna"])


if __name__ == "__main__":
    unittest.main()
