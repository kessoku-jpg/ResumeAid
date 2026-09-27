"""
Semantic Matching Engine
------------------------
Compares a parsed resume against a job description and produces a
match score. The architecture has three distinct layers, kept
deliberately separate:

  1. DATA GROUNDING — before anything gets compared, both the resume
     and the job description are enriched using REAL external data:
       - O*NET's skill/tools taxonomy (skill_taxonomy.json), so a
         specific skill like "Java" is understood to also mean
         "computer skills" / "programming" — not because of a string
         match, but because O*NET's own government-maintained data
         says so.
       - O*NET's occupation profiles (occupation_profiles.json), so a
         job title like "Java Developer" pulls in that occupation's
         real, importance-ranked skills — useful when a scraped
         posting's actual text is thin or vague.
     Both are optional. If the JSON files aren't present, the engine
     still works — it just skips this enrichment step.

  2. SEMANTIC SCORING — the ONLY thing that produces the match score
     is Sentence-BERT embeddings compared with cosine similarity.
     There is NO keyword/substring matching anywhere in this layer.
     Matching runs at "fine-grained" granularity: every individual
     job requirement is compared against every individual resume
     segment, so a specific strong match doesn't get diluted by
     unrelated content sitting elsewhere in either text.

  3. EXPLANATION — a short human-readable comment describing the
     score. This layer MAY mention literal keyword overlaps (e.g.
     "you directly listed Python, which this posting also mentions")
     as supporting evidence for the person reading it — but this is
     cosmetic only. It never feeds back into the score itself.

Install dependencies first:
    pip install sentence-transformers

The first time you run this, it downloads the model (~90MB)
automatically — that only happens once, then it's cached.
"""

import json
import os
import re

from sentence_transformers import SentenceTransformer, util


# =======================================================================
# MODEL — loaded once at import time (loading it repeatedly is slow).
# 'all-MiniLM-L6-v2' is a good default: small, fast, and accurate
# enough for this kind of matching.
# =======================================================================

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# =======================================================================
# LAYER 1a: SKILL TAXONOMY — specific skill -> broader category(ies)
# =======================================================================
#
# Primary source: skill_taxonomy.json, built by running
# build_taxonomy_from_onet.py against O*NET's Software Skills / Tools
# Used files. Real, government-maintained data covering tens of
# thousands of specific skills/tools.
#
# Fallback: a small hand-typed dictionary below, which only fills in
# gaps O*NET doesn't cover (mainly generic phrasing like "computer
# skills" that O*NET's more specific category labels don't spell out).

_SKILL_TAXONOMY_PATH = os.path.join(_THIS_DIR, "skill_taxonomy.json")

try:
    with open(_SKILL_TAXONOMY_PATH, encoding="utf-8") as f:
        ONET_SKILL_TAXONOMY = json.load(f)
except FileNotFoundError:
    ONET_SKILL_TAXONOMY = {}

FALLBACK_SKILL_TAXONOMY = {
    "computer skills": [
        "java", "python", "c++", "c#", "javascript", "html", "css", "sql",
        "kotlin", "swift", "php", "typescript", "programming", "coding",
        "software development", "excel",
    ],
    "technical skills": [
        "java", "python", "c++", "c#", "javascript", "sql", "docker",
        "kubernetes", "aws", "azure", "git", "linux", "networking",
    ],
    "programming": [
        "java", "python", "c++", "c#", "javascript", "kotlin", "swift",
        "php", "typescript", "ruby", "go",
    ],
    "communication skills": [
        "public speaking", "presentation", "writing", "documentation",
        "customer service", "interpersonal skills",
    ],
    "leadership": [
        "team lead", "project management", "mentoring", "supervised",
        "managed a team", "led a team",
    ],
}


def expand_skills_with_categories(skills):
    """
    Given a list of resume skills, returns the ORIGINAL list plus
    broader category names implied by those skills. This is what lets
    the semantic layer "understand" that a specific skill implies a
    broader capability — the understanding comes from real data
    (O*NET), not from string matching against the job description.
    """
    skills_lower = {s.lower() for s in skills}
    implied = set()

    for skill in skills_lower:
        if skill in ONET_SKILL_TAXONOMY:
            implied.update(ONET_SKILL_TAXONOMY[skill])

    for category, related_terms in FALLBACK_SKILL_TAXONOMY.items():
        if category in implied:
            continue
        if any(term in skills_lower for term in related_terms):
            implied.add(category)

    return list(skills) + list(implied)


# =======================================================================
# LAYER 1b: OCCUPATION PROFILES — job title -> real important skills
# =======================================================================
#
# Loaded from occupation_profiles.json, built by running
# build_occupation_profiles.py against O*NET's Occupation Data /
# Alternate Titles / Skills / Knowledge / Abilities files.

_OCCUPATION_PROFILES_PATH = os.path.join(_THIS_DIR, "occupation_profiles.json")
_OCCUPATION_TITLE_CACHE_PATH = os.path.join(_THIS_DIR, "occupation_title_embeddings.pt")

try:
    with open(_OCCUPATION_PROFILES_PATH, encoding="utf-8") as f:
        OCCUPATION_PROFILES = json.load(f)
except FileNotFoundError:
    OCCUPATION_PROFILES = {}

# Minimum title-to-title similarity required before trusting an
# occupation match enough to inject its skills. Titles are short, so
# this is calibrated differently than the main document matching —
# print real similarity scores from find_best_occupation() on your own
# job titles and adjust this if it's rejecting good matches or
# accepting bad ones.
OCCUPATION_MATCH_THRESHOLD = 0.55


def _build_occupation_title_index():
    """
    Builds a flat list of every (title_text, soc_code) pair — the
    canonical title AND every alternate title for each occupation —
    plus their embeddings, so an incoming job title can be compared
    against all of them at once. Cached to disk after the first run
    since embedding thousands of titles takes a few seconds.
    """
    if not OCCUPATION_PROFILES:
        return [], None

    title_entries = []
    for soc_code, profile in OCCUPATION_PROFILES.items():
        title_entries.append((profile["title"], soc_code))
        for alt_title in profile.get("alt_titles", []):
            title_entries.append((alt_title, soc_code))

    if os.path.exists(_OCCUPATION_TITLE_CACHE_PATH):
        try:
            import torch
            cached = torch.load(_OCCUPATION_TITLE_CACHE_PATH, weights_only=False)
            if cached.get("count") == len(title_entries):
                return title_entries, cached["embeddings"]
        except Exception:
            pass  # cache unreadable/stale — rebuild below

    titles_only = [t[0] for t in title_entries]
    embeddings = model.encode(titles_only, convert_to_tensor=True, show_progress_bar=False)

    try:
        import torch
        torch.save({"count": len(title_entries), "embeddings": embeddings}, _OCCUPATION_TITLE_CACHE_PATH)
    except Exception:
        pass  # caching is a nice-to-have, not required

    return title_entries, embeddings


_OCCUPATION_TITLE_ENTRIES, _OCCUPATION_TITLE_EMBEDDINGS = _build_occupation_title_index()


def find_best_occupation(job_title):
    """
    Returns (soc_code, profile, similarity) for the O*NET occupation
    whose title/alternate-titles best match job_title, or None if no
    profiles are loaded or nothing clears OCCUPATION_MATCH_THRESHOLD.
    """
    if not job_title or _OCCUPATION_TITLE_EMBEDDINGS is None:
        return None

    query_embedding = model.encode(job_title, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, _OCCUPATION_TITLE_EMBEDDINGS)[0]

    best_idx = int(similarities.argmax())
    best_score = similarities[best_idx].item()
    if best_score < OCCUPATION_MATCH_THRESHOLD:
        return None

    _title_text, soc_code = _OCCUPATION_TITLE_ENTRIES[best_idx]
    return soc_code, OCCUPATION_PROFILES[soc_code], best_score


# =======================================================================
# RESUME -> JOB TITLE inference (the other direction from find_best_occupation)
# =======================================================================
#
# find_best_occupation goes: job title -> matching occupation.
# infer_job_title goes: resume -> likely occupation -> a search query.
# This is what makes job search actually automatic instead of requiring
# a manually-typed query — the resume's own skills/experience tell us
# what to search for.

_OCCUPATION_PROFILE_TEXT_CACHE_PATH = os.path.join(_THIS_DIR, "occupation_profile_text_embeddings.pt")

# Lower than OCCUPATION_MATCH_THRESHOLD on purpose: this compares a
# whole resume's worth of skills/experience text against a short
# "title + top skills" profile blurb — structurally different text
# lengths naturally score lower than title-vs-title comparisons even
# for a good match. Print real scores via infer_job_title() on your
# own resumes and adjust this if it's too strict/loose.
JOB_TITLE_INFERENCE_THRESHOLD = 0.25


def _build_occupation_profile_text_index():
    """
    Builds one embedding PER OCCUPATION (not per alt-title like the
    index above) representing "what this occupation is" — its title
    plus its top distinctive skills, e.g. "Software Developers.
    Skills: Programming, Systems Analysis, Technology Design." A
    resume's skills get compared against these to find the closest
    occupation. Cached to disk after the first run, same reasoning as
    _build_occupation_title_index above.
    """
    if not OCCUPATION_PROFILES:
        return [], None

    soc_codes = list(OCCUPATION_PROFILES.keys())
    profile_texts = []
    for soc_code in soc_codes:
        profile = OCCUPATION_PROFILES[soc_code]
        elements_text = ", ".join(profile.get("top_elements", []))
        profile_texts.append(f"{profile['title']}. Skills: {elements_text}")

    if os.path.exists(_OCCUPATION_PROFILE_TEXT_CACHE_PATH):
        try:
            import torch
            cached = torch.load(_OCCUPATION_PROFILE_TEXT_CACHE_PATH, weights_only=False)
            if cached.get("count") == len(soc_codes):
                return soc_codes, cached["embeddings"]
        except Exception:
            pass

    embeddings = model.encode(profile_texts, convert_to_tensor=True, show_progress_bar=False)

    try:
        import torch
        torch.save({"count": len(soc_codes), "embeddings": embeddings}, _OCCUPATION_PROFILE_TEXT_CACHE_PATH)
    except Exception:
        pass

    return soc_codes, embeddings


_OCCUPATION_PROFILE_SOC_CODES, _OCCUPATION_PROFILE_TEXT_EMBEDDINGS = _build_occupation_profile_text_index()


def infer_job_title(parsed_resume):
    """
    Given a parsed resume, guesses the most likely job title to search
    for — WITHOUT the person having to type one. Embeds the resume's
    content and finds the closest-matching O*NET occupation profile.
    Returns (title, similarity_score), or None if no profiles are
    loaded or nothing clears JOB_TITLE_INFERENCE_THRESHOLD (a resume
    too thin/unclear to confidently guess from).

    Builds its query text via build_resume_segments() — the SAME
    function used for actual scoring — rather than assembling a
    separate text blob. This matters: an earlier version fell back to
    the noisy COMBINED "skills" field (explicit + noun-chunk fallback
    candidates) whenever a resume had no dedicated Skills section,
    which reintroduced exactly the noise build_resume_segments()
    already excludes for scoring (stray words like "day", "Main
    Campus", "January" pulled from body text). That caused a real
    resume with strong HR-specific vocabulary in its experience
    section (buried among the noise) to get misread as "Telemarketers"
    instead of an HR-related occupation. Reusing build_resume_segments
    guarantees inference sees exactly the same trustworthy content the
    real matching does — nothing more, nothing noisier.
    """
    if _OCCUPATION_PROFILE_TEXT_EMBEDDINGS is None:
        return None

    resume_segments = build_resume_segments(parsed_resume)
    if not resume_segments:
        return None

    query_text = ". ".join(resume_segments)

    query_embedding = model.encode(query_text, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, _OCCUPATION_PROFILE_TEXT_EMBEDDINGS)[0]

    best_idx = int(similarities.argmax())
    best_score = similarities[best_idx].item()
    if best_score < JOB_TITLE_INFERENCE_THRESHOLD:
        return None

    soc_code = _OCCUPATION_PROFILE_SOC_CODES[best_idx]
    return OCCUPATION_PROFILES[soc_code]["title"], best_score


def expand_job_description_with_occupation(job_title, job_description):
    """
    If job_title confidently matches a known O*NET occupation, appends
    that occupation's top skills as an extra line — enriching thin
    postings with real, external substance before matching runs.
    Returns job_description unchanged if no confident match is found.

    Joined with " / " rather than ", " on purpose: split_into_requirements
    splits on commas to get atomic requirement chunks, and a naive
    comma-join here would let one profile with 10-15 elements explode
    into 10-15 SEPARATE requirement chunks — heavily outweighing the
    job posting's own actual requirements in the averaged score, even
    after the generic-element filtering done in
    build_occupation_profiles.py. Using " / " keeps this enrichment as
    ONE proportionate chunk, contributing one data point to the
    average like everything else, rather than dominating it.
    """
    match = find_best_occupation(job_title)
    if match is None:
        return job_description

    _soc_code, profile, _score = match
    top_elements = profile.get("top_elements", [])
    if not top_elements:
        return job_description

    enrichment = "Related occupational skills: " + " / ".join(top_elements)
    return job_description + "\n" + enrichment


# =======================================================================
# LAYER 2: SEMANTIC SCORING — embeddings + cosine similarity ONLY.
# No keyword/substring matching happens anywhere below this point.
# =======================================================================

def _is_noise_segment(segment):
    """
    Filters resume segments that are pure extraction artifacts (bare
    numbers, number ranges, stray bullets) WITHOUT dropping real short
    skills like "C". This is a data-cleanliness filter, not a
    matching mechanism — it runs before any embedding happens.
    """
    stripped = segment.strip(" -•*\t")
    if not stripped:
        return True
    if re.fullmatch(r'\d+(\s*(and|to|-|–|—)\s*\d+)?', stripped, re.IGNORECASE):
        return True
    return False


def build_resume_segments(parsed_resume):
    """
    Breaks the resume into a list of separate short segments used as
    matching candidates. Two sources, deliberately NOT treated equally:

      1. explicit_skills (preferred) — items from an actual Skills
         section. Unambiguous: the candidate wrote "Python" as a
         skill, so "Python" as a segment means Python-the-skill.
         Falls back to the older combined "skills" field if a resume
         was parsed with an earlier version of resume_parser.py that
         doesn't provide explicit_skills separately.

      2. Full sentences from experience/other sections. Kept as
         WHOLE LINES, not broken into individual words, specifically
         to preserve context — "updated employee database" as one
         segment can't be confused with "SQL databases" the way the
         bare word "database" alone can.

      inline_skills (the noisy noun-chunk fallback candidates) are
      DELIBERATELY EXCLUDED from scoring. They're useful for display
      ("skills" combines everything for that purpose) but not for
      matching: a context-free fragment like "database" pulled out of
      a sentence about employee records can score misleadingly high
      against an unrelated technical requirement like "SQL databases"
      just from sharing a root word, with none of the context that
      would show they mean different things. Real bug found via
      debug_match() — an HR resume's stray "database" fragment was
      outscoring an actual programmer's resume on a job requiring SQL.
    """
    segments = []

    explicit_skills = parsed_resume.get("explicit_skills")
    if explicit_skills is None:
        explicit_skills = parsed_resume.get("skills", [])  # backward compatibility

    if explicit_skills:
        expanded_skills = expand_skills_with_categories(explicit_skills)
        segments.extend(s for s in expanded_skills if not _is_noise_segment(s))

    text_blocks = []
    if parsed_resume.get("experience"):
        text_blocks.append(parsed_resume["experience"])
    covered_keys = {"skills", "explicit_skills", "inline_skills", "education", "experience"}
    for section_name, content in parsed_resume.get("raw_sections", {}).items():
        if section_name in ("header",) or not content or section_name in covered_keys:
            continue
        text_blocks.append(content)

    for block in text_blocks:
        for line in block.split("\n"):
            line = line.strip(" -\t•")
            if line and len(line) > 3 and not _is_noise_segment(line):
                segments.append(line)

    return segments


def split_into_requirements(job_description):
    """
    Splits a job description into individual requirement-sized chunks
    instead of treating it as one block — the other half of
    fine-grained matching.

    Two-stage split: first on sentence/bullet boundaries, THEN on
    commas. Splitting only on sentences isn't fine enough — a real
    sentence like "requiring Python experience, SQL databases, and
    Docker" bundles three separate technologies into one chunk, which
    dilutes the embedding so even a perfect match on one of them only
    scores moderately. Comma-splitting breaks these into near-atomic
    pieces so each technology can be judged on its own.
    """
    sentence_parts = re.split(r'[.\n;•]+', job_description)

    chunks = []
    for sentence in sentence_parts:
        for piece in sentence.split(","):
            cleaned = piece.strip()
            cleaned = re.sub(r'^(and|or|with|using|including)\s+', '', cleaned, flags=re.IGNORECASE)
            if len(cleaned) > 3:
                chunks.append(cleaned)

    return chunks


def compute_semantic_match_score(resume_segments, job_description, return_details=False):
    """
    THE core scoring function. Embeds every job requirement chunk and
    every resume segment, then for each requirement keeps only its
    single best-matching resume segment (cosine similarity). The
    final score blends the AVERAGE of those per-requirement best
    matches with the PEAK (single best) match.

    Why blend instead of just averaging: a plain average unfairly
    punishes a job with few requirement chunks when one of those
    chunks has nothing in the resume to match — one strong, specific
    match (e.g. a literal Python skill against a Python requirement)
    can get dragged down to a mediocre overall score by an unrelated
    requirement sitting right next to it. Meanwhile a vague job
    description with uniformly mediocre-but-consistent chunks can
    score just as high through sheer consistency, without ever having
    a genuinely strong match. Blending in the peak score rewards
    resumes that have at least one clearly strong, specific fit,
    without ignoring overall coverage entirely.

    Nothing here does string/substring matching. This is pure
    embedding similarity.

    If return_details=True, also returns a list of dicts (one per
    requirement) showing which resume segment matched it best and at
    what score — use this for debugging via debug_match(), not in
    normal scoring calls.
    """
    requirements = split_into_requirements(job_description)
    if not requirements or not resume_segments:
        return (0.0, []) if return_details else 0.0

    requirement_embeddings = model.encode(requirements, convert_to_tensor=True)
    segment_embeddings = model.encode(resume_segments, convert_to_tensor=True)

    similarity_grid = util.cos_sim(requirement_embeddings, segment_embeddings)
    best_per_requirement = similarity_grid.max(dim=1).values
    best_segment_idx_per_requirement = similarity_grid.argmax(dim=1)

    average_score = best_per_requirement.mean().item()
    peak_score = best_per_requirement.max().item()
    blended_score = (average_score + peak_score) / 2

    percentage = round(max(0, min(100, blended_score * 100)), 1)

    if not return_details:
        return percentage

    details = []
    for i, requirement in enumerate(requirements):
        segment_idx = best_segment_idx_per_requirement[i].item()
        details.append({
            "requirement": requirement,
            "best_segment": resume_segments[segment_idx],
            "score": round(best_per_requirement[i].item() * 100, 1),
        })
    return percentage, details


# =======================================================================
# LAYER 3: EXPLANATION — cosmetic only. Never feeds the score.
# =======================================================================

def generate_match_comment(score, resume_skills, job_description):
    """
    Produces a short explanation for the score. May mention literal
    skill overlaps as supporting evidence for the reader (using
    word-boundary matching, NOT plain substring — a plain substring
    check would wrongly "match" a skill like "C" against a word like
    "containerization"). This is display text only; it has no
    influence on the score computed above.
    """
    matched_skills = [
        skill for skill in resume_skills
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', job_description.lower())
    ]

    if score >= 80:
        band_comment = "Strong match — this candidate's background aligns closely with the role."
    elif score >= 60:
        band_comment = "Good match — solid overlap with several role requirements."
    elif score >= 40:
        band_comment = "Partial match — some relevant background, but notable gaps remain."
    else:
        band_comment = "Weak match — limited overlap with this role's requirements."

    if matched_skills:
        skills_note = f" Directly mentioned skills that match the listing: {', '.join(matched_skills)}."
    else:
        skills_note = ""

    return band_comment + skills_note, matched_skills


def compute_breakdown_scores(parsed_resume, job_description, job_title=None):
    """
    Computes per-dimension semantic similarity sub-scores for display.
    Returns a dict with:
      - skills_score     : cosine sim between resume skills text and job description (0-100)
      - experience_score : cosine sim between resume experience text and job description (0-100)
      - title_score      : cosine sim between the job title and the inferred resume occupation (0-100)
      - matched_skills   : list of skill strings that appear literally in the job description

    These scores are DISPLAY-ONLY — they do not feed back into the main
    match_score in any way.
    """
    skills = parsed_resume.get("explicit_skills") or parsed_resume.get("skills") or []
    experience = parsed_resume.get("experience") or ""

    matched_skills = [
        s for s in skills
        if re.search(r'\b' + re.escape(s.lower()) + r'\b', job_description.lower())
    ]

    # Skills score — embed the resume's skill list vs the job description
    if skills:
        expanded = expand_skills_with_categories(skills)
        skills_text = ", ".join(expanded)
        skills_emb = model.encode(skills_text, convert_to_tensor=True)
        jd_emb = model.encode(job_description, convert_to_tensor=True)
        skills_score = round(max(0, min(100, util.cos_sim(skills_emb, jd_emb).item() * 100)), 1)
    else:
        skills_score = 0.0

    # Experience score — embed the resume's experience vs the job description
    if experience.strip():
        exp_emb = model.encode(experience, convert_to_tensor=True)
        jd_emb2 = model.encode(job_description, convert_to_tensor=True)
        experience_score = round(max(0, min(100, util.cos_sim(exp_emb, jd_emb2).item() * 100)), 1)
    else:
        experience_score = 0.0

    # Title score — how well the job title aligns with the inferred occupation from the resume
    title_score = 0.0
    if job_title:
        resume_segments = build_resume_segments(parsed_resume)
        if resume_segments:
            resume_text = ". ".join(resume_segments)
            resume_emb = model.encode(resume_text, convert_to_tensor=True)
            title_emb = model.encode(job_title, convert_to_tensor=True)
            title_score = round(max(0, min(100, util.cos_sim(resume_emb, title_emb).item() * 100)), 1)

    return {
        "skills_score": skills_score,
        "experience_score": experience_score,
        "title_score": title_score,
        "matched_skills": matched_skills,
    }


# =======================================================================
# PUBLIC API
# =======================================================================

def match_resume_to_job(parsed_resume, job_description, job_title=None):
    """
    Main entry point. Give it a parsed resume dict (from
    resume_parser.py), a job description string, and optionally the
    job's title, and get back a score, explanation, and breakdown.
    """
    expanded_description = expand_job_description_with_occupation(job_title, job_description)
    resume_segments = build_resume_segments(parsed_resume)

    score = compute_semantic_match_score(resume_segments, expanded_description)
    comment, _matched = generate_match_comment(score, parsed_resume.get("skills", []), expanded_description)
    breakdown = compute_breakdown_scores(parsed_resume, expanded_description, job_title)

    return {"match_score": score, "comment": comment, "breakdown": breakdown}


def rank_jobs_for_resume(parsed_resume, job_listings):
    """
    Given a parsed resume and a list of job listings (each a dict
    with at least a 'description' key, ideally 'title' too), returns
    them ranked by match score, highest first.

    Returns TWO scores per job plus a breakdown:
      - match_score: the semantic matching percentage
      - relative_score: that job's score rescaled against the OTHER
        jobs in this batch (best -> ~100%, worst -> ~0%). Raw cosine
        similarity between any two pieces of normal professional
        English has a noise floor — even an unrelated resume/job pair
        tends to land around 20-40% — so relative_score is what
        actually differentiates good fits from bad ones WITHIN one
        batch. Use match_score if you need a standalone number
        instead.
      - breakdown: per-dimension scores (skills, experience, title)
        and matched_skills list — for frontend visualization only.
    """
    resume_segments = build_resume_segments(parsed_resume)
    results = []

    for job in job_listings:
        expanded_description = expand_job_description_with_occupation(
            job.get("title"), job["description"]
        )
        score = compute_semantic_match_score(resume_segments, expanded_description)
        comment, _matched = generate_match_comment(
            score, parsed_resume.get("skills", []), expanded_description
        )
        breakdown = compute_breakdown_scores(parsed_resume, expanded_description, job.get("title"))
        results.append({**job, "match_score": score, "comment": comment, "breakdown": breakdown})

    raw_scores = [r["match_score"] for r in results]
    min_score, max_score = min(raw_scores), max(raw_scores)
    score_range = max_score - min_score

    for r in results:
        if score_range > 0.01:
            r["relative_score"] = round((r["match_score"] - min_score) / score_range * 100, 1)
        else:
            r["relative_score"] = 50.0  # all jobs scored about the same

    return sorted(results, key=lambda r: r["match_score"], reverse=True)


def debug_match(parsed_resume, job_description, job_title=None):
    """
    Diagnostic tool: prints exactly which resume segment matched each
    job requirement chunk, and at what score, plus the overall
    blended score. Use this whenever a result looks surprising —
    it shows you exactly where the number came from.

    Example:
        from resume_parser import parse_resume
        from matching_engine import debug_match
        resume = parse_resume("my_resume.pdf")
        debug_match(resume, "Full-stack role requiring Python...")
    """
    expanded_description = expand_job_description_with_occupation(job_title, job_description)
    resume_segments = build_resume_segments(parsed_resume)

    score, details = compute_semantic_match_score(
        resume_segments, expanded_description, return_details=True
    )

    print(f"Overall match score: {score}%\n")
    print(f"{'Requirement':<55} {'Best resume match':<35} Score")
    print("-" * 100)
    for d in details:
        req_display = d["requirement"][:52] + "..." if len(d["requirement"]) > 55 else d["requirement"]
        seg_display = d["best_segment"][:32] + "..." if len(d["best_segment"]) > 35 else d["best_segment"]
        print(f"{req_display:<55} {seg_display:<35} {d['score']}%")

    return score


# =======================================================================
# Quick test runner
# =======================================================================

if __name__ == "__main__":
    sample_resume = {
        "skills": ["Python", "React", "SQL", "Docker", "Agile"],
        "education": "Cavite State University, BS Computer Science, 2023-2027",
        "experience": "Intern at TechCorp Manila. Led sprint planning using Agile "
                       "and Jira for a 4-person team. Built internal dashboards "
                       "with React and Python.",
        "raw_sections": {},
    }

    sample_jobs = [
        {
            "title": "Frontend Developer",
            "description": "Looking for a front-end developer experienced in React "
                            "and modern JavaScript frameworks. Agile team environment."
        },
        {
            "title": "Data Entry Clerk",
            "description": "Seeking a detail-oriented data entry clerk for manual "
                            "spreadsheet work. No programming experience required."
        },
        {
            "title": "Backend/Full-Stack Developer",
            "description": "Full-stack role requiring Python backend experience, "
                            "SQL databases, and containerization with Docker."
        },
    ]

    ranked = rank_jobs_for_resume(sample_resume, sample_jobs)

    print("Ranked matches for sample resume:\n")
    for job in ranked:
        print(f"{job['match_score']}% raw / {job['relative_score']}% relative  —  {job['title']}")
        print(f"   {job['comment']}\n")
