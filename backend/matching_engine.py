"""
Semantic Matching Engine
------------------------
Compares a parsed resume against a job description using a 3-layer architecture:
1. DATA GROUNDING (O*NET enrichment)
2. SEMANTIC SCORING (Sentence-BERT embeddings)
3. EXPLANATION (Human-readable comments)
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
    """Expands resume skills with broader categories using O*NET data and fallbacks."""
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
    """Builds a cached flat list of (title, soc_code) pairs and their embeddings."""
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
    """Returns the best matching O*NET occupation for a job title, or None."""
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
    """Builds and caches one embedding per occupation representing its title and top skills."""
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
    """Infers the most likely job title to search for based on resume content."""
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
    """Appends top occupational skills to the job description if the title matches O*NET."""
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
    """Filters out noise segments (e.g., bare numbers) before embedding."""
    stripped = segment.strip(" -•*\t")
    if not stripped:
        return True
    if re.fullmatch(r'\d+(\s*(and|to|-|–|—)\s*\d+)?', stripped, re.IGNORECASE):
        return True
    return False


def build_resume_segments(parsed_resume):
    """Extracts resume segments for matching from explicit skills and full sentences."""
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
    """Splits job description into individual requirement chunks."""
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
    """Computes semantic match score using Sentence-BERT embeddings."""
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
    """Generates a human-readable explanation for the match score."""
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
    """Computes individual similarity scores for skills, experience, and title."""
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
    """Main entry point. Returns a match score, explanation, and breakdown."""
    expanded_description = expand_job_description_with_occupation(job_title, job_description)
    resume_segments = build_resume_segments(parsed_resume)

    score = compute_semantic_match_score(resume_segments, expanded_description)
    comment, _matched = generate_match_comment(score, parsed_resume.get("skills", []), expanded_description)
    breakdown = compute_breakdown_scores(parsed_resume, expanded_description, job_title)

    return {"match_score": score, "comment": comment, "breakdown": breakdown}


def rank_jobs_for_resume(parsed_resume, job_listings):
    """Ranks a list of job listings against a resume by match score."""
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
    """Diagnostic tool to inspect fine-grained matches and scores."""
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
