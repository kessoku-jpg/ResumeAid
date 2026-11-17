"""
Semantic Matching Engine
------------------------
Compares a parsed resume against a job description using a 3-layer architecture:
1. DATA GROUNDING (O*NET skill tree + occupation enrichment)
2. SCORING
   2a. Semantic (Sentence-BERT embeddings + cosine similarity)
   2b. Ontology-guided (resume & job mapped onto O*NET tree concepts,
       compared by Wu-Palmer similarity), fused with 2a
3. EXPLANATION (Human-readable comments + ontology links)
"""

import json
import os
import re

from sentence_transformers import SentenceTransformer, util

try:
    from session_logger import (
        get_active_trace,
        start_session,
        export_session_log,
        end_session,
        _vec_preview,
    )
except ImportError:
    def get_active_trace():
        return None

    def start_session(mode, metadata=None):
        return None

    def export_session_log(trace, engine_info):
        return None

    def end_session():
        pass

    def _vec_preview(tensor, n=6):
        return "None"


# =======================================================================
# MODEL — loaded once at import time (loading it repeatedly is slow).
# 'all-MiniLM-L6-v2' is a good default: small, fast, and accurate
# enough for this kind of matching.
# =======================================================================

MODEL_NAME = "all-MiniLM-L6-v2"


def _load_model(name):
    """
    Loads from the local Hugging Face cache first so startup works offline
    and doesn't ping the Hub. Only downloads (one-time, ~90 MB) if the
    model isn't cached yet.
    """
    try:
        return SentenceTransformer(name, local_files_only=True)
    except Exception:
        print(f"[matching_engine] '{name}' not cached locally — downloading (one-time, needs internet)...")
        return SentenceTransformer(name)


model = _load_model(MODEL_NAME)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# =======================================================================
# LAYER 1a: SKILL TREE — specific skill -> full O*NET ancestor path
# =======================================================================
#
# Primary source: skill_tree.json, built by build_skill_tree_from_onet.py
# from the O*NET 31.0 database in ../onet. It keeps O*NET's Content Model
# hierarchy (Domain -> Category -> Element -> Example, and for work
# activities GWA -> IWA -> DWA), so a resume skill expands into its whole
# path instead of a single flat category:
#
#   "jira" -> project management software -> business function specific
#             software -> software skills
#
# Skills that aren't an exact key in the tree are placed in it by
# embedding similarity against every node name (SKILL_TREE_SEMANTIC_*).
#
# Fallback: the small hand-typed dictionary below, which fills gaps for
# generic phrasing like "computer skills".

_SKILL_TREE_PATH = os.path.join(_THIS_DIR, "skill_tree.json")
_SKILL_TREE_EMBEDDINGS_PATH = os.path.join(_THIS_DIR, "skill_tree_embeddings.pt")

# Minimum cosine similarity for placing an unknown skill on a tree node,
# and how many nodes it may attach to. Skill names are short, so this is
# calibrated like OCCUPATION_MATCH_THRESHOLD rather than document scoring.
SKILL_TREE_SEMANTIC_THRESHOLD = 0.62
SKILL_TREE_SEMANTIC_TOP_K = 2
# Software "example" leaves are product names (e.g. "Chef" is DevOps
# software, not cooking), so semantic hits on them need to be stronger.
SKILL_TREE_SEMANTIC_EXAMPLE_THRESHOLD = 0.75

# Depth 0 = domain ("Software Skills", "Knowledge"...). Domains are very
# generic, so they're only added for exact hits, never semantic ones.
_SKILL_TREE_SEMANTIC_MIN_DEPTH = 1

try:
    with open(_SKILL_TREE_PATH, encoding="utf-8") as f:
        _skill_tree = json.load(f)
    SKILL_TREE_NODES = _skill_tree["nodes"]
    SKILL_TREE_INDEX = _skill_tree["index"]
    del _skill_tree
except FileNotFoundError:
    SKILL_TREE_NODES, SKILL_TREE_INDEX = {}, {}

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


def get_skill_tree_path(node_id, include_self=True):
    """Returns node names from node_id up to its domain root, lowercase."""
    path = []
    node = SKILL_TREE_NODES.get(node_id)
    if node and not include_self:
        parent_id = node.get("parent")
        node = SKILL_TREE_NODES.get(parent_id) if parent_id else None
    while node:
        path.append(node["name"].lower())
        parent_id = node.get("parent")
        node = SKILL_TREE_NODES.get(parent_id) if parent_id else None
    return path


_SKILL_TREE_NODE_IDS = None
_SKILL_TREE_NODE_EMBEDDINGS = None


def _get_skill_tree_embeddings():
    """Lazily builds/caches one embedding per tree node name (first call only)."""
    global _SKILL_TREE_NODE_IDS, _SKILL_TREE_NODE_EMBEDDINGS
    if _SKILL_TREE_NODE_IDS is not None:
        return _SKILL_TREE_NODE_IDS, _SKILL_TREE_NODE_EMBEDDINGS

    node_ids = [nid for nid, n in SKILL_TREE_NODES.items()
                if n.get("depth", 0) >= _SKILL_TREE_SEMANTIC_MIN_DEPTH]
    if not node_ids:
        _SKILL_TREE_NODE_IDS, _SKILL_TREE_NODE_EMBEDDINGS = [], None
        return _SKILL_TREE_NODE_IDS, _SKILL_TREE_NODE_EMBEDDINGS

    import torch
    if os.path.exists(_SKILL_TREE_EMBEDDINGS_PATH):
        try:
            cached = torch.load(_SKILL_TREE_EMBEDDINGS_PATH, weights_only=False)
            if cached.get("node_ids") == node_ids:
                _SKILL_TREE_NODE_IDS, _SKILL_TREE_NODE_EMBEDDINGS = node_ids, cached["embeddings"]
                return _SKILL_TREE_NODE_IDS, _SKILL_TREE_NODE_EMBEDDINGS
        except Exception:
            pass  # stale/unreadable cache — rebuild below

    print(f"[matching_engine] Embedding {len(node_ids)} skill tree nodes (one-time, cached)...")
    names = [SKILL_TREE_NODES[nid]["name"] for nid in node_ids]
    embeddings = model.encode(names, convert_to_tensor=True, show_progress_bar=False, batch_size=256)
    try:
        torch.save({"node_ids": node_ids, "embeddings": embeddings}, _SKILL_TREE_EMBEDDINGS_PATH)
    except Exception:
        pass

    _SKILL_TREE_NODE_IDS, _SKILL_TREE_NODE_EMBEDDINGS = node_ids, embeddings
    return _SKILL_TREE_NODE_IDS, _SKILL_TREE_NODE_EMBEDDINGS


# skill (lowercase) -> list of implied category names; avoids re-encoding
# the same skill for every resume / every call.
_SKILL_EXPANSION_CACHE = {}


def _normalize_skill(skill):
    return re.sub(r"\s+", " ", skill.lower()).strip(" .,;:-•*")


def map_skills_to_tree(skills):
    """
    Places each skill on the O*NET skill tree.
    Returns {skill_lower: [(node_id, score, how)]} where how is
    'exact' (alias lookup) or 'semantic' (embedding nearest neighbours).
    """
    placements = {}
    unknown = []
    for skill in skills:
        key = _normalize_skill(skill)
        if not key or key in placements:
            continue
        node_ids = SKILL_TREE_INDEX.get(key)
        if node_ids:
            placements[key] = [(nid, 1.0, "exact") for nid in node_ids]
        else:
            placements[key] = []
            unknown.append(key)

    if unknown:
        node_ids, node_embeddings = _get_skill_tree_embeddings()
        if node_embeddings is not None:
            query = model.encode(unknown, convert_to_tensor=True, show_progress_bar=False)
            sims = util.cos_sim(query, node_embeddings)
            k = min(SKILL_TREE_SEMANTIC_TOP_K, len(node_ids))
            top_scores, top_idx = sims.topk(k, dim=1)
            for row, key in enumerate(unknown):
                for score, idx in zip(top_scores[row].tolist(), top_idx[row].tolist()):
                    node_id = node_ids[idx]
                    threshold = (SKILL_TREE_SEMANTIC_EXAMPLE_THRESHOLD
                                 if SKILL_TREE_NODES[node_id]["type"] == "example"
                                 else SKILL_TREE_SEMANTIC_THRESHOLD)
                    if score >= threshold:
                        placements[key].append((node_id, round(score, 3), "semantic"))

    return placements


def _expand_with_skill_tree(skills_lower):
    implied = set()
    todo = [s for s in skills_lower if s not in _SKILL_EXPANSION_CACHE]
    if todo:
        for key, hits in map_skills_to_tree(todo).items():
            names = []
            for node_id, _score, how in hits:
                # Exact hit: the skill IS the node, so add its canonical
                # name + ancestors. Semantic hit: the node is a nearby
                # concept, so add it + ancestors (minus the generic domain).
                path = get_skill_tree_path(node_id)
                if how == "semantic":
                    path = path[:-1] or path
                names.extend(path)
            _SKILL_EXPANSION_CACHE[key] = list(dict.fromkeys(names))
    for key in skills_lower:
        implied.update(_SKILL_EXPANSION_CACHE.get(key, []))
    return implied


def expand_skills_with_categories(skills):
    """
    Expands resume skills with their O*NET skill-tree ancestors (exact alias
    hits + semantic placement of unknown skills) and generic fallback categories.
    """
    skills_lower = {_normalize_skill(s) for s in skills if s and s.strip()}
    skills_lower.discard("")

    implied = _expand_with_skill_tree(skills_lower) if SKILL_TREE_NODES else set()

    fallback_hits = []
    for category, related_terms in FALLBACK_SKILL_TAXONOMY.items():
        if category in implied:
            continue
        if any(term in skills_lower for term in related_terms):
            implied.add(category)
            fallback_hits.append(category)

    implied -= skills_lower  # don't duplicate skills already listed
    implied_sorted = sorted(implied)
    expanded = list(skills) + implied_sorted

    trace = get_active_trace()
    if trace is not None and not trace.skill_expansion:
        mappings_log = {}
        if SKILL_TREE_NODES and skills_lower:
            raw_map = map_skills_to_tree(sorted(skills_lower))
            for sk, hits in raw_map.items():
                mappings_log[sk] = [
                    {
                        "node_id": nid,
                        "node_name": SKILL_TREE_NODES[nid]["name"],
                        "node_type": SKILL_TREE_NODES[nid]["type"],
                        "depth": SKILL_TREE_NODES[nid].get("depth", 0),
                        "score": sc,
                        "how": how,
                        "ancestor_path": get_skill_tree_path(nid),
                    }
                    for nid, sc, how in hits
                ]
        trace.skill_expansion = {
            "input_skills": list(skills),
            "mappings": mappings_log,
            "fallback_hits": fallback_hits,
            "implied_categories": implied_sorted,
            "expanded_skills": expanded,
        }

    return expanded


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


_LAST_OCCUPATION_TRACE = {}


def find_best_occupation(job_title):
    """Returns the best matching O*NET occupation for a job title, or None."""
    global _LAST_OCCUPATION_TRACE
    if not job_title or _OCCUPATION_TITLE_EMBEDDINGS is None:
        _LAST_OCCUPATION_TRACE = {"matched": False, "job_title": job_title}
        return None

    query_embedding = model.encode(job_title, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, _OCCUPATION_TITLE_EMBEDDINGS)[0]

    best_idx = int(similarities.argmax())
    best_score = similarities[best_idx].item()
    title_text, soc_code = _OCCUPATION_TITLE_ENTRIES[best_idx]
    if best_score < OCCUPATION_MATCH_THRESHOLD:
        _LAST_OCCUPATION_TRACE = {
            "matched": False,
            "job_title": job_title,
            "best_candidate_title": title_text,
            "best_soc_code": soc_code,
            "best_score": round(best_score, 4),
        }
        return None

    profile = OCCUPATION_PROFILES[soc_code]
    _LAST_OCCUPATION_TRACE = {
        "matched": True,
        "job_title": job_title,
        "matched_title": title_text,
        "soc_code": soc_code,
        "occupation_title": profile.get("title", ""),
        "score": round(best_score, 4),
        "top_elements": profile.get("top_elements", []),
    }
    return soc_code, profile, best_score


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

    trace = get_active_trace()
    if trace is not None:
        top_k = min(5, len(_OCCUPATION_PROFILE_SOC_CODES))
        top_vals, top_idxs = similarities.topk(top_k)
        trace.occupation_inference["top_candidates"] = [
            {
                "soc_code": _OCCUPATION_PROFILE_SOC_CODES[int(idx)],
                "title": OCCUPATION_PROFILES[_OCCUPATION_PROFILE_SOC_CODES[int(idx)]]["title"],
                "score": round(float(val), 4),
            }
            for val, idx in zip(top_vals.tolist(), top_idxs.tolist())
        ]

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


# =======================================================================
# EDUCATION QUALIFICATION CHECK
# Jobs that explicitly require a certain degree (e.g. College Graduate /
# Bachelor's) only match if the resume meets that education level, and
# only those jobs include the resume's Education section in scoring.
# =======================================================================

EDU_NONE = 0
EDU_HIGH_SCHOOL = 1
EDU_VOCATIONAL_OR_UNDERGRAD = 2
EDU_BACHELOR = 3
EDU_POSTGRAD = 4

EDU_LABELS = {
    EDU_NONE: None,
    EDU_HIGH_SCHOOL: "High School Graduate",
    EDU_VOCATIONAL_OR_UNDERGRAD: "Vocational / College Undergraduate",
    EDU_BACHELOR: "Bachelor's / College Graduate",
    EDU_POSTGRAD: "Master's / Postgraduate",
}

_RESUME_POSTGRAD_RE = re.compile(
    r"\b(master\s+of|master'?s\b|m\.s\.|m\.a\.|\bmba\b|ph\.?d\.?\b|doctor\s+of|doctorate)\b",
    re.IGNORECASE,
)
_RESUME_BACHELOR_RE = re.compile(
    r"\b("
    r"bachelor|baccalaureate|b\.s\.|b\.a\.|"
    r"bs\s+(?:in\s+)?[a-z]|ba\s+(?:in\s+)?[a-z]|ab\s+(?:in\s+)?[a-z]|"
    r"bscs|bsit|bsis|bsba|bsa|bsn|bsee|bsme|bsce|bsece|bsed|beed|bshm|bstm|"
    r"college\s+graduate|university\s+graduate|degree\s+holder|"
    r"cum\s+laude|magna\s+cum\s+laude|summa\s+cum\s+laude"
    r")\b",
    re.IGNORECASE,
)
_RESUME_UNDERGRAD_MARKER_RE = re.compile(
    r"\b("
    r"undergraduate|college\s+level|currently\s+enrolled|ongoing|in\s+progress|"
    r"expected\s+graduation|expected\s+to\s+graduate|incomplete|"
    r"1st\s+year|2nd\s+year|3rd\s+year|4th\s+year|"
    r"first\s+year|second\s+year|third\s+year|fourth\s+year"
    r")\b|"
    r"\b20\d{2}\s*[-–—]\s*(?:present|current)\b",
    re.IGNORECASE,
)
_RESUME_VOCATIONAL_RE = re.compile(
    r"\b(associate|diploma|vocational|tesda|nc\s*[i1234v]+|national\s+certificate|college\s+level|undergraduate)\b",
    re.IGNORECASE,
)
_RESUME_HIGH_SCHOOL_RE = re.compile(
    r"\b(high\s+school|senior\s+high|junior\s+high|\bshs\b|\bjhs\b|secondary|k-?12|\bstem\b|\babm\b|\bhumss\b|\bgas\b|\btvl\b)\b",
    re.IGNORECASE,
)

# Job description patterns for required education level
_JOB_NO_DEGREE_RE = re.compile(
    r"\b(no\s+degree\s+required|degree\s+not\s+required|without\s+a\s+degree|regardless\s+of\s+educational|any\s+educational\s+attainment)\b",
    re.IGNORECASE,
)
_JOB_HIGH_SCHOOL_OK_RE = re.compile(
    r"\b(high\s+school\s+(?:or\s+vocational\s+)?graduate|senior\s+high\s+school|\bshs\s+graduate|at\s+least\s+high\s+school|open\s+to\s+high\s+school|high\s+school\s+diploma)\b",
    re.IGNORECASE,
)
_JOB_UNDERGRAD_OR_VOC_OK_RE = re.compile(
    r"\b("
    r"college\s+level|college\s+undergraduate|undergraduates?\s+(?:are\s+)?welcome|open\s+to\s+undergrad|"
    r"at\s+least\s+(?:1st|2nd|3rd|first|second|third)\s+year|completed\s+at\s+least\s+\d+\s+years?\s+in\s+college|"
    r"vocational\s+(?:or\s+high\s+school\s+)?graduate|associate'?s?\s+degree|tesda\s+[a-z0-9\s]*nc\s*[i1234v]+"
    r")\b",
    re.IGNORECASE,
)
_JOB_POSTGRAD_REQ_RE = re.compile(
    r"\b(master'?s\s+degree|master\s+of\s+[a-z]+|\bmba\s+required|ph\.?d\.?\s+required|doctorate\s+degree|post[- ]?graduate\s+degree)\b",
    re.IGNORECASE,
)
_JOB_BACHELOR_REQ_RE = re.compile(
    r"\b("
    r"bachelor'?s?(?:\s+degree|\s+of\b)|baccalaureate|b\.s\.\s+in|b\.a\.\s+in|"
    r"bs\s+in\s+[a-z]+|ba\s+in\s+[a-z]+|bs/ba\b|"
    r"college\s+graduate|university\s+graduate|degree\s+holder|college\s+degree|university\s+degree|"
    r"4[- ]year\s+(?:college\s+)?(?:course|degree)|four[- ]year\s+(?:college\s+)?(?:course|degree)|"
    r"graduate\s+of\s+(?:any\s+)?(?:a\s+)?(?:4[- ]year|four[- ]year|bachelor|college|bs\b|ba\b|ab\b)|"
    r"(?:accountancy|accounting|finance|business|marketing|engineering|computer\s+science|information\s+technology|"
    r"nursing|medical\s+technology|pharmacy|psychology|education|hospitality|tourism|architecture|hrm)\s+graduate|"
    r"licensed\s+(?:civil|mechanical|electrical|electronics|chemical)\s+engineer|"
    r"licensed\s+registered\s+nurse|registered\s+nurse\b|licensed\s+rmt\b|"
    r"\bcpa\b|certified\s+public\s+accountant|\blet\s+passer\b|licensed\s+professional\s+teacher"
    r")\b",
    re.IGNORECASE,
)


def detect_resume_education_level(parsed_resume):
    """
    Infers the candidate's highest completed education level from parsed_resume['education'].
    Returns (level_int, label_str).
    """
    edu_text = (parsed_resume.get("education") or "").strip()
    if not edu_text:
        return EDU_NONE, EDU_LABELS[EDU_NONE]

    if _RESUME_POSTGRAD_RE.search(edu_text):
        return EDU_POSTGRAD, EDU_LABELS[EDU_POSTGRAD]

    if _RESUME_BACHELOR_RE.search(edu_text):
        # Check each line: if a bachelor's line is NOT marked as ongoing/undergrad,
        # the candidate has completed a bachelor's degree.
        lines = [ln.strip() for ln in edu_text.split("\n") if ln.strip()]
        bachelor_lines = [ln for ln in lines if _RESUME_BACHELOR_RE.search(ln)] or [edu_text]
        if any(not _RESUME_UNDERGRAD_MARKER_RE.search(ln) for ln in bachelor_lines):
            return EDU_BACHELOR, EDU_LABELS[EDU_BACHELOR]
        return EDU_VOCATIONAL_OR_UNDERGRAD, EDU_LABELS[EDU_VOCATIONAL_OR_UNDERGRAD]

    if _RESUME_VOCATIONAL_RE.search(edu_text):
        return EDU_VOCATIONAL_OR_UNDERGRAD, EDU_LABELS[EDU_VOCATIONAL_OR_UNDERGRAD]

    if _RESUME_HIGH_SCHOOL_RE.search(edu_text):
        return EDU_HIGH_SCHOOL, EDU_LABELS[EDU_HIGH_SCHOOL]

    # Non-empty education section without specific keywords defaults to high school / undergrad
    return EDU_VOCATIONAL_OR_UNDERGRAD, EDU_LABELS[EDU_VOCATIONAL_OR_UNDERGRAD]


def detect_required_education(job_description, job_title=""):
    """
    Checks whether a job listing requires a specific education level.
    Returns (level_int, label_str). Returns (0, None) if the listing does not
    specify an education requirement.
    """
    text = f"{job_title or ''}\n{job_description or ''}"
    if not text.strip() or _JOB_NO_DEGREE_RE.search(text):
        return EDU_NONE, None

    # Explicit openness to high school or vocational/undergrad sets the floor
    if _JOB_HIGH_SCHOOL_OK_RE.search(text):
        return EDU_HIGH_SCHOOL, EDU_LABELS[EDU_HIGH_SCHOOL]
    if _JOB_UNDERGRAD_OR_VOC_OK_RE.search(text) and not _JOB_BACHELOR_REQ_RE.search(text):
        return EDU_VOCATIONAL_OR_UNDERGRAD, EDU_LABELS[EDU_VOCATIONAL_OR_UNDERGRAD]

    if _JOB_BACHELOR_REQ_RE.search(text):
        return EDU_BACHELOR, EDU_LABELS[EDU_BACHELOR]

    if _JOB_POSTGRAD_REQ_RE.search(text):
        return EDU_POSTGRAD, EDU_LABELS[EDU_POSTGRAD]

    return EDU_NONE, None


def _extract_education_segments(parsed_resume):
    """Returns cleaned lines from the resume's Education section."""
    edu_text = (parsed_resume.get("education") or "").strip()
    if not edu_text:
        return []
    segments = []
    for line in edu_text.split("\n"):
        line = line.strip(" -\t•")
        if line and len(line) > 3 and not _is_noise_segment(line):
            segments.append(line)
    return segments


def _resume_text_lines(parsed_resume, include_education=False):
    """Experience / other free-text section lines (everything except the skills list)."""
    text_blocks = []
    if parsed_resume.get("experience"):
        text_blocks.append(parsed_resume["experience"])
    if include_education and parsed_resume.get("education"):
        text_blocks.append(parsed_resume["education"])
    covered_keys = {"skills", "explicit_skills", "inline_skills", "education", "experience"}
    for section_name, content in parsed_resume.get("raw_sections", {}).items():
        if section_name in ("header",) or not content or section_name in covered_keys:
            continue
        text_blocks.append(content)

    lines = []
    for block in text_blocks:
        for line in block.split("\n"):
            line = line.strip(" -\t•")
            if line and len(line) > 3 and not _is_noise_segment(line):
                lines.append(line)
    return lines


def build_resume_segments(parsed_resume, include_education=False):
    """Extracts resume segments for matching from explicit skills and full sentences."""
    segments = []

    explicit_skills = parsed_resume.get("explicit_skills")
    if explicit_skills is None:
        explicit_skills = parsed_resume.get("skills", [])  # backward compatibility

    if explicit_skills:
        expanded_skills = expand_skills_with_categories(explicit_skills)
        segments.extend(s for s in expanded_skills if not _is_noise_segment(s))

    segments.extend(_resume_text_lines(parsed_resume, include_education))
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


# In-memory LRU-like caches for job requirements and job descriptions
_REQUIREMENT_EMBEDDINGS_CACHE = {}
_JD_EMBEDDINGS_CACHE = {}


def precompute_resume_embeddings(parsed_resume):
    """
    Pre-computes all Sentence-BERT embeddings for a parsed resume once.
    This turns matching from O(N) redundant encodes into an O(1) one-time encode,
    speeding up job ranking by up to 70-85%.
    """
    import torch

    resume_segments = build_resume_segments(parsed_resume, include_education=False)
    segment_embeddings = model.encode(resume_segments, convert_to_tensor=True) if resume_segments else None

    # Pre-compute education segment embeddings so jobs that require a degree
    # can include education in semantic scoring without re-encoding the rest.
    edu_segments = _extract_education_segments(parsed_resume)
    if edu_segments:
        edu_seg_emb = model.encode(edu_segments, convert_to_tensor=True)
        resume_segments_with_edu = resume_segments + edu_segments
        segment_embeddings_with_edu = (
            torch.cat([segment_embeddings, edu_seg_emb], dim=0)
            if segment_embeddings is not None else edu_seg_emb
        )
    else:
        resume_segments_with_edu = resume_segments
        segment_embeddings_with_edu = segment_embeddings

    edu_level, edu_label = detect_resume_education_level(parsed_resume)

    explicit_skills = parsed_resume.get("explicit_skills")
    if explicit_skills is None:
        explicit_skills = parsed_resume.get("skills", [])

    skills_emb = None
    if explicit_skills:
        expanded = expand_skills_with_categories(explicit_skills)
        skills_text = ", ".join(expanded)
        skills_emb = model.encode(skills_text, convert_to_tensor=True)

    experience = (parsed_resume.get("experience") or "").strip()
    exp_emb = None
    if experience:
        exp_emb = model.encode(experience, convert_to_tensor=True)

    resume_text_emb = None
    if resume_segments:
        resume_text = ". ".join(resume_segments)
        resume_text_emb = model.encode(resume_text, convert_to_tensor=True)

    ontology_concepts = extract_resume_concepts(parsed_resume)

    trace = get_active_trace()
    if trace is not None:
        trace.resume_encoding = {
            "edu_level": edu_level,
            "edu_label": edu_label,
            "resume_segments": list(resume_segments),
            "edu_segments": list(edu_segments),
            "segment_embeddings_info": _vec_preview(segment_embeddings),
            "segment_embeddings_with_edu_info": _vec_preview(segment_embeddings_with_edu),
            "skills_emb_info": _vec_preview(skills_emb),
            "exp_emb_info": _vec_preview(exp_emb),
            "resume_text_emb_info": _vec_preview(resume_text_emb),
            "ontology_concepts": [
                {
                    "node_id": nid,
                    "name": SKILL_TREE_NODES[nid]["name"],
                    "type": SKILL_TREE_NODES[nid]["type"],
                    "depth": _ontology_depth(nid),
                    "weight": info["weight"],
                    "how": info["how"],
                    "evidence": info["evidence"],
                }
                for nid, info in ontology_concepts.items()
                if nid in SKILL_TREE_NODES
            ],
        }

    return {
        "resume_segments": resume_segments,
        "segment_embeddings": segment_embeddings,
        "resume_segments_with_edu": resume_segments_with_edu,
        "segment_embeddings_with_edu": segment_embeddings_with_edu,
        "edu_level": edu_level,
        "edu_label": edu_label,
        "skills": explicit_skills,
        "skills_emb": skills_emb,
        "experience": experience,
        "exp_emb": exp_emb,
        "resume_text_emb": resume_text_emb,
        "ontology_concepts": ontology_concepts,
    }


def _get_requirement_embeddings(requirements):
    """Encodes job requirement chunks, reusing the in-memory cache."""
    req_cache_key = " || ".join(requirements)
    requirement_embeddings = _REQUIREMENT_EMBEDDINGS_CACHE.get(req_cache_key)
    if requirement_embeddings is None:
        requirement_embeddings = model.encode(requirements, convert_to_tensor=True)
        if len(_REQUIREMENT_EMBEDDINGS_CACHE) < 2000:
            _REQUIREMENT_EMBEDDINGS_CACHE[req_cache_key] = requirement_embeddings
    return requirement_embeddings


_LAST_SEMANTIC_TRACE = {}


def compute_semantic_match_score(resume_segments, job_description, return_details=False,
                                 precomputed_segment_embeddings=None):
    """
    Computes semantic match score using Sentence-BERT embeddings.
    Accepts precomputed_segment_embeddings to avoid re-encoding resume segments.
    """
    global _LAST_SEMANTIC_TRACE
    requirements = split_into_requirements(job_description)
    if not requirements or (not resume_segments and precomputed_segment_embeddings is None):
        _LAST_SEMANTIC_TRACE = {"num_segments": len(resume_segments or []), "requirements": [], "mean_cos": 0.0, "peak_cos": 0.0}
        return (0.0, []) if return_details else 0.0

    requirement_embeddings = _get_requirement_embeddings(requirements)

    if precomputed_segment_embeddings is not None:
        segment_embeddings = precomputed_segment_embeddings
    else:
        segment_embeddings = model.encode(resume_segments, convert_to_tensor=True)

    similarity_grid = util.cos_sim(requirement_embeddings, segment_embeddings)
    best_per_requirement = similarity_grid.max(dim=1).values
    best_segment_idx_per_requirement = similarity_grid.argmax(dim=1)

    average_score = best_per_requirement.mean().item()
    peak_score = best_per_requirement.max().item()
    blended_score = (average_score + peak_score) / 2

    percentage = round(max(0, min(100, blended_score * 100)), 1)

    details = []
    req_trace = []
    for i, requirement in enumerate(requirements):
        segment_idx = int(best_segment_idx_per_requirement[i].item())
        raw_cos = float(best_per_requirement[i].item())
        best_seg = resume_segments[segment_idx] if resume_segments and segment_idx < len(resume_segments) else ""
        sc = round(raw_cos * 100, 1)
        details.append({
            "requirement": requirement,
            "best_segment": best_seg,
            "score": sc,
        })
        req_trace.append({
            "requirement": requirement,
            "best_segment_idx": segment_idx,
            "best_segment": best_seg,
            "raw_cos": raw_cos,
            "score": sc,
        })

    _LAST_SEMANTIC_TRACE = {
        "num_segments": len(resume_segments or []),
        "requirements": req_trace,
        "mean_cos": average_score,
        "peak_cos": peak_score,
    }

    if not return_details:
        return percentage
    return percentage, details


# =======================================================================
# LAYER 2b: ONTOLOGY-GUIDED SCORING — O*NET skill tree as the ontology
# =======================================================================
#
# SBERT compares raw text. This layer instead maps BOTH the resume and the
# job onto concepts (nodes) of the O*NET skill tree and compares them by
# their position in the hierarchy:
#
#   1. Concept extraction  text -> tree nodes
#        exact:    the skill / phrase is a known alias of a node
#        semantic: SBERT nearest node above ONTOLOGY_CONCEPT_THRESHOLD
#   2. Concept similarity  Wu-Palmer on the tree
#        wup(a, b) = 2 * depth(LCA) / (depth(a) + depth(b))
#        (same node = 1.0, parent/child ~0.86, siblings 0.75 ...)
#   3. Ontology score      for every job concept, the best-matching
#        resume concept; weighted mean over job concepts (0-100)
#   4. Fusion              final = (1 - w) * SBERT + w * ontology
#
# Every ontology match carries its path ("Excel <-> spreadsheet
# proficiency via Spreadsheet software"), which makes scores explainable.

ONTOLOGY_WEIGHT = 0.30               # max share of the final score given to the ontology
ONTOLOGY_CONCEPT_THRESHOLD = 0.60    # free text -> concept (semantic)
ONTOLOGY_CONCEPT_TOP_K = 2
ONTOLOGY_MIN_LCA_DEPTH = 2           # must share at least a category (root=0, domain=1, category=2)
ONTOLOGY_ENRICHMENT_WEIGHT = 0.25    # O*NET occupation skills appended to the job count less
# The ontology's influence scales with how much of the job it recognised:
# full ONTOLOGY_WEIGHT once the job's own text yields this much concept weight.
ONTOLOGY_FULL_CONFIDENCE_WEIGHT = 3.0
ONTOLOGY_MAX_LINKS = 8

_ENRICHMENT_PREFIX = "Related occupational skills:"
_LEAD_IN_RE = re.compile(
    r"^(?:(?:must\s+(?:be|have)|should\s+have|has|have|with|a|an)\s+)?"
    r"(?:(?:strong|good|excellent|basic|advanced|working|solid|proven|hands[- ]on)\s+)?"
    r"(?:knowledge|experience|proficiency|proficient|skilled|skills?|familiarity|familiar|"
    r"expertise|background|competence|competent)\s+(?:in|with|of|on|using)\s+",
    re.IGNORECASE,
)
_PHRASE_TOKEN_RE = re.compile(r"[a-z0-9][\w+#]*(?:\.[a-z0-9]+)*")

_ONTOLOGY_ANCESTORS = {}
_ONTOLOGY_PHRASES = None
_ONTOLOGY_MAX_PHRASE_LEN = 6
_JOB_CONCEPTS_CACHE = {}
_ONTOLOGY_EXAMPLE_MASK = None


def _ontology_example_mask(node_ids):
    """Boolean tensor over embedded nodes: True for software product leaves."""
    global _ONTOLOGY_EXAMPLE_MASK
    if _ONTOLOGY_EXAMPLE_MASK is None or len(_ONTOLOGY_EXAMPLE_MASK) != len(node_ids):
        import torch
        _ONTOLOGY_EXAMPLE_MASK = torch.tensor(
            [SKILL_TREE_NODES[nid]["type"] == "example" for nid in node_ids], dtype=torch.bool
        )
    return _ONTOLOGY_EXAMPLE_MASK


def _ontology_ancestors(node_id):
    """Node id followed by all its ancestors up to the domain (nearest first)."""
    chain = _ONTOLOGY_ANCESTORS.get(node_id)
    if chain is None:
        chain, cur = [], node_id
        while cur:
            chain.append(cur)
            cur = SKILL_TREE_NODES[cur].get("parent")
        chain = tuple(chain)
        _ONTOLOGY_ANCESTORS[node_id] = chain
    return chain


def _ontology_depth(node_id):
    return SKILL_TREE_NODES[node_id].get("depth", 0) + 1  # virtual root = 0


def wu_palmer_similarity(node_a, node_b):
    """Wu-Palmer similarity of two tree nodes. Returns (similarity, lca_node_id)."""
    if node_a == node_b:
        return 1.0, node_a
    ancestors_b = set(_ontology_ancestors(node_b))
    lca = next((n for n in _ontology_ancestors(node_a) if n in ancestors_b), None)
    if lca is None:
        return 0.0, None
    lca_depth = _ontology_depth(lca)
    if lca_depth < ONTOLOGY_MIN_LCA_DEPTH:
        return 0.0, None  # only share a broad domain like "Software Skills"
    return 2 * lca_depth / (_ontology_depth(node_a) + _ontology_depth(node_b)), lca


def _ontology_phrase_index():
    """
    Aliases safe to spot inside free text: multi-word aliases of any node,
    and single-word aliases of skill/knowledge elements. Single-word product
    names ("Chef", "Go", "React") are excluded — inside a sentence they are
    usually ordinary words. They still match when they ARE the whole chunk.
    """
    global _ONTOLOGY_PHRASES
    if _ONTOLOGY_PHRASES is None:
        phrases = {}
        for alias, node_ids in SKILL_TREE_INDEX.items():
            n_tokens = len(alias.split())
            if n_tokens > _ONTOLOGY_MAX_PHRASE_LEN:
                continue
            keep = [nid for nid in node_ids
                    if n_tokens >= 2 or SKILL_TREE_NODES[nid]["type"] != "example"]
            if keep:
                phrases[alias] = keep
        _ONTOLOGY_PHRASES = phrases
    return _ONTOLOGY_PHRASES


def _scan_text_for_concepts(text):
    """Greedy longest-match scan of a sentence for known tree aliases."""
    phrases = _ontology_phrase_index()
    tokens = _PHRASE_TOKEN_RE.findall(text.lower())
    found, i = [], 0
    while i < len(tokens):
        step = 1
        for length in range(min(_ONTOLOGY_MAX_PHRASE_LEN, len(tokens) - i), 0, -1):
            node_ids = phrases.get(" ".join(tokens[i:i + length]))
            if node_ids:
                found.extend(node_ids)
                step = length
                break
        i += step
    return found


def _strip_lead_in(text):
    return _LEAD_IN_RE.sub("", _normalize_skill(text)).strip()


# Splits "proficient in spreadsheets and MS Office" into "spreadsheets",
# "ms office" so each skill mention can be mapped on its own.
_SUB_PHRASE_SPLIT_RE = re.compile(
    r"\s*(?:\band\b|\bor\b|\bin\b|\bwith\b|\busing\b|\bincluding\b|\bsuch as\b|"
    r"\blike\b|\bvia\b|\bthrough\b|\bfor\b|\bto\b|[/()&])\s*",
    re.IGNORECASE,
)


def _sub_phrases(text):
    whole = _strip_lead_in(text)
    phrases = []
    for part in _SUB_PHRASE_SPLIT_RE.split(text):
        part = _strip_lead_in(part)
        if part and len(part) > 2 and part != whole and part not in phrases:
            phrases.append(part)
    return phrases


def extract_ontology_concepts(texts, embeddings=None, weight=1.0):
    """
    Maps free-text segments onto tree concepts.
    Returns {node_id: {"weight": float, "evidence": text, "how": "exact"|"semantic"}}.

    Exact: the whole segment / a sub-phrase is a known alias, or a safe
    multi-word alias appears inside it. Semantic: SBERT nearest node for the
    segment and each sub-phrase — restricted to non-product nodes, because
    in free text product names ("Chef", "Go") are usually ordinary words.
    """
    concepts = {}

    def add(node_id, w, text, how):
        current = concepts.get(node_id)
        if current is None or w > current["weight"] + 1e-6:
            concepts[node_id] = {"weight": round(min(w, 1.0), 3), "evidence": text, "how": how}

    if embeddings is None:
        texts = [t for t in texts if t and t.strip()]
    if not texts or not SKILL_TREE_NODES:
        return concepts

    # units = (text to map, original segment used as evidence)
    phrase_units = []
    for text in texts:
        for unit in [_strip_lead_in(text)] + _sub_phrases(text):
            for node_id in SKILL_TREE_INDEX.get(unit, []):
                add(node_id, weight, text, "exact")
        for node_id in _scan_text_for_concepts(text):
            add(node_id, weight, text, "exact")
        phrase_units.extend((p, text) for p in _sub_phrases(text))

    node_ids, node_embeddings = _get_skill_tree_embeddings()
    if node_embeddings is None:
        return concepts

    import torch
    if embeddings is None:
        embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    unit_evidence = list(texts)
    if phrase_units:
        phrase_emb = model.encode([p for p, _ in phrase_units], convert_to_tensor=True, show_progress_bar=False)
        embeddings = torch.cat([embeddings, phrase_emb], dim=0)
        unit_evidence += [text for _, text in phrase_units]

    sims = util.cos_sim(embeddings, node_embeddings)
    # Product leaves are never semantic targets in free text
    sims[:, _ontology_example_mask(node_ids)] = -1.0
    top_scores, top_idx = sims.topk(min(ONTOLOGY_CONCEPT_TOP_K, len(node_ids)), dim=1)
    for row, text in enumerate(unit_evidence):
        for score, idx in zip(top_scores[row].tolist(), top_idx[row].tolist()):
            if score >= ONTOLOGY_CONCEPT_THRESHOLD:
                add(node_ids[idx], score * weight, text, "semantic")

    return concepts


def extract_resume_concepts(parsed_resume, include_education=False):
    """Resume -> tree concepts, from the skills list and from experience lines."""
    concepts = {}
    if not SKILL_TREE_NODES:
        return concepts

    explicit_skills = parsed_resume.get("explicit_skills")
    if explicit_skills is None:
        explicit_skills = parsed_resume.get("skills", [])
    original = {_normalize_skill(s): s for s in explicit_skills or []}
    for key, hits in map_skills_to_tree(list(original)).items():
        for node_id, score, how in hits:
            if node_id not in concepts or score > concepts[node_id]["weight"]:
                concepts[node_id] = {"weight": score, "evidence": original.get(key, key), "how": how}

    lines = _resume_text_lines(parsed_resume, include_education)
    for node_id, info in extract_ontology_concepts(lines).items():
        if node_id not in concepts or info["weight"] > concepts[node_id]["weight"]:
            concepts[node_id] = info
    return concepts


def extract_job_concepts(expanded_description):
    """Job description (incl. O*NET occupation enrichment line) -> tree concepts."""
    cached = _JOB_CONCEPTS_CACHE.get(expanded_description)
    if cached is not None:
        return cached

    requirements = split_into_requirements(expanded_description)
    plain = [r for r in requirements if not r.startswith(_ENRICHMENT_PREFIX)]
    plain_embeddings = None
    if plain:
        if len(plain) == len(requirements):
            plain_embeddings = _get_requirement_embeddings(requirements)
        else:
            plain_embeddings = model.encode(plain, convert_to_tensor=True, show_progress_bar=False)
    concepts = extract_ontology_concepts(plain, plain_embeddings)

    # O*NET occupation skills appended by expand_job_description_with_occupation
    for line in expanded_description.split("\n"):
        if line.startswith(_ENRICHMENT_PREFIX):
            for item in line[len(_ENRICHMENT_PREFIX):].split(" / "):
                for node_id in SKILL_TREE_INDEX.get(_normalize_skill(item), []):
                    if SKILL_TREE_NODES[node_id]["type"] == "example":
                        continue  # O*NET occupation top_elements are skills/knowledge, not software brands
                    if node_id not in concepts:
                        concepts[node_id] = {"weight": ONTOLOGY_ENRICHMENT_WEIGHT,
                                             "evidence": f"O*NET occupation profile: {item.strip()}",
                                             "how": "occupation"}

    if len(_JOB_CONCEPTS_CACHE) < 2000:
        _JOB_CONCEPTS_CACHE[expanded_description] = concepts
    return concepts


def _relation_label(job_node, resume_node, lca):
    if job_node == resume_node:
        return "same concept"
    if lca == job_node:
        return "resume skill is a specific kind of the requirement"
    if lca == resume_node:
        return "resume skill is broader than the requirement"
    return "related through a shared category"


_LAST_ONTOLOGY_TRACE = {}


def compute_ontology_score(resume_concepts, job_concepts):
    """
    Ontology-guided score (0-100): for each job concept, the best Wu-Palmer
    similarity to any resume concept (weighted by the confidence of both),
    averaged over job concepts.
    Returns (score or None, links, confidence 0-1).
    Confidence = how much concept weight the job's own text produced,
    relative to ONTOLOGY_FULL_CONFIDENCE_WEIGHT.
    """
    global _LAST_ONTOLOGY_TRACE
    if not job_concepts or not resume_concepts:
        _LAST_ONTOLOGY_TRACE = {"job_concepts": [], "total_weight": 0.0, "weighted_sum": 0.0, "own_text_weight": 0.0}
        return None, [], 0.0

    total_weight, weighted_sum, links = 0.0, 0.0, []
    own_text_weight = 0.0
    job_concept_traces = []
    for job_node, job_info in job_concepts.items():
        if job_info["how"] != "occupation":
            own_text_weight += job_info["weight"]
        best_sim, best_raw_wup, best_resume, best_lca = 0.0, 0.0, None, None
        for resume_node, resume_info in resume_concepts.items():
            raw_wup, lca = wu_palmer_similarity(job_node, resume_node)
            sim = raw_wup * resume_info["weight"]
            if sim > best_sim:
                best_sim, best_raw_wup, best_resume, best_lca = sim, raw_wup, resume_node, lca
        total_weight += job_info["weight"]
        weighted_sum += job_info["weight"] * best_sim
        jc_entry = {
            "job_node_id": job_node,
            "job_concept": SKILL_TREE_NODES[job_node]["name"],
            "job_type": SKILL_TREE_NODES[job_node]["type"],
            "job_depth": _ontology_depth(job_node),
            "job_weight": job_info["weight"],
            "job_how": job_info["how"],
            "job_evidence": job_info["evidence"],
            "best_resume_node_id": best_resume,
        }
        if best_resume is not None and best_lca is not None:
            rel = _relation_label(job_node, best_resume, best_lca)
            jc_entry.update({
                "best_resume_concept": SKILL_TREE_NODES[best_resume]["name"],
                "resume_depth": _ontology_depth(best_resume),
                "resume_weight": resume_concepts[best_resume]["weight"],
                "resume_evidence": resume_concepts[best_resume]["evidence"],
                "lca_node_id": best_lca,
                "lca_concept": SKILL_TREE_NODES[best_lca]["name"],
                "lca_depth": _ontology_depth(best_lca),
                "relation": rel,
                "raw_wup": best_raw_wup,
                "weighted_sim": best_sim,
                "contribution": job_info["weight"] * best_sim,
            })
            links.append({
                "job_concept": SKILL_TREE_NODES[job_node]["name"],
                "resume_concept": SKILL_TREE_NODES[best_resume]["name"],
                "shared_concept": SKILL_TREE_NODES[best_lca]["name"],
                "relation": rel,
                "similarity": round(best_sim * 100, 1),
                "job_evidence": job_info["evidence"],
                "resume_evidence": resume_concepts[best_resume]["evidence"],
                "_rank": job_info["weight"] * best_sim,
            })
        job_concept_traces.append(jc_entry)

    links.sort(key=lambda link: float(link.pop("_rank")), reverse=True)
    score = round(weighted_sum / total_weight * 100, 1) if total_weight else None
    confidence = round(min(1.0, own_text_weight / ONTOLOGY_FULL_CONFIDENCE_WEIGHT), 2)
    _LAST_ONTOLOGY_TRACE = {
        "job_concepts": job_concept_traces,
        "total_weight": total_weight,
        "weighted_sum": weighted_sum,
        "own_text_weight": own_text_weight,
    }
    return score, links[:ONTOLOGY_MAX_LINKS], confidence


def fuse_scores(semantic_score, ontology_score, confidence=1.0):
    """
    Final match score: SBERT score blended with the ontology score.
    The ontology's share is ONTOLOGY_WEIGHT * confidence, so jobs whose
    text maps to few O*NET concepts are scored mostly by SBERT.
    """
    if ontology_score is None or confidence <= 0:
        return semantic_score
    w = ONTOLOGY_WEIGHT * confidence
    return round((1 - w) * semantic_score + w * ontology_score, 1)


def _ontology_comment(links):
    shared = list(dict.fromkeys(link["shared_concept"] for link in links if link["similarity"] >= 50))[:3]
    if not shared:
        return ""
    return f" O*NET-related strengths: {', '.join(shared)}."


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


_LAST_BREAKDOWN_COS = {"skills": 0.0, "exp": 0.0, "title": 0.0}


def compute_breakdown_scores(parsed_resume, job_description, job_title=None,
                             precomputed_resume=None, precomputed_jd_emb=None) -> dict[str, object]:
    """
    Computes individual similarity scores for skills, experience, and title.
    Reuses pre-computed resume and job embeddings to eliminate redundant encodes.
    """
    global _LAST_BREAKDOWN_COS
    if precomputed_resume:
        skills = precomputed_resume.get("skills") or []
        experience = precomputed_resume.get("experience") or ""
    else:
        skills = parsed_resume.get("explicit_skills") or parsed_resume.get("skills") or []
        experience = (parsed_resume.get("experience") or "").strip()

    matched_skills = [
        s for s in skills
        if re.search(r'\b' + re.escape(s.lower()) + r'\b', job_description.lower())
    ]

    # Re-use or cache job description embedding once for all breakdown dimensions
    if precomputed_jd_emb is not None:
        jd_emb = precomputed_jd_emb
    else:
        jd_emb = _JD_EMBEDDINGS_CACHE.get(job_description)
        if jd_emb is None:
            jd_emb = model.encode(job_description, convert_to_tensor=True)
            if len(_JD_EMBEDDINGS_CACHE) < 2000:
                _JD_EMBEDDINGS_CACHE[job_description] = jd_emb

    raw_skills_cos = 0.0
    # Skills score
    if precomputed_resume and precomputed_resume.get("skills_emb") is not None:
        raw_skills_cos = float(util.cos_sim(precomputed_resume["skills_emb"], jd_emb).item())
        skills_score = round(max(0, min(100, raw_skills_cos * 100)), 1)
    elif skills:
        expanded = expand_skills_with_categories(skills)
        skills_text = ", ".join(expanded)
        skills_emb = model.encode(skills_text, convert_to_tensor=True)
        raw_skills_cos = float(util.cos_sim(skills_emb, jd_emb).item())
        skills_score = round(max(0, min(100, raw_skills_cos * 100)), 1)
    else:
        skills_score = 0.0

    raw_exp_cos = 0.0
    # Experience score — reusing the SAME jd_emb!
    if precomputed_resume and precomputed_resume.get("exp_emb") is not None:
        raw_exp_cos = float(util.cos_sim(precomputed_resume["exp_emb"], jd_emb).item())
        experience_score = round(max(0, min(100, raw_exp_cos * 100)), 1)
    elif experience:
        exp_emb = model.encode(experience, convert_to_tensor=True)
        raw_exp_cos = float(util.cos_sim(exp_emb, jd_emb).item())
        experience_score = round(max(0, min(100, raw_exp_cos * 100)), 1)
    else:
        experience_score = 0.0

    raw_title_cos = 0.0
    # Title score
    title_score = 0.0
    if job_title:
        title_emb = model.encode(job_title, convert_to_tensor=True)
        if precomputed_resume and precomputed_resume.get("resume_text_emb") is not None:
            raw_title_cos = float(util.cos_sim(precomputed_resume["resume_text_emb"], title_emb).item())
            title_score = round(max(0, min(100, raw_title_cos * 100)), 1)
        else:
            resume_segments = build_resume_segments(parsed_resume)
            if resume_segments:
                resume_text = ". ".join(resume_segments)
                resume_emb = model.encode(resume_text, convert_to_tensor=True)
                raw_title_cos = float(util.cos_sim(resume_emb, title_emb).item())
                title_score = round(max(0, min(100, raw_title_cos * 100)), 1)

    _LAST_BREAKDOWN_COS = {"skills": raw_skills_cos, "exp": raw_exp_cos, "title": raw_title_cos}

    return {
        "skills_score": skills_score,
        "experience_score": experience_score,
        "title_score": title_score,
        "matched_skills": matched_skills,
    }


def get_engine_info() -> dict[str, object]:
    """Returns computational model, ontology, and hyperparameter metadata for session logs."""
    embedded_count = (
        len(_SKILL_TREE_NODE_IDS)
        if _SKILL_TREE_NODE_IDS is not None
        else sum(1 for n in SKILL_TREE_NODES.values() if n.get("depth", 0) >= _SKILL_TREE_SEMANTIC_MIN_DEPTH)
    )
    return {
        "model_name": MODEL_NAME,
        "embedding_dim": 384,
        "tree_node_count": len(SKILL_TREE_NODES),
        "tree_alias_count": len(SKILL_TREE_INDEX),
        "tree_embedded_count": embedded_count,
        "occupation_count": len(OCCUPATION_PROFILES),
        "occupation_title_count": len(_OCCUPATION_TITLE_ENTRIES),
        "hyperparameters": {
            "OCCUPATION_MATCH_THRESHOLD": OCCUPATION_MATCH_THRESHOLD,
            "JOB_TITLE_INFERENCE_THRESHOLD": JOB_TITLE_INFERENCE_THRESHOLD,
            "SKILL_TREE_SEMANTIC_THRESHOLD": SKILL_TREE_SEMANTIC_THRESHOLD,
            "SKILL_TREE_SEMANTIC_EXAMPLE_THRESHOLD": SKILL_TREE_SEMANTIC_EXAMPLE_THRESHOLD,
            "SKILL_TREE_SEMANTIC_TOP_K": SKILL_TREE_SEMANTIC_TOP_K,
            "ONTOLOGY_WEIGHT": ONTOLOGY_WEIGHT,
            "ONTOLOGY_CONCEPT_THRESHOLD": ONTOLOGY_CONCEPT_THRESHOLD,
            "ONTOLOGY_CONCEPT_TOP_K": ONTOLOGY_CONCEPT_TOP_K,
            "ONTOLOGY_MIN_LCA_DEPTH": ONTOLOGY_MIN_LCA_DEPTH,
            "ONTOLOGY_ENRICHMENT_WEIGHT": ONTOLOGY_ENRICHMENT_WEIGHT,
            "ONTOLOGY_FULL_CONFIDENCE_WEIGHT": ONTOLOGY_FULL_CONFIDENCE_WEIGHT,
        },
    }


def _populate_trace_from_parsed_resume(trace, parsed_resume, source_type="parsed_resume_dict"):
    """Fills Stage 1 parsing telemetry on trace if it was not already populated by pipeline.py."""
    if trace is None or trace.parsing:
        return
    meta = parsed_resume.get("_parse_meta") or {}
    raw_sections = parsed_resume.get("raw_sections") or {}
    sections_summary = {
        k: {"chars": len(v or ""), "lines": len((v or "").splitlines())}
        for k, v in raw_sections.items()
    }
    trace.parsing = {
        "source_type": "file_upload" if meta.get("filepath") else source_type,
        "filepath": meta.get("filepath"),
        "file_ext": meta.get("file_ext", ""),
        "ocr_used": meta.get("ocr_used", False),
        "raw_char_count": meta.get("raw_char_count", 0),
        "raw_line_count": meta.get("raw_line_count", 0),
        "clean_char_count": meta.get("clean_char_count", 0),
        "clean_line_count": meta.get("clean_line_count", 0),
        "sections": sections_summary,
        "name": parsed_resume.get("name"),
        "email": parsed_resume.get("email"),
        "phone": parsed_resume.get("phone"),
        "orgs": parsed_resume.get("organizations_mentioned", []),
        "dates": parsed_resume.get("dates_mentioned", []),
        "explicit_skills": parsed_resume.get("explicit_skills") or parsed_resume.get("skills") or [],
        "inline_skills": parsed_resume.get("inline_skills", []),
        "experience": parsed_resume.get("experience") or "",
        "education": parsed_resume.get("education") or "",
    }


def _extract_semantic_score(result):
    """Unpacks percentage score whether compute_semantic_match_score returned a float or tuple."""
    return result[0] if isinstance(result, tuple) else result

def match_resume_to_job(parsed_resume, job_description, job_title=None):
    """Main entry point. Returns a match score, explanation, and breakdown."""
    standalone = get_active_trace() is None
    trace = start_session("single_job_match", {"Job Title": job_title or "Untitled"}) if standalone else get_active_trace()
    try:
        _populate_trace_from_parsed_resume(trace, parsed_resume, source_type="single_job_match")
        req_edu_level, req_edu_label = detect_required_education(job_description, job_title)
        precomputed = precompute_resume_embeddings(parsed_resume)

        if req_edu_level > 0 and precomputed["edu_level"] < req_edu_level:
            if trace is not None:
                trace.excluded_jobs.append({
                    "title": job_title or "Untitled",
                    "company": "",
                    "req_edu_level": req_edu_level,
                    "req_edu_label": req_edu_label,
                    "resume_edu_level": precomputed["edu_level"],
                })
            return {
                "match_score": 0.0,
                "comment": f"Not matched — this role requires {req_edu_label}.",
                "breakdown": {"skills_score": 0.0, "experience_score": 0.0, "title_score": 0.0, "matched_skills": [],
                              "semantic_score": 0.0, "ontology_score": None, "ontology_links": [],
                              "ontology_confidence": 0.0},
                "education_qualified": False,
                "required_education": req_edu_label,
            }

        expanded_description = expand_job_description_with_occupation(job_title, job_description)
        occ_trace = dict(_LAST_OCCUPATION_TRACE)
        use_edu = req_edu_level > 0
        segments = precomputed["resume_segments_with_edu"] if use_edu else precomputed["resume_segments"]
        seg_emb = precomputed["segment_embeddings_with_edu"] if use_edu else precomputed["segment_embeddings"]

        semantic_score = _extract_semantic_score(compute_semantic_match_score(
            segments,
            expanded_description,
            precomputed_segment_embeddings=seg_emb
        ))
        sem_trace = dict(_LAST_SEMANTIC_TRACE)
        ontology_score, ontology_links, ontology_confidence = compute_ontology_score(
            precomputed["ontology_concepts"], extract_job_concepts(expanded_description)
        )
        onto_trace = dict(_LAST_ONTOLOGY_TRACE)
        score = fuse_scores(semantic_score, ontology_score, ontology_confidence)
        comment, _matched = generate_match_comment(score, precomputed["skills"], expanded_description)
        comment += _ontology_comment(ontology_links)
        breakdown = compute_breakdown_scores(
            parsed_resume, expanded_description, job_title, precomputed_resume=precomputed
        )
        raw_cos = dict(_LAST_BREAKDOWN_COS)
        breakdown["semantic_score"] = semantic_score
        breakdown["ontology_score"] = ontology_score
        breakdown["ontology_links"] = ontology_links
        breakdown["ontology_confidence"] = ontology_confidence

        if trace is not None:
            eff_w = (ONTOLOGY_WEIGHT * ontology_confidence) if (ontology_score is not None and ontology_confidence > 0) else 0.0
            trace.job_traces.append({
                "title": job_title or "Untitled",
                "company": "",
                "source": "direct",
                "req_edu_level": req_edu_level,
                "req_edu_label": req_edu_label,
                "use_edu": use_edu,
                "occupation_match": occ_trace,
                "semantic_score": semantic_score,
                "semantic_details": sem_trace,
                "ontology_score": ontology_score,
                "ontology_confidence": ontology_confidence,
                "ontology_details": onto_trace,
                "fusion_details": {
                    "ontology_weight_const": ONTOLOGY_WEIGHT,
                    "confidence": ontology_confidence,
                    "effective_weight": eff_w,
                },
                "final_score": score,
                "relative_score": 100.0,
                "breakdown": breakdown,
                "raw_skills_cos": raw_cos["skills"],
                "raw_exp_cos": raw_cos["exp"],
                "raw_title_cos": raw_cos["title"],
                "comment": comment,
            })

        return {
            "match_score": score,
            "comment": comment,
            "breakdown": breakdown,
            "education_qualified": True,
            "required_education": req_edu_label,
        }
    finally:
        if standalone and trace is not None:
            trace.mark_stage("single_job_match")
            export_session_log(trace, get_engine_info())
            end_session()


def rank_jobs_for_resume(parsed_resume, job_listings):
    """
    Ranks a list of job listings against a resume by match score.
    Uses pre-computed resume embeddings and cached JD embeddings for high performance.
    Jobs that require a higher education level than the resume has are excluded,
    and jobs that require a degree include the resume's Education section in scoring.
    """
    standalone = get_active_trace() is None
    trace = start_session("rank_jobs_for_resume", {"Job Count": len(job_listings)}) if standalone else get_active_trace()
    try:
        _populate_trace_from_parsed_resume(trace, parsed_resume, source_type="rank_jobs_direct")
        precomputed = precompute_resume_embeddings(parsed_resume)
        if trace is not None:
            trace.mark_stage("resume_encoding_and_ontology")
        resume_segments = precomputed["resume_segments"]
        segment_emb = precomputed["segment_embeddings"]
        resume_segments_with_edu = precomputed["resume_segments_with_edu"]
        segment_emb_with_edu = precomputed["segment_embeddings_with_edu"]
        resume_edu_level = precomputed["edu_level"]
        skills = precomputed["skills"]

        results = []
        job_trace_By_idx = {}

        for idx, job in enumerate(job_listings):
            req_edu_level, req_edu_label = detect_required_education(
                job.get("description", ""), job.get("title", "")
            )
            # Exclude jobs requiring a higher education level than the candidate has
            if req_edu_level > 0 and resume_edu_level < req_edu_level:
                if trace is not None:
                    trace.excluded_jobs.append({
                        "title": job.get("title", "Untitled"),
                        "company": job.get("company", ""),
                        "req_edu_level": req_edu_level,
                        "req_edu_label": req_edu_label,
                        "resume_edu_level": resume_edu_level,
                    })
                continue

            expanded_description = expand_job_description_with_occupation(
                job.get("title"), job["description"]
            )
            occ_trace = dict(_LAST_OCCUPATION_TRACE)
            # Include education segments only when the job listing requires a degree
            use_edu = req_edu_level > 0
            semantic_score = _extract_semantic_score(compute_semantic_match_score(
                resume_segments_with_edu if use_edu else resume_segments,
                expanded_description,
                precomputed_segment_embeddings=(segment_emb_with_edu if use_edu else segment_emb)
            ))
            sem_trace = dict(_LAST_SEMANTIC_TRACE)
            ontology_score, ontology_links, ontology_confidence = compute_ontology_score(
                precomputed["ontology_concepts"], extract_job_concepts(expanded_description)
            )
            onto_trace = dict(_LAST_ONTOLOGY_TRACE)
            score = fuse_scores(semantic_score, ontology_score, ontology_confidence)
            comment, _matched = generate_match_comment(
                score, skills, expanded_description
            )
            comment += _ontology_comment(ontology_links)
            breakdown = compute_breakdown_scores(
                parsed_resume,
                expanded_description,
                job.get("title"),
                precomputed_resume=precomputed
            )
            raw_cos = dict(_LAST_BREAKDOWN_COS)
            breakdown["semantic_score"] = semantic_score
            breakdown["ontology_score"] = ontology_score
            breakdown["ontology_links"] = ontology_links
            breakdown["ontology_confidence"] = ontology_confidence
            res_entry = {
                **job,
                "match_score": score,
                "comment": comment,
                "breakdown": breakdown,
                "required_education": req_edu_label,
            }
            results.append(res_entry)

            if trace is not None:
                eff_w = (ONTOLOGY_WEIGHT * ontology_confidence) if (ontology_score is not None and ontology_confidence > 0) else 0.0
                jt_entry = {
                    "title": job.get("title", "Untitled"),
                    "company": job.get("company", ""),
                    "source": job.get("source", ""),
                    "req_edu_level": req_edu_level,
                    "req_edu_label": req_edu_label,
                    "use_edu": use_edu,
                    "occupation_match": occ_trace,
                    "semantic_score": semantic_score,
                    "semantic_details": sem_trace,
                    "ontology_score": ontology_score,
                    "ontology_confidence": ontology_confidence,
                    "ontology_details": onto_trace,
                    "fusion_details": {
                        "ontology_weight_const": ONTOLOGY_WEIGHT,
                        "confidence": ontology_confidence,
                        "effective_weight": eff_w,
                    },
                    "final_score": score,
                    "relative_score": 50.0,
                    "breakdown": breakdown,
                    "raw_skills_cos": raw_cos["skills"],
                    "raw_exp_cos": raw_cos["exp"],
                    "raw_title_cos": raw_cos["title"],
                    "comment": comment,
                }
                trace.job_traces.append(jt_entry)
                job_trace_By_idx[id(res_entry)] = jt_entry

        raw_scores = [r["match_score"] for r in results]
        if raw_scores:
            min_score, max_score = min(raw_scores), max(raw_scores)
            score_range = round(max_score - min_score, 2)

            for r in results:
                if score_range > 0.01:
                    r["relative_score"] = round((r["match_score"] - min_score) / score_range * 100, 1)
                else:
                    r["relative_score"] = 50.0  # all jobs scored about the same
                if id(r) in job_trace_By_idx:
                    job_trace_By_idx[id(r)]["relative_score"] = r["relative_score"]

            if trace is not None:
                trace.ranking_summary = {
                    "min_score": min_score,
                    "max_score": max_score,
                    "score_range": score_range,
                }
        else:
            return []

        return sorted(results, key=lambda r: r["match_score"], reverse=True)
    finally:
        if standalone and trace is not None:
            trace.mark_stage("job_matching_and_ranking")
            export_session_log(trace, get_engine_info())
            end_session()



def debug_match(parsed_resume, job_description, job_title=None):
    """Diagnostic tool to inspect fine-grained matches and scores."""
    expanded_description = expand_job_description_with_occupation(job_title, job_description)
    resume_segments = build_resume_segments(parsed_resume)

    res = compute_semantic_match_score(
        resume_segments, expanded_description, return_details=True
    )
    score, details = res if isinstance(res, tuple) else (res, [])

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
