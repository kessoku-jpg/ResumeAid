"""
Standalone Session Logger for ResumeAid.

Generates a clean, defense-ready text report after each matching session
without cluttering matching_engine.py. All inspection and formatting happen
here by reading the pipeline's input/output data and querying the loaded
O*NET skill tree structure.

Output files (written to <repo_root>/logs/):
  - logs/session_YYYYMMDD_HHMMSS_<id>.txt  (timestamped archive per session)
  - logs/latest_session.txt                (always holds the most recent run)
"""

from __future__ import annotations

import datetime
import logging
import os
import uuid
from collections import defaultdict
from typing import Any

import matching_engine as me

logger = logging.getLogger("resumeaid.session_logger")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
LOGS_DIR = os.path.join(_REPO_ROOT, "logs")

DOMAIN_NAMES = {
    "2.A": "Essential Skills",
    "2.B": "Transferable Skills",
    "2.C": "Knowledge",
    "2.E": "Software Skills",
    "4.A": "Work Activities",
}


def _rule(char: str = "=", width: int = 96) -> str:
    return char * width


def _build_resume_skill_tree_lines(parsed_resume: dict[str, Any]) -> tuple[list[str], int]:
    """
    Constructs and renders an ASCII/Unicode tree of the O*NET subtree
    activated by the candidate's resume (both explicit skills and experience).
    Returns (lines, total_matched_concepts).
    """
    nodes = me.SKILL_TREE_NODES
    if not nodes:
        return ["   (O*NET skill_tree.json not loaded)"], 0

    concepts = me.extract_resume_concepts(parsed_resume)
    if not concepts:
        return ["   (No O*NET tree nodes matched from resume)"], 0

    # Collect all ancestors up to the domain root for every matched concept
    active_nodes: set[str] = set()
    children_map: dict[str | None, set[str]] = defaultdict(set)

    for node_id in concepts:
        if node_id not in nodes:
            continue
        cur: str | None = node_id
        while cur and cur in nodes:
            active_nodes.add(cur)
            parent = nodes[cur].get("parent")
            if parent and parent in nodes:
                children_map[parent].add(cur)
            else:
                children_map[None].add(cur)
            cur = parent

    lines: list[str] = ["   O*NET 31.0 Resume Skill Hierarchy"]

    def _sort_key(nid: str) -> tuple[str, str]:
        return (nodes[nid].get("domain", ""), nodes[nid].get("name", ""))

    def _render_node(nid: str, prefix: str, is_last: bool) -> None:
        connector = "└── " if is_last else "├── "
        node = nodes[nid]
        ntype = node.get("type", "node").upper()
        name = node.get("name", nid)
        code_tag = nid if ":" not in nid else nid.split(":", 1)[0]

        hit = concepts.get(nid)
        if hit:
            ev = hit["evidence"]
            if len(ev) > 42:
                ev = ev[:39] + "..."
            badge = f"  <== Resume: \"{ev}\" [{hit['how'].upper()}, sim={hit['weight']:.2f}]"
        else:
            badge = ""

        if node.get("type") == "domain":
            label = f"[Domain {code_tag}] {name}"
        elif node.get("type") == "example":
            label = f"[Tool/Example] {name}"
        else:
            label = f"[{ntype.title()} {code_tag}] {name}"

        lines.append(f"   {prefix}{connector}{label}{badge}")

        child_prefix = prefix + ("    " if is_last else "│   ")
        kids = sorted(children_map.get(nid, set()), key=_sort_key)
        for idx, child_id in enumerate(kids):
            _render_node(child_id, child_prefix, idx == len(kids) - 1)

    roots = sorted(children_map.get(None, set()), key=_sort_key)
    for idx, root_id in enumerate(roots):
        _render_node(root_id, "", idx == len(roots) - 1)

    return lines, len(concepts)


def format_session_report(
    parsed_resume: dict[str, Any],
    health_check: dict[str, Any],
    search_info: dict[str, Any],
    ranked_results: list[dict[str, Any]],
    mode: str = "pipeline",
    elapsed_ms: float = 0.0,
    top_detailed_jobs: int = 5,
) -> tuple[str, str]:
    """
    Formats a concise, defense-ready computational report for a matching session.
    Returns (session_id, report_text).
    """
    session_id = uuid.uuid4().hex[:8]
    now = datetime.datetime.now().astimezone()
    lines: list[str] = []
    w = lines.append

    # ------------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------------
    w(_rule("="))
    w("RESUMEAID — BACKEND COMPUTATIONAL & ONTOLOGY SESSION LOG")
    w(_rule("="))
    w(f"Session ID      : {session_id}")
    w(f"Timestamp       : {now.strftime('%Y-%m-%d %H:%M:%S %Z (UTC%z)')}")
    w(f"Session Mode    : {mode}")
    w(f"Execution Time  : {elapsed_ms:.1f} ms")
    w(f"Bi-Encoder Model: {me.MODEL_NAME} (384-dimensional dense vector embeddings)")
    w(f"O*NET Ontology  : O*NET 31.0 Skill Tree ({len(me.SKILL_TREE_NODES):,} nodes, {len(me.SKILL_TREE_INDEX):,} lookup aliases)")
    w("")

    # ------------------------------------------------------------------
    # 1. PARSED RESUME SUMMARY
    # ------------------------------------------------------------------
    w(_rule("-"))
    w("1. PARSED RESUME & CANDIDATE PROFILE")
    w(_rule("-"))
    edu_level, edu_label = me.detect_resume_education_level(parsed_resume)
    explicit_skills = parsed_resume.get("explicit_skills") or parsed_resume.get("skills") or []
    inline_skills = parsed_resume.get("inline_skills") or []

    w(f"Candidate Name  : {parsed_resume.get('name') or 'Not detected'}")
    w(f"Education Level : {edu_label or 'Not specified'} (Gate Level {edu_level}/4)")
    w(f"Sections Found  : {', '.join(parsed_resume.get('detected_section_names') or []) or 'None'}")
    w(f"Explicit Skills : ({len(explicit_skills)}) {', '.join(explicit_skills) if explicit_skills else 'None'}")
    if inline_skills:
        w(f"Inline Mentions : ({len(inline_skills)}) {', '.join(inline_skills[:15])}")
    if health_check and health_check.get("issues"):
        w(f"Health Issues   : {'; '.join(health_check['issues'])}")
    w("")

    # ------------------------------------------------------------------
    # 2. VISUALIZED RESUME SKILL TREE (O*NET 31.0 SUBTREE)
    # ------------------------------------------------------------------
    w(_rule("-"))
    w("2. RESUME O*NET SKILL TREE (HIERARCHICAL GROUNDING VISUALIZATION)")
    w(_rule("-"))
    w("Each resume skill and experience bullet is mapped to the O*NET 31.0 Content Model")
    w("(via exact alias lookup or SBERT cosine similarity >= 0.62) and expanded up to its root:")
    w("")
    tree_lines, concept_count = _build_resume_skill_tree_lines(parsed_resume)
    lines.extend(tree_lines)
    w("")

    expanded_skills = me.expand_skills_with_categories(explicit_skills) if explicit_skills else []
    implied_only = [s for s in expanded_skills if s not in explicit_skills]
    w(f"Active Ontology Concepts : {concept_count} O*NET nodes")
    w(f"Implied Parent Categories: ({len(implied_only)}) {', '.join(implied_only) if implied_only else 'None'}")
    w("")

    # ------------------------------------------------------------------
    # 3. OCCUPATION INFERENCE & JOB RETRIEVAL
    # ------------------------------------------------------------------
    w(_rule("-"))
    w("3. OCCUPATION INFERENCE & JOB RETRIEVAL")
    w(_rule("-"))
    top_occs = me.rank_occupations_for_resume(parsed_resume, k=3)
    if top_occs:
        w("Closest O*NET Occupations to Resume Vector cos(v_resume, v_occupation):")
        for idx, (soc, title, sc) in enumerate(top_occs, 1):
            w(f"   #{idx} [{soc}] {title:<42} cos_sim = {sc:.4f} ({sc*100:.1f}%)")
    w(f"Search Query Used: {search_info.get('query')!r}")
    w(
        f"Job Sources      : requested={search_info.get('source')} | "
        f"cached={search_info.get('cached_sources', [])} | "
        f"live={search_info.get('live_sources', [])} | "
        f"total_ranked={len(ranked_results)}"
    )
    w("")

    # ------------------------------------------------------------------
    # 4. COMPUTATIONAL SCORING FORMULAS (DEFENSE REFERENCE)
    # ------------------------------------------------------------------
    w(_rule("-"))
    w("4. COMPUTATIONAL SCORING FORMULAS (3-LAYER ENGINE)")
    w(_rule("-"))
    w("  (A) Layer 2a — SBERT Semantic Score (all-MiniLM-L6-v2, 384-d cosine similarity):")
    w("      For each job requirement r_i and resume segment s_j:  s_i = max_j cos(r_i, s_j)")
    w("      S_semantic = ((mean(s_i) + max(s_i)) / 2) * 100")
    w("")
    w("  (B) Layer 2b — O*NET Ontology Score (Wu-Palmer Tree Similarity):")
    w("      WuP(c_job, c_res) = (2 * depth(LCA(c_job, c_res))) / (depth(c_job) + depth(c_res))")
    w("      S_ontology = (sum(w_job * max_res(WuP * w_res)) / sum(w_job)) * 100")
    w("")
    w("  (C) Confidence-Weighted Score Fusion:")
    w(f"      w_eff   = ONTOLOGY_WEIGHT ({me.ONTOLOGY_WEIGHT}) * confidence")
    w("      S_final = (1 - w_eff) * S_semantic + w_eff * S_ontology")
    w("")

    # ------------------------------------------------------------------
    # 5. RANKED JOB LEADERBOARD
    # ------------------------------------------------------------------
    w(_rule("-"))
    w(f"5. FINAL RANKED JOB LEADERBOARD ({len(ranked_results)} JOBS)")
    w(_rule("-"))
    w(
        f"{'Rank':<5} {'Final':<7} {'Rel%':<7} {'SBERT':<7} {'Onto':<7} "
        f"{'Conf':<6} {'Skill%':<7} {'Exp%':<7} {'Title':<32} {'Company'}"
    )
    w(_rule(".", 96))
    for idx, job in enumerate(ranked_results, 1):
        bd = job.get("breakdown") or {}
        onto_str = f"{bd.get('ontology_score')}%" if bd.get("ontology_score") is not None else "N/A"
        w(
            f"#{idx:<4d} {str(job.get('match_score'))+'%':<7} {str(job.get('relative_score'))+'%':<7} "
            f"{str(bd.get('semantic_score'))+'%':<7} {onto_str:<7} "
            f"{str(bd.get('ontology_confidence', 0.0)):<6} "
            f"{str(bd.get('skills_score'))+'%':<7} {str(bd.get('experience_score'))+'%':<7} "
            f"{(job.get('title') or 'Untitled')[:30]:<32} {(job.get('company') or '')[:22]}"
        )
    w("")

    # ------------------------------------------------------------------
    # 6. STEP-BY-STEP BREAKDOWN OF TOP MATCHED JOBS
    # ------------------------------------------------------------------
    top_n = min(top_detailed_jobs, len(ranked_results))
    w(_rule("-"))
    w(f"6. DETAILED SCORING BREAKDOWN (TOP {top_n} MATCHES)")
    w(_rule("-"))
    resume_segments = me.build_resume_segments(parsed_resume, include_education=True)

    for idx, job in enumerate(ranked_results[:top_n], 1):
        bd = job.get("breakdown") or {}
        sem_score = bd.get("semantic_score", 0.0)
        onto_score = bd.get("ontology_score")
        conf = bd.get("ontology_confidence", 0.0)
        w_eff = round(me.ONTOLOGY_WEIGHT * conf, 4) if (onto_score is not None and conf > 0) else 0.0

        w(f"[{idx}] {job.get('title', 'Untitled')} — {job.get('company') or 'N/A'} ({job.get('source') or 'N/A'})")
        if onto_score is not None and w_eff > 0:
            w(
                f"    Fusion Math : (1 - {w_eff:.2f}) * {sem_score}% [SBERT] + "
                f"{w_eff:.2f} * {onto_score}% [Ontology] = {job.get('match_score')}% Final "
                f"({job.get('relative_score')}% Relative)"
            )
        else:
            w(f"    Fusion Math : 1.00 * {sem_score}% [SBERT] = {job.get('match_score')}% Final")

        occ = me.find_best_occupation(job.get("title", ""))
        if occ:
            soc_code, profile, occ_sim = occ
            w(
                f"    O*NET Role  : [{soc_code}] {profile['title']} (title cos_sim={occ_sim:.2f}) "
                f"-> added skills: {', '.join(profile.get('top_elements', [])[:5])}"
            )

        links = bd.get("ontology_links") or []
        if links:
            w("    O*NET Tree Concept Links (Wu-Palmer LCA):")
            for lk in links[:5]:
                w(
                    f"      * Resume '{lk['resume_concept']}' <--> Job '{lk['job_concept']}' "
                    f"| LCA: '{lk['shared_concept']}' ({lk['similarity']}% — {lk['relation']})"
                )

        # Top 3 requirement-to-segment semantic alignments for this job
        expanded_desc = me.expand_job_description_with_occupation(job.get("title"), job.get("description", ""))
        sem_res = me.compute_semantic_match_score(resume_segments, expanded_desc, return_details=True)
        details = sem_res[1] if isinstance(sem_res, tuple) else []
        if details:
            top_reqs = sorted(details, key=lambda d: d["score"], reverse=True)[:3]
            w("    Top SBERT Requirement Alignments:")
            for d in top_reqs:
                req_txt = d["requirement"][:52] + "..." if len(d["requirement"]) > 55 else d["requirement"]
                seg_txt = d["best_segment"][:42] + "..." if len(d["best_segment"]) > 45 else d["best_segment"]
                w(f"      * {d['score']:5.1f}% | Req: \"{req_txt}\"  <--  Resume: \"{seg_txt}\"")
        w("")

    w(_rule("="))
    return session_id, "\n".join(lines) + "\n"


def write_session_log(
    parsed_resume: dict[str, Any],
    health_check: dict[str, Any],
    search_info: dict[str, Any],
    ranked_results: list[dict[str, Any]],
    mode: str = "pipeline",
    elapsed_ms: float = 0.0,
) -> str | None:
    """Generates and writes the session log to logs/latest_session.txt and a timestamped archive."""
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        session_id, report = format_session_report(
            parsed_resume=parsed_resume,
            health_check=health_check,
            search_info=search_info,
            ranked_results=ranked_results,
            mode=mode,
            elapsed_ms=elapsed_ms,
        )
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = os.path.join(LOGS_DIR, f"session_{ts}_{session_id}.txt")
        latest_path = os.path.join(LOGS_DIR, "latest_session.txt")

        with open(archive_path, "w", encoding="utf-8") as f:
            f.write(report)
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info("Exported session log to %s", latest_path)
        return archive_path
    except Exception as e:
        logger.warning("Failed to write session log: %s", e)
        return None
