"""
Session Process & Computational Logger for ResumeAid.

Exports a comprehensive, human-readable text report after every matching
session (both file upload via run_pipeline and profile rematch via
run_pipeline_from_profile). The report documents every backend stage and
all underlying computational computer science details:

  - Stage 0: Session, Environment & Model Architecture Metadata
  - Stage 1: Document Ingestion, Unicode Normalization & NLP Parsing (spaCy)
  - Stage 2: Resume Diagnostics & Greedy Vocabulary Token Scanning
  - Stage 3: Layer 1a — O*NET Skill Tree Grounding & Ancestor Expansion
  - Stage 4: Dense Vector Encoding (SBERT 384-d) & Ontology Concept Extraction
  - Stage 5: Semantic Occupation Inference & Multi-Source Job Retrieval
  - Stage 6: Per-Job Deep Computational Breakdown (Layer 1b, 2a, 2b, Fusion)
  - Stage 7: Min-Max Relative Normalization & Final Ranking Leaderboard

Output files (written to <repo_root>/logs/):
  - logs/session_YYYYMMDD_HHMMSS_<id>.txt  (timestamped archive per session)
  - logs/latest_session.txt                (always holds the most recent run)
"""

from __future__ import annotations

import datetime
import logging
import os
import threading
import time
import uuid
from typing import Any

logger = logging.getLogger("resumeaid.session_logger")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
LOGS_DIR = os.path.join(_REPO_ROOT, "logs")

_tls = threading.local()


class SessionTrace:
    """Collects step-by-step computational telemetry during a single pipeline run."""

    def __init__(self, mode: str, metadata: dict[str, Any] | None = None) -> None:
        self.session_id: str = uuid.uuid4().hex[:8]
        self.mode: str = mode
        self.started_at: datetime.datetime = datetime.datetime.now().astimezone()
        self.start_perf: float = time.perf_counter()
        self.metadata: dict[str, Any] = metadata or {}
        self.timings_ms: dict[str, float] = {}
        self._stage_start: float = self.start_perf

        # Stage 1: Parsing
        self.parsing: dict[str, Any] = {}
        # Stage 2: Health check / improver
        self.diagnostics: dict[str, Any] = {}
        # Stage 3: Skill tree expansion
        self.skill_expansion: dict[str, Any] = {}
        # Stage 4: Precomputed resume embeddings & ontology concepts
        self.resume_encoding: dict[str, Any] = {}
        # Stage 5: Query inference & job fetching
        self.occupation_inference: dict[str, Any] = {}
        self.job_retrieval: dict[str, Any] = {}
        # Stage 6: Per-job detailed traces
        self.job_traces: list[dict[str, Any]] = []
        self.excluded_jobs: list[dict[str, Any]] = []
        # Stage 7: Final ranking summary
        self.ranking_summary: dict[str, Any] = {}

    def mark_stage(self, stage_name: str) -> None:
        now = time.perf_counter()
        self.timings_ms[stage_name] = round((now - self._stage_start) * 1000.0, 2)
        self._stage_start = now

    def total_elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.start_perf) * 1000.0, 2)


def start_session(mode: str, metadata: dict[str, Any] | None = None) -> SessionTrace:
    """Starts and activates a new SessionTrace on the current thread."""
    trace = SessionTrace(mode=mode, metadata=metadata)
    _tls.current_trace = trace
    return trace


def get_active_trace() -> SessionTrace | None:
    """Returns the active SessionTrace for the current thread, if any."""
    return getattr(_tls, "current_trace", None)


def end_session() -> None:
    """Clears the active SessionTrace on the current thread."""
    _tls.current_trace = None


def _vec_preview(tensor: Any, n: int = 6) -> str:
    """Formats the shape, L2 norm, and first n components of a PyTorch tensor."""
    if tensor is None:
        return "None"
    try:
        flat = tensor.detach().cpu().flatten()
        dims = list(tensor.shape)
        norm = float(tensor.detach().cpu().norm().item())
        vals = ", ".join(f"{float(v):+.4f}" for v in flat[:n].tolist())
        return f"shape={dims}, L2_norm={norm:.4f},head=[{vals}, ...]"
    except Exception:
        return str(type(tensor))


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in (text or "").splitlines())


def _rule(char: str = "=", width: int = 100) -> str:
    return char * width


def format_session_report(trace: SessionTrace, engine_info: dict[str, Any]) -> str:
    """Renders the full computational breakdown into a structured text document."""
    lines: list[str] = []
    w = lines.append

    w(_rule("="))
    w("RESUMEAID — BACKEND COMPUTATIONAL & PIPELINE EXECUTION LOG")
    w(_rule("="))
    w(f"Session ID       : {trace.session_id}")
    w(f"Timestamp        : {trace.started_at.strftime('%Y-%m-%d %H:%M:%S %Z (UTC%z)')}")
    w(f"Execution Mode   : {trace.mode}")
    for k, v in trace.metadata.items():
        w(f"{k:<16} : {v}")
    w(f"Total Runtime    : {trace.total_elapsed_ms():.2f} ms")
    w("")

    # ------------------------------------------------------------------
    # STAGE 0: COMPUTATIONAL ARCHITECTURE & HYPERPARAMETERS
    # ------------------------------------------------------------------
    w(_rule("-"))
    w("STAGE 0: COMPUTATIONAL MODELS, KNOWLEDGE BASES & HYPERPARAMETERS")
    w(_rule("-"))
    w("1. Neural Bi-Encoder (Semantic Vector Space):")
    w(f"   - Model Name            : {engine_info.get('model_name', 'all-MiniLM-L6-v2')}")
    w(f"   - Embedding Dimension   : d = {engine_info.get('embedding_dim', 384)} (R^384 dense vector space)")
    w("   - Similarity Metric     : Cosine Similarity  cos(u, v) = (u · v) / (||u||_2 * ||v||_2)")
    w("2. O*NET 31.0 Hierarchical Skill Ontology (skill_tree.json):")
    w(f"   - Total Tree Nodes |V|  : {engine_info.get('tree_node_count', 0):,}")
    w(f"   - Alias Lookup Keys     : {engine_info.get('tree_alias_count', 0):,}")
    w(f"   - Embedded Node Matrix  : {engine_info.get('tree_embedded_count', 0):,} x {engine_info.get('embedding_dim', 384)}")
    w("   - Tree Similarity Metric: Wu-Palmer Similarity")
    w("                             WuP(c_1, c_2) = (2 * depth(LCA(c_1, c_2))) / (depth(c_1) + depth(c_2))")
    w("                             where virtual root depth = 0, domain = 1, category = 2, element/GWA = 3, ...")
    w("3. O*NET Occupation Knowledge Base (occupation_profiles.json):")
    w(f"   - SOC Occupations       : {engine_info.get('occupation_count', 0):,}")
    w(f"   - Title Index Vectors   : {engine_info.get('occupation_title_count', 0):,} titles + alternate titles")
    w("4. Scoring & Threshold Hyperparameters:")
    for hp_key, hp_val in engine_info.get("hyperparameters", {}).items():
        w(f"   - {hp_key:<36} = {hp_val}")
    w("")

    # ------------------------------------------------------------------
    # STAGE 1: RESUME INGESTION & NLP PARSING
    # ------------------------------------------------------------------
    w(_rule("-"))
    w("STAGE 1: RESUME INGESTION, UNICODE NORMALIZATION & NLP PARSING")
    w(_rule("-"))
    p = trace.parsing
    if p:
        w(f"Source Type              : {p.get('source_type', 'unknown')}")
        if p.get("filepath"):
            w(f"File Extension           : {p.get('file_ext', '')}")
            w(f"OCR Fallback Triggered   : {p.get('ocr_used', False)}")
            w(f"Raw Extracted Length     : {p.get('raw_char_count', 0):,} chars ({p.get('raw_line_count', 0)} lines)")
            w(f"Normalized Text Length   : {p.get('clean_char_count', 0):,} chars ({p.get('clean_line_count', 0)} lines) [Unicode NFKD + whitespace collapse]")
        w(f"Detected Sections ({len(p.get('sections', {}))})    : {', '.join(p.get('sections', {}).keys()) or 'None'}")
        for sec_name, sec_info in p.get("sections", {}).items():
            w(f"   * [{sec_name}] ({sec_info['chars']} chars, {sec_info['lines']} lines)")
        w("Extracted Contact & NER Entities (spaCy en_core_web_sm):")
        w(f"   - Candidate Name (PERSON): {p.get('name') or 'Not detected'}")
        w(f"   - Email (Regex)          : {p.get('email') or 'Not detected'}")
        w(f"   - Phone (Regex)          : {p.get('phone') or 'Not detected'}")
        w(f"   - Organizations (ORG)    : {', '.join(p.get('orgs', [])) or 'None'}")
        w(f"   - Dates (DATE)           : {', '.join(p.get('dates', [])) or 'None'}")
        w("Skill Extraction:")
        w(f"   - Explicit Skills ({len(p.get('explicit_skills', []))})   : {p.get('explicit_skills', [])}")
        w(f"   - Inline Noun Chunks ({len(p.get('inline_skills', []))}): {p.get('inline_skills', [])}")
        if p.get("experience"):
            w("Extracted Experience Text:")
            w(_indent(p["experience"], "   | "))
        if p.get("education"):
            w("Extracted Education Text:")
            w(_indent(p["education"], "   | "))
    else:
        w("No parsing telemetry recorded.")
    w("")

    # ------------------------------------------------------------------
    # STAGE 2: RESUME HEALTH CHECK & VOCABULARY SCANNING
    # ------------------------------------------------------------------
    w(_rule("-"))
    w("STAGE 2: RESUME DIAGNOSTICS & O*NET VOCABULARY SCANNING")
    w(_rule("-"))
    d = trace.diagnostics
    if d:
        w(f"Structural Issues ({len(d.get('issues', []))})     : {d.get('issues', [])}")
        w(f"Suggestions ({len(d.get('suggestions', []))})           : {d.get('suggestions', [])}")
        w(f"O*NET Greedy Scan Hits   : {d.get('suggested_skills', [])}")
        w(f"Auto-Applied for Scoring : {d.get('auto_applied', False)}")
    else:
        w("Diagnostics skipped or unavailable.")
    w("")

    # ------------------------------------------------------------------
    # STAGE 3: LAYER 1a — O*NET SKILL TREE GROUNDING & EXPANSION
    # ------------------------------------------------------------------
    w(_rule("-"))
    w("STAGE 3: LAYER 1a — O*NET SKILL TREE GROUNDING & ANCESTOR EXPANSION")
    w(_rule("-"))
    se = trace.skill_expansion
    if se:
        w(f"Input Explicit Skills ({len(se.get('input_skills', []))}): {se.get('input_skills', [])}")
        w("Skill-to-Tree Mapping (Exact Alias Lookup vs. SBERT k-NN over 11,615 Node Embeddings):")
        mappings = se.get("mappings", {})
        if mappings:
            for skill_key, hits in mappings.items():
                if not hits:
                    w(f"   - '{skill_key}' -> [NO TREE MATCH >= threshold]")
                for h in hits:
                    chain_str = " -> ".join(h.get("ancestor_path", []))
                    w(
                        f"   - '{skill_key}' -> [{h['how'].upper()} sim={h['score']:.3f}] "
                        f"Node {h['node_id']} '{h['node_name']}' (type={h['node_type']}, depth={h['depth']})"
                    )
                    w(f"       Ancestor Path: {chain_str}")
        else:
            w("   (No explicit skills to map)")
        if se.get("fallback_hits"):
            w(f"Fallback Taxonomy Hits   : {se.get('fallback_hits')}")
        w(f"Implied Ancestor Terms ({len(se.get('implied_categories', []))}):")
        w(f"   {se.get('implied_categories', [])}")
        w(f"Total Expanded Skills    : {len(se.get('expanded_skills', []))} items")
    else:
        w("No skill expansion recorded.")
    w("")

    # ------------------------------------------------------------------
    # STAGE 4: RESUME VECTOR ENCODING & ONTOLOGY CONCEPT EXTRACTION
    # ------------------------------------------------------------------
    w(_rule("-"))
    w("STAGE 4: RESUME DENSE VECTOR ENCODING (SBERT) & ONTOLOGY CONCEPT EXTRACTION")
    w(_rule("-"))
    re_enc = trace.resume_encoding
    if re_enc:
        w(f"Candidate Education Level: Level {re_enc.get('edu_level')} ({re_enc.get('edu_label') or 'None/Unspecified'})")
        segs = re_enc.get("resume_segments", [])
        w(f"Resume Segments Constructed for Bi-Encoder (M = {len(segs)} segments, + { len(re_enc.get('edu_segments', [])) } education segments):")
        for idx, seg in enumerate(segs):
            w(f"   [{idx:02d}] {seg}")
        for idx, seg in enumerate(re_enc.get("edu_segments", []), start=len(segs)):
            w(f"   [{idx:02d}] (EDU) {seg}")
        w("Pre-Computed Resume Embedding Tensors (R^384):")
        w(f"   - segment_embeddings          : {re_enc.get('segment_embeddings_info')}")
        w(f"   - segment_embeddings_with_edu : {re_enc.get('segment_embeddings_with_edu_info')}")
        w(f"   - skills_emb                  : {re_enc.get('skills_emb_info')}")
        w(f"   - exp_emb                     : {re_enc.get('exp_emb_info')}")
        w(f"   - resume_text_emb             : {re_enc.get('resume_text_emb_info')}")
        concepts = re_enc.get("ontology_concepts", [])
        w(f"Extracted Resume Ontology Concept Set |C_resume| = {len(concepts)} O*NET nodes:")
        for c in concepts:
            w(
                f"   * Node {c['node_id']:<28} | {c['name']:<48} | type={c['type']:<11} "
                f"| depth={c['depth']} | weight={c['weight']:.3f} ({c['how']}) | evidence='{c['evidence']}'"
            )
    w("")

    # ------------------------------------------------------------------
    # STAGE 5: OCCUPATION INFERENCE & JOB RETRIEVAL
    # ------------------------------------------------------------------
    w(_rule("-"))
    w("STAGE 5: SEMANTIC OCCUPATION INFERENCE & MULTI-SOURCE JOB RETRIEVAL")
    w(_rule("-"))
    occ = trace.occupation_inference
    if occ:
        w(f"Query Mode               : {occ.get('mode', 'auto')}")
        if occ.get("top_candidates"):
            w("Top-5 O*NET Occupation Profile Cosine Similarities cos(v_resume, v_occupation):")
            for rank_i, cand in enumerate(occ["top_candidates"], 1):
                w(f"   #{rank_i}: {cand['soc_code']}  {cand['title']:<45}  cos_sim = {cand['score']:.4f} ({cand['score']*100:.2f}%)")
        w(f"Selected Search Query    : {occ.get('selected_query')!r} (confidence={occ.get('confidence')})")
    jr = trace.job_retrieval
    if jr:
        w(f"Requested Job Sources    : {jr.get('source')}")
        w(f"Cached Sources Hit       : {jr.get('cached_sources', [])}")
        w(f"Live Sources Scraped     : {jr.get('live_sources', [])}")
        w(f"Used Placeholder Jobs    : {jr.get('is_placeholder', False)}")
        w(f"Total Unique Listings    : {jr.get('job_count', 0)}")
    w("")

    # ------------------------------------------------------------------
    # STAGE 6: PER-JOB DEEP COMPUTATIONAL BREAKDOWN
    # ------------------------------------------------------------------
    w(_rule("="))
    w(f"STAGE 6: PER-JOB DEEP COMPUTATIONAL BREAKDOWN ({len(trace.job_traces)} QUALIFIED, {len(trace.excluded_jobs)} EXCLUDED)")
    w(_rule("="))

    if trace.excluded_jobs:
        w("\n[EXCLUDED BY EDUCATION QUALIFICATION GATE]")
        for ex in trace.excluded_jobs:
            w(
                f"   - {ex.get('title')} ({ex.get('company')}) — Requires Level {ex.get('req_edu_level')} "
                f"({ex.get('req_edu_label')}), Candidate has Level {ex.get('resume_edu_level')}"
            )

    # Sort job traces by final match_score descending so the log matches the final ranking order
    sorted_jobs = sorted(trace.job_traces, key=lambda j: j.get("final_score", 0.0), reverse=True)
    for rank_idx, jt in enumerate(sorted_jobs, 1):
        w("")
        w(_rule("*"))
        w(
            f"JOB #{rank_idx:02d}: {jt.get('title', 'Untitled')}  |  Company: {jt.get('company') or 'N/A'}  "
            f"|  Source: {jt.get('source') or 'N/A'}"
        )
        w(
            f"FINAL SCORE: {jt.get('final_score')}%  (Relative: {jt.get('relative_score', 'N/A')}%)  "
            f"[SBERT Semantic = {jt.get('semantic_score')}%  |  O*NET Ontology = {jt.get('ontology_score')}% "
            f"(conf={jt.get('ontology_confidence')})]"
        )
        w(_rule("*"))

        # 6.1 Education check & O*NET Occupation Enrichment
        w("6.1 Education Gate & Layer 1b O*NET Occupation Enrichment:")
        w(
            f"   - Education Requirement : {jt.get('req_edu_label') or 'None specified'} "
            f"(Level {jt.get('req_edu_level', 0)}) -> Education segments included in scoring: {jt.get('use_edu', False)}"
        )
        occ_match = jt.get("occupation_match")
        if occ_match and occ_match.get("matched"):
            w(
                f"   - O*NET Title Match     : MATCHED '{occ_match['matched_title']}' -> "
                f"SOC {occ_match['soc_code']} ({occ_match['occupation_title']}) "
                f"with cos_sim = {occ_match['score']:.4f} >= {engine_info['hyperparameters']['OCCUPATION_MATCH_THRESHOLD']}"
            )
            w(f"   - Injected O*NET Skills : {occ_match.get('top_elements', [])}")
        elif occ_match:
            w(
                f"   - O*NET Title Match     : REJECTED (best candidate '{occ_match.get('best_candidate_title')}' "
                f"SOC {occ_match.get('best_soc_code')} had cos_sim = {occ_match.get('best_score', 0.0):.4f} "
                f"< {engine_info['hyperparameters']['OCCUPATION_MATCH_THRESHOLD']})"
            )

        # 6.2 Layer 2a: SBERT Semantic Scoring
        sem = jt.get("semantic_details", {})
        reqs = sem.get("requirements", [])
        w(f"\n6.2 Layer 2a — SBERT Semantic Scoring (Cosine Similarity Grid: {len(reqs)} reqs x {sem.get('num_segments', 0)} resume segments):")
        for req_i, r_detail in enumerate(reqs, 1):
            w(
                f"   Req [{req_i:02d}] cos_sim = {r_detail['raw_cos']:.4f} ({r_detail['score']:5.1f}%) "
                f"| Req: \"{r_detail['requirement']}\""
            )
            w(
                f"            Best Resume Segment [{r_detail['best_segment_idx']:02d}]: \"{r_detail['best_segment']}\""
            )
        w("   Semantic Aggregation Formula:")
        w(f"     mean_cos = (1/{len(reqs) or 1}) * sum(s_i) = {sem.get('mean_cos', 0.0):.4f} ({sem.get('mean_cos', 0.0)*100:.2f}%)")
        w(f"     peak_cos = max(s_i)                = {sem.get('peak_cos', 0.0):.4f} ({sem.get('peak_cos', 0.0)*100:.2f}%)")
        w(
            f"     S_semantic = ((mean_cos + peak_cos) / 2) * 100 "
            f"= (({sem.get('mean_cos', 0.0):.4f} + {sem.get('peak_cos', 0.0):.4f}) / 2) * 100 = {jt.get('semantic_score')}%"
        )

        # 6.3 Layer 2b: Ontology-Guided Scoring (Wu-Palmer)
        onto = jt.get("ontology_details", {})
        j_concepts = onto.get("job_concepts", [])
        w(f"\n6.3 Layer 2b — Ontology-Guided Scoring (Wu-Palmer on O*NET Tree, |C_job| = {len(j_concepts)} concepts):")
        if j_concepts:
            for jc_i, jc in enumerate(j_concepts, 1):
                w(
                    f"   Job Concept [{jc_i:02d}]: Node {jc['job_node_id']} '{jc['job_concept']}' "
                    f"(type={jc['job_type']}, depth={jc['job_depth']}, w_job={jc['job_weight']:.3f}, how={jc['job_how']})"
                )
                w(f"                     Evidence: \"{jc['job_evidence']}\"")
                if jc.get("best_resume_node_id"):
                    w(
                        f"                     Best Resume Match: Node {jc['best_resume_node_id']} "
                        f"'{jc['best_resume_concept']}' (depth={jc['resume_depth']}, w_res={jc['resume_weight']:.3f}, "
                        f"evidence=\"{jc['resume_evidence']}\")"
                    )
                    w(
                        f"                     LCA Node: {jc['lca_node_id']} '{jc['lca_concept']}' (depth={jc['lca_depth']}) "
                        f"| Relation: {jc['relation']}"
                    )
                    w(
                        f"                     WuP = (2 * {jc['lca_depth']}) / ({jc['job_depth']} + {jc['resume_depth']}) "
                        f"= {jc['raw_wup']:.4f}  ->  weighted_sim = {jc['raw_wup']:.4f} * {jc['resume_weight']:.3f} "
                        f"= {jc['weighted_sim']:.4f} ({jc['weighted_sim']*100:.1f}%)  |  "
                        f"w_job * sim = {jc['contribution']:.4f}"
                    )
                else:
                    w("                     Best Resume Match: None (no shared ancestor at depth >= 2) -> sim = 0.0000")
            w("   Ontology Aggregation & Confidence Math:")
            w(f"     Total Job Concept Weight (sum w_job)         = {onto.get('total_weight', 0.0):.4f}")
            w(f"     Weighted Similarity Sum (sum w_job * sim)    = {onto.get('weighted_sum', 0.0):.4f}")
            w(
                f"     S_ontology = ({onto.get('weighted_sum', 0.0):.4f} / {onto.get('total_weight', 1.0):.4f}) * 100 "
                f"= {jt.get('ontology_score')}%"
            )
            w(
                f"     Own-Text Concept Weight W_own = {onto.get('own_text_weight', 0.0):.3f}  ->  "
                f"confidence = min(1.0, {onto.get('own_text_weight', 0.0):.3f} / "
                f"{engine_info['hyperparameters']['ONTOLOGY_FULL_CONFIDENCE_WEIGHT']}) = {jt.get('ontology_confidence')}"
            )
        else:
            w("   (No O*NET concepts extracted from job description)")

        # 6.4 Score Fusion & Sub-Score Breakdown
        fus = jt.get("fusion_details", {})
        bd = jt.get("breakdown", {})
        w("\n6.4 Score Fusion & Dimension Breakdown:")
        w(
            f"   - Effective Ontology Weight: w_eff = ONTOLOGY_WEIGHT * confidence "
            f"= {fus.get('ontology_weight_const', 0.30)} * {fus.get('confidence', 0.0)} = {fus.get('effective_weight', 0.0):.4f}"
        )
        w(
            f"   - Fusion Formula: S_final = (1 - {fus.get('effective_weight', 0.0):.4f}) * {jt.get('semantic_score')} "
            f"+ {fus.get('effective_weight', 0.0):.4f} * {jt.get('ontology_score')} = {jt.get('final_score')}%"
        )
        w(f"   - Skills Vector Cosine Score     : {bd.get('skills_score')}%  (raw cos = {jt.get('raw_skills_cos', 0.0):.4f})")
        w(f"   - Experience Vector Cosine Score : {bd.get('experience_score')}%  (raw cos = {jt.get('raw_exp_cos', 0.0):.4f})")
        w(f"   - Role Title Vector Cosine Score : {bd.get('title_score')}%  (raw cos = {jt.get('raw_title_cos', 0.0):.4f})")
        w(f"   - Exact Keyword Matches          : {bd.get('matched_skills', [])}")
        w(f"   - Explanation Comment            : {jt.get('comment')}")

    # ------------------------------------------------------------------
    # STAGE 7: RELATIVE SCORE NORMALIZATION & LEADERBOARD
    # ------------------------------------------------------------------
    w("")
    w(_rule("="))
    w("STAGE 7: MIN-MAX RELATIVE NORMALIZATION & FINAL RANKING LEADERBOARD")
    w(_rule("="))
    rs = trace.ranking_summary
    if rs:
        w(
            f"Score Distribution: min = {rs.get('min_score')}%  |  max = {rs.get('max_score')}%  "
            f"|  range = {rs.get('score_range')}%  |  Formula: (S_final - min) / (max - min) * 100"
        )
    w(f"{'Rank':<5} {'Final':<8} {'Rel%':<8} {'SBERT':<8} {'Onto':<8} {'Conf':<6} {'Title':<36} {'Company'}")
    w(_rule("-", 100))
    for rank_idx, jt in enumerate(sorted_jobs, 1):
        w(
            f"#{rank_idx:<4d} {str(jt.get('final_score'))+'%':<8} {str(jt.get('relative_score'))+'%':<8} "
            f"{str(jt.get('semantic_score'))+'%':<8} {str(jt.get('ontology_score'))+'%':<8} "
            f"{str(jt.get('ontology_confidence')):<6} {(jt.get('title') or '')[:34]:<36} {jt.get('company') or ''}"
        )

    w("")
    w("Stage Timing Breakdown (ms):")
    for st_name, st_ms in trace.timings_ms.items():
        w(f"   - {st_name:<34}: {st_ms:8.2f} ms")
    w(f"   - {'TOTAL PIPELINE RUNTIME':<34}: {trace.total_elapsed_ms():8.2f} ms")
    w(_rule("="))
    return "\n".join(lines) + "\n"


def export_session_log(trace: SessionTrace, engine_info: dict[str, Any]) -> str | None:
    """Writes the formatted session report to logs/session_<timestamp>.txt and logs/latest_session.txt."""
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        report = format_session_report(trace, engine_info)
        ts = trace.started_at.strftime("%Y%m%d_%H%M%S")
        archive_filename = f"session_{ts}_{trace.session_id}.txt"
        archive_path = os.path.join(LOGS_DIR, archive_filename)
        latest_path = os.path.join(LOGS_DIR, "latest_session.txt")

        with open(archive_path, "w", encoding="utf-8") as f:
            f.write(report)
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info("Exported full backend session log to %s (and %s)", archive_path, latest_path)
        return archive_path
    except Exception as e:
        logger.warning("Failed to export session log: %s", e)
        return None
