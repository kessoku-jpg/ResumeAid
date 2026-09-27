"""
Resume Improver
----------------
Diagnoses structural problems in a resume BEFORE it reaches the
matching engine, and — where possible — fixes the biggest one
automatically: a missing or thin Skills section.

WHY THIS MATTERS: every real bug we've found in this project so far
(the false "database" match, the "Telemarketers" misclassification)
traced back to the same root cause — a resume with no dedicated
Skills section forces the parser to guess at skills from body text,
which is inherently noisier than a real listed skill. This tool
attacks that problem directly: it scans the ENTIRE resume against
O*NET's real, government-maintained vocabulary of ~8,700 known
skills/tools (skill_taxonomy.json) and surfaces genuine matches the
person can add as an explicit Skills section — turning noisy inline
mentions into clean, trustworthy ones.

Requires skill_taxonomy.json (see build_taxonomy_from_onet.py).
Without it, this tool still runs the structural diagnostics, just
skips the skill-suggestion step.

Usage:
    python resume_improver.py my_resume.pdf
"""

import json
import os
import re
import sys

from resume_parser import parse_resume, clean_text, extract_text, split_into_sections

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SKILL_TAXONOMY_PATH = os.path.join(_THIS_DIR, "skill_taxonomy.json")

# Real O*NET entries are often vendor-prefixed ("Oracle Java",
# "Microsoft Excel", "Adobe Photoshop") because O*NET tracks specific
# commercial products. A resume almost always just says "Java" or
# "Excel" plainly — so an exact-match lookup against raw O*NET keys
# would miss the vast majority of real mentions. Stripping these
# known vendor prefixes at load time creates an ADDITIONAL alias
# ("java" -> same category as "oracle java") without losing the
# original entry.
KNOWN_VENDOR_PREFIXES = [
    "oracle", "microsoft", "ibm", "adobe", "google", "apache", "amazon",
    "sap", "autodesk", "atlassian", "salesforce.com inc", "salesforce.com",
    "red hat", "vmware", "cisco", "intuit", "novell", "sun microsystems",
    "hewlett packard", "apple",
]

# Some vendor products ARE common English words on their own (Word,
# Access, Excel is borderline-fine but Office/Database/Server are
# genuinely ambiguous) — stripping the vendor prefix from these would
# recreate the exact "bare ambiguous word" bug already found and fixed
# once in matching_engine.py (a resume mentioning "employee database"
# in an unrelated sense would falsely "match" Oracle Database). Block
# these specific words from ever becoming a bare alias; the full
# vendor-prefixed form ("oracle database") stays in the vocabulary,
# just not the ambiguous bare word alone.
AMBIGUOUS_BARE_WORDS = {
    "word", "access", "office", "database", "works", "project", "server",
    "exchange", "explorer", "edge", "teams", "forms", "planner", "stream",
    "flow", "power", "designer", "paint", "notes", "one", "view", "reader",
    "writer", "impress", "draw", "base", "calc", "publisher", "visio",
}


def _load_skill_vocabulary():
    """
    Loads skill_taxonomy.json and builds a SET of known skill/tool
    names to scan for, including vendor-prefix-stripped aliases (see
    above). Returns (vocabulary_set, display_names) where
    display_names maps the lowercased term back to a nicely-cased
    version for output (title case, since these read like proper
    nouns/product names).
    """
    try:
        with open(_SKILL_TAXONOMY_PATH, encoding="utf-8") as f:
            taxonomy = json.load(f)
    except FileNotFoundError:
        return set(), {}

    vocabulary = set()
    display_names = {}

    for raw_key in taxonomy.keys():
        vocabulary.add(raw_key)
        display_names[raw_key] = raw_key.title()

        for vendor in KNOWN_VENDOR_PREFIXES:
            if raw_key.startswith(vendor + " "):
                alias = raw_key[len(vendor):].strip()
                if alias and alias not in vocabulary and alias not in AMBIGUOUS_BARE_WORDS:
                    vocabulary.add(alias)
                    display_names[alias] = alias.title()
                break

    return vocabulary, display_names


_SKILL_VOCABULARY, _SKILL_DISPLAY_NAMES = _load_skill_vocabulary()


def scan_for_known_skills(text, max_phrase_length=4):
    """
    Scans resume text for mentions of real O*NET skill/tool names,
    using a greedy longest-match n-gram scan against _SKILL_VOCABULARY
    (word-boundary safe — tokenizes first, so this won't match "java"
    inside an unrelated longer word). Longest-match means "javascript"
    correctly wins over a shorter overlapping match like "java" when
    both could apply to the same text.

    Returns a list of matched skill names (nicely-cased for display),
    deduplicated, in the order first encountered.
    """
    if not _SKILL_VOCABULARY:
        return []

    # Tokenize into words, keeping things like "c++", "c#", "node.js"
    # intact. The dot is only included between two word characters
    # (so "node.js" stays one token) — a dot NOT followed by a word
    # character (e.g. a sentence-ending period after "Access.") is
    # excluded, otherwise "Access." would tokenize as "access." and
    # never match the vocabulary's "access" entry.
    tokens = re.findall(r"[a-zA-Z][\w+#]*(?:\.[a-zA-Z0-9]+)*", text.lower())

    found = []
    seen = set()
    i = 0
    while i < len(tokens):
        matched_length = 0
        # Try longest phrase first (greedy longest-match)
        for length in range(min(max_phrase_length, len(tokens) - i), 0, -1):
            phrase = " ".join(tokens[i:i + length])
            if phrase in _SKILL_VOCABULARY:
                matched_length = length
                if phrase not in seen:
                    seen.add(phrase)
                    found.append(_SKILL_DISPLAY_NAMES.get(phrase, phrase.title()))
                break
        i += matched_length if matched_length else 1

    return found


def analyze_parsed_resume(parsed, filepath="resume"):
    """
    Runs the diagnostic checks against an ALREADY-parsed resume dict
    (avoids re-parsing the file — pipeline.py already has one).
    Returns (issues, suggestions, suggested_skills) without printing
    anything — see print_report() for that, or diagnose_resume() for
    the combined parse+analyze+print flow used by the CLI.
    """
    raw_text = extract_text(filepath) if os.path.exists(filepath) else ""
    cleaned_full_text = clean_text(raw_text) if raw_text else ""

    # Scan only the BODY text — excluding just the "header" bucket
    # (name/contact block before the first detected section). See
    # module docstring reasoning above for why this is more permissive
    # than resume_parser's own _build_body_text.
    body_text = ""
    if cleaned_full_text:
        sections = split_into_sections(cleaned_full_text)
        body_text = "\n".join(content for name, content in sections.items() if name != "header" and content)

    issues = []
    suggestions = []

    # --- Contact info ---
    if not parsed.get("email"):
        issues.append("No email address detected.")
        suggestions.append("Add a clear email address near the top of the resume.")
    if not parsed.get("phone"):
        issues.append("No phone number detected.")

    # --- Skills section ---
    explicit_skills = parsed.get("explicit_skills") or []
    suggested_skills = []
    if not explicit_skills:
        issues.append(
            "No dedicated Skills section detected. Without one, the matching "
            "engine has to guess at skills from body text, which is far less "
            "reliable — this is the single biggest fix available here."
        )
        if body_text:
            suggested_skills = scan_for_known_skills(body_text)
        if suggested_skills:
            suggestions.append(
                "Add a \"Skills\" section listing these terms found elsewhere "
                "in your resume: " + ", ".join(suggested_skills)
            )
        else:
            suggestions.append(
                "Add a dedicated \"Skills\" section listing your key skills "
                "explicitly (no matching real-world skills were found "
                "scanning the rest of the resume automatically)."
            )
    elif len(explicit_skills) < 3:
        issues.append(f"Skills section is thin (only {len(explicit_skills)} listed).")
        suggestions.append("Consider listing more specific skills, tools, or technologies.")

    # --- Experience section ---
    if not parsed.get("experience"):
        issues.append("No Experience/Work History section detected.")
        suggestions.append(
            "Add an \"Experience\" or \"Work History\" section header so "
            "this content is recognized as your work background."
        )

    # --- Overall length sanity check ---
    if cleaned_full_text:
        word_count = len(cleaned_full_text.split())
        if word_count < 80:
            issues.append(f"Resume text is very short ({word_count} words) — may be a parsing issue or a very sparse resume.")

    return issues, suggestions, suggested_skills


def print_report(filepath, issues, suggestions, suggested_skills):
    """Prints the full diagnostic report — split out so pipeline.py can print a shorter version if it wants."""
    print(f"=== Resume Health Check: {filepath} ===\n")
    if not issues:
        print("No structural issues detected — resume is well-formed for matching.\n")
    else:
        print(f"{len(issues)} issue(s) found:\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print()
        print("Suggestions:")
        for s in suggestions:
            print(f"  - {s}")
        print()

    if suggested_skills:
        print("--- Suggested Skills section (copy-paste ready) ---")
        print("Skills: " + ", ".join(suggested_skills))
        print()


def diagnose_resume(filepath):
    """
    CLI entry point: parses the resume file, runs diagnostics, prints
    the full report. Returns the parsed resume dict plus the list of
    suggested skills, in case a caller wants to use them
    programmatically.
    """
    parsed = parse_resume(filepath)
    issues, suggestions, suggested_skills = analyze_parsed_resume(parsed, filepath)
    print_report(filepath, issues, suggestions, suggested_skills)
    return parsed, suggested_skills


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python resume_improver.py <path_to_resume.pdf_or_docx>")
        sys.exit(1)

    diagnose_resume(sys.argv[1])
