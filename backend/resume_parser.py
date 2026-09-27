"""
Resume Parser
-------------
Extracts structured data (name, skills, education, experience, etc.)
from PDF and DOCX resumes.

Install dependencies first:
    pip install pdfplumber python-docx pytesseract pdf2image spacy
    python -m spacy download en_core_web_sm

Note: pytesseract also requires the Tesseract OCR engine installed on
your system (not just the Python package). On Windows, download it from
https://github.com/UB-Mannheim/tesseract/wiki and add it to PATH.
On Mac: brew install tesseract. On Linux: sudo apt install tesseract-ocr.

pdf2image also requires poppler installed separately:
Mac: brew install poppler | Linux: sudo apt install poppler-utils
Windows: download poppler binaries and add to PATH.
"""

import re
import json
import unicodedata

import pdfplumber
from docx import Document
import spacy

# OCR imports are optional — only needed if you hit scanned PDFs.
# Wrapped in try/except so the script still runs without them installed.
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# Load the spaCy English model once, at import time (loading it repeatedly is slow)
nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------------------
# STEP 1-2: Extract raw text from PDF or DOCX
# ---------------------------------------------------------------------

def extract_text_from_pdf(filepath):
    """Pull text out of a PDF, page by page."""
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_text_from_docx(filepath):
    """Pull text out of a Word document, including text inside tables."""
    doc = Document(filepath)

    # Regular paragraphs (most of the resume)
    paragraphs = [para.text for para in doc.paragraphs]

    # Tables are stored separately in python-docx and won't show up
    # in doc.paragraphs, so we pull them out manually
    table_text = []
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                table_text.append(cell.text)

    return "\n".join(paragraphs + table_text)


def extract_text(filepath):
    """Route to the right extractor based on file extension."""
    if filepath.lower().endswith(".pdf"):
        return extract_text_from_pdf(filepath)
    elif filepath.lower().endswith(".docx"):
        return extract_text_from_docx(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")


# ---------------------------------------------------------------------
# STEP 6: OCR fallback for scanned (image-based) PDFs
# ---------------------------------------------------------------------

def ocr_pdf(filepath):
    """
    Used when a PDF has little/no extractable text, meaning it's
    probably a scanned image rather than real text. Converts each
    page to an image, then reads text off the image.
    """
    if not OCR_AVAILABLE:
        raise RuntimeError(
            "OCR libraries not installed. Run: "
            "pip install pytesseract pdf2image "
            "(and install the Tesseract + poppler system tools)"
        )
    images = convert_from_path(filepath)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image) + "\n"
    return text


# ---------------------------------------------------------------------
# STEP 3: Clean and normalize text
# ---------------------------------------------------------------------

def clean_text(text):
    """Fix unicode quirks and collapse messy whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'[ \t]+', ' ', text)      # multiple spaces -> one space
    text = re.sub(r'\n{2,}', '\n', text)     # multiple blank lines -> one
    return text.strip()


# ---------------------------------------------------------------------
# STEP 4: Split text into sections — detected dynamically, not from a
# fixed list. This means it adapts to whatever headers the resume
# actually uses ("Work History", "Tech Stack", "Volunteering", etc.)
# instead of only recognizing words we hardcoded in advance.
# ---------------------------------------------------------------------

def is_likely_header(line):
    """
    Decides whether a line LOOKS like a section header, based on
    formatting patterns rather than matching specific words.

    Real resume headers tend to be:
      - short (a handful of words, not a full sentence)
      - written in ALL CAPS or Title Case
      - not ending in a period or comma (headers don't end sentences)
      - not starting with a bullet symbol (that's body content, not a header)

    This is a set of rules of thumb ("heuristics") — it won't be
    perfect on every resume template, but it generalizes far better
    than a fixed list of expected header words.
    """
    stripped = line.strip()

    if not stripped:
        return False

    # Bullet points are content, never headers
    if stripped.startswith(("-", "•", "*", "◦", "‣")):
        return False

    # A comma-separated list ("Python, Java, C++") is content, not a
    # header — real section headers essentially never contain a comma.
    # Found via a real bug: a capitalized skills list like this one
    # passed the Title Case check below and got misdetected as its
    # own header, splitting it away from the actual "Skills" section
    # and leaving that section empty.
    if "," in stripped:
        return False

    # Headers are short. Long lines are sentences/descriptions.
    if len(stripped) > 40:
        return False

    words = stripped.split()
    if len(words) == 0 or len(words) > 5:
        return False

    # Headers don't usually end a sentence with . or , (a trailing colon
    # is fine and common, e.g. "Technical Skills:")
    if stripped.endswith((".", ",")):
        return False

    # Strip a trailing colon before checking the casing pattern
    check_text = stripped.rstrip(":").strip()

    # Accept ALL CAPS headers ("EDUCATION") or Title Case headers
    # ("Work Experience"). str.isupper()/.istitle() are built-in
    # Python string methods that check capitalization patterns.
    if check_text.isupper() and len(check_text) > 1:
        return True
    if check_text.istitle():
        return True

    return False


def split_into_sections(text):
    """
    Walks through the resume line by line. Whenever a line looks like
    a header (per is_likely_header), everything after it gets filed
    under a section named after that line, until the next header.
    No predetermined list of section names is needed — whatever
    headers actually appear in the resume become the section names.
    """
    lines = text.split("\n")
    sections = {}
    current_section = "header"  # anything before the first detected header
    sections[current_section] = []

    for line in lines:
        stripped = line.strip()

        if is_likely_header(stripped):
            # Use the header text itself (lowercased, colon stripped) as the
            # section name, e.g. "Work Experience:" -> "work experience"
            current_section = stripped.rstrip(":").strip().lower()
            sections.setdefault(current_section, [])
        else:
            sections[current_section].append(line)

    # Convert each list of lines back into a single block of text
    return {section: "\n".join(lines_).strip() for section, lines_ in sections.items()}


def find_section(sections, *keywords):
    """
    Since section names are now whatever the resume calls them
    (not fixed keys), this searches the detected section names for
    ANY of the given keywords and returns the first match.
    e.g. find_section(sections, "experience", "work history", "employment")
    will match a section titled "Work Experience" OR "Employment History".
    """
    for name, content in sections.items():
        for keyword in keywords:
            if keyword in name:
                return content
    return ""


# ---------------------------------------------------------------------
# STEP 5: Extract structured fields (name, orgs, dates, skills)
# ---------------------------------------------------------------------

def extract_entities(text):
    """
    Runs spaCy's named entity recognition over the text and sorts
    results into buckets: people's names, organizations, dates.
    """
    doc = nlp(text)
    entities = {"names": [], "orgs": [], "dates": []}

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["names"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["orgs"].append(ent.text)
        elif ent.label_ == "DATE":
            entities["dates"].append(ent.text)

    return entities


def extract_skills(text, sections):
    """
    Pulls individual skills out WITHOUT any predetermined list of
    known skills, and WITHOUT relying only on a dedicated Skills
    section — since a skill mentioned only inside an Experience
    bullet (e.g. "led sprint planning using Agile and Jira") would
    otherwise never get picked up.

    Returns TWO separate lists — kept apart on purpose, not merged:
      1. explicit_skills — items from a detected Skills section, if
         one exists. Clean, high-confidence: this is a deliberate
         list the candidate wrote as a list.
      2. inline_candidates — noun phrases scanned from the ENTIRE
         resume text (catches skills mentioned inline in
         Experience/Projects). Noisier: a bare word like "database"
         pulled out of "updated employee database" loses its context
         and can look deceptively like a strong match for something
         unrelated (e.g. "SQL databases") to a matching engine, since
         it's now indistinguishable from the real thing. Useful for
         display, NOT safe to use as an equal-confidence match
         candidate — matching_engine.py only uses explicit_skills for
         scoring for this reason.
    """
    skills_section = find_section(
        sections, "skills", "technical skills", "tech stack",
        "core competencies", "competencies", "tools"
    )

    explicit_skills = _split_into_items(skills_section) if skills_section else []
    body_text = _build_body_text(sections)
    inline_candidates = _extract_noun_chunk_candidates(body_text)

    # Drop inline candidates that duplicate an explicit skill
    # (case-insensitive) — no need to list "Python" twice
    explicit_lower = {s.lower() for s in explicit_skills}
    inline_candidates = [s for s in inline_candidates if s.lower() not in explicit_lower]

    return explicit_skills, inline_candidates


# Recognized resume section keywords used ONLY to decide where the
# contact block (name, address, phone, email at the top) ends. This is
# NOT the same as the old fixed header list — section names themselves
# are still detected dynamically. This list just marks "once we hit a
# section whose name contains one of these words, we're past the
# contact block and into real content worth scanning for skills."
KNOWN_CONTENT_INDICATORS = [
    "education", "experience", "skill", "project", "certification",
    "summary", "objective", "competenc", "tool", "language", "award",
    "reference", "publication", "volunteer", "leadership", "training",
    "achievement", "activities", "interest", "qualification"
]


def _build_body_text(sections):
    """
    Joins section content into one block, EXCLUDING the contact block
    at the top of the resume (name, address, phone, email). This
    matters because a name like "Maria Santos" or an address line can
    get misdetected by is_likely_header() as its own short "section" —
    so simply excluding the literal "header" bucket isn't enough; we
    also skip any section BEFORE the first one that looks like real
    resume content (education, experience, skills, etc.).
    """
    body_parts = []
    past_contact_block = False

    for section_name, content in sections.items():
        if section_name == "header":
            continue
        if not past_contact_block:
            if any(indicator in section_name for indicator in KNOWN_CONTENT_INDICATORS):
                past_contact_block = True
            else:
                continue  # still in the contact block — skip it
        body_parts.append(content)

    return "\n".join(body_parts)


def _split_into_items(section_text):
    """
    Takes a block of text (the Skills section) and splits it into
    individual items using common list delimiters. Filters out
    anything too long to plausibly be a single skill (that's more
    likely a leftover sentence than a skill name).
    """
    # Normalize bullet characters into commas so everything splits the same way
    normalized = re.sub(r'[•▪◦‣]', ',', section_text)
    # Also treat line breaks and pipes as delimiters, alongside commas/semicolons
    raw_items = re.split(r'[,;|\n]+', normalized)

    items = []
    for raw in raw_items:
        item = raw.strip(" -\t")
        if not item:
            continue
        # Skip anything too long/word-heavy to be a single skill —
        # that's usually a stray sentence, not a listed skill
        if len(item) > 40 or len(item.split()) > 5:
            continue
        items.append(item)

    return items


def _extract_noun_chunk_candidates(text):
    """
    Scans resume body text for short noun phrases (e.g. "machine
    learning", "customer service", "Jira") that might be skills,
    using spaCy's noun-chunk detection. Catches skills mentioned
    inline inside Experience or Project bullets, not just ones listed
    in a dedicated Skills section.

    Processes the text ONE LINE AT A TIME rather than as one big
    block. This matters: resume lines are often fragments (bullet
    points, headers, dates) rather than full sentences, and running
    spaCy over the whole merged block let noun phrases incorrectly
    span across two unrelated lines (e.g. a date ending one line
    fusing with a word starting the next).

    This is inherently noisier than a labeled Skills section — not
    every short noun phrase is actually a skill. Filtering below
    removes the most obvious non-skill noise, but expect some
    irrelevant phrases in the output. That's an acceptable tradeoff
    since this list is for display/explainability, not for the core
    semantic matching (which reads full passages of text, not this
    list).
    """
    # Generic words/phrases that pass the noun-chunk/casing filters but
    # are almost never skills — filtered out explicitly since spaCy
    # alone won't know they're not domain-relevant
    GENERIC_NOISE = {
        "the company", "the team", "the role", "this position",
        "last year", "this project", "the project", "a team",
        "our team", "the client", "the department", "years",
    }

    # Noun phrases starting with these words are almost always
    # generic descriptions ("a 4-person team") rather than named skills
    STOP_STARTERS = {
        "a", "an", "the", "this", "that", "these", "those",
        "our", "my", "your", "his", "her", "its", "their"
    }

    # spaCy entity labels that mean "this is a name/place/org", not a
    # skill — a noun chunk overlapping one of these gets excluded
    NON_SKILL_LABELS = {"PERSON", "GPE", "LOC", "FAC", "ORG", "NORP"}

    candidates = []
    seen = set()

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        doc = nlp(line)
        non_skill_spans = [
            (ent.start_char, ent.end_char)
            for ent in doc.ents if ent.label_ in NON_SKILL_LABELS
        ]

        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()
            phrase_lower = phrase.lower()
            words = phrase.split()

            # Keep it short (1-3 words) and skip duplicates / pronouns / noise
            if len(words) == 0 or len(words) > 3:
                continue
            if phrase_lower in seen or phrase_lower in GENERIC_NOISE:
                continue
            if chunk.root.pos_ == "PRON":
                continue
            if words[0].lower() in STOP_STARTERS:
                continue
            if any(
                chunk.start_char < end and chunk.end_char > start
                for start, end in non_skill_spans
            ):
                continue
            # Skip anything that looks like an address (starts with
            # digits, e.g. "123 Main Street") or an email fragment
            if re.match(r'^\d', phrase) or "@" in phrase:
                continue

            seen.add(phrase_lower)
            candidates.append(phrase)

    return candidates


def extract_contact_info(text):
    """Pulls out email and phone number using regex patterns."""
    email_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
    phone_match = re.search(r'(\+?\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}', text)

    return {
        "email": email_match.group(0) if email_match else None,
        "phone": phone_match.group(0) if phone_match else None,
    }


# ---------------------------------------------------------------------
# STEP 7: Tie everything together into one function
# ---------------------------------------------------------------------

def parse_resume(filepath):
    """
    Main entry point. Give it a path to a PDF or DOCX resume,
    get back a structured dictionary.
    """
    raw_text = extract_text(filepath)

    # If almost no text came out, this is probably a scanned PDF — fall back to OCR
    if len(raw_text.strip()) < 50 and filepath.lower().endswith(".pdf"):
        print("Little/no text found — falling back to OCR...")
        raw_text = ocr_pdf(filepath)

    cleaned = clean_text(raw_text)
    sections = split_into_sections(cleaned)
    entities = extract_entities(cleaned)
    explicit_skills, inline_skills = extract_skills(cleaned, sections)
    contact = extract_contact_info(cleaned)

    return {
        "name": entities["names"][0] if entities["names"] else None,
        "email": contact["email"],
        "phone": contact["phone"],
        # "skills" stays as the combined list for display/backward
        # compatibility (what a UI would show as "detected skills").
        # "explicit_skills" is the high-confidence subset (from an
        # actual Skills section) — matching_engine.py uses ONLY this
        # one for scoring, since "skills" can contain ambiguous
        # context-free fragments (see extract_skills' docstring).
        "skills": explicit_skills + inline_skills,
        "explicit_skills": explicit_skills,
        "inline_skills": inline_skills,
        "organizations_mentioned": entities["orgs"],
        "dates_mentioned": entities["dates"],
        "education": find_section(sections, "education"),
        "experience": find_section(sections, "experience", "employment", "work history"),
        "detected_section_names": list(sections.keys()),
        "raw_sections": sections,
    }


# ---------------------------------------------------------------------
# STEP 8: Quick test runner
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python resume_parser.py <path_to_resume.pdf_or_docx>")
        sys.exit(1)

    filepath = sys.argv[1]
    result = parse_resume(filepath)

    print(json.dumps(result, indent=2, ensure_ascii=False))
