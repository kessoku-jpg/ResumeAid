"""
Build Occupation Profiles from O*NET
--------------------------------------
One-time converter: builds occupation_profiles.json, which maps each
O*NET occupation to its title, known alternate titles, and its most
important generic skills/knowledge/abilities (per O*NET's own
importance ratings).

WHY THIS MATTERS: right now, matching only works with whatever text is
literally IN the scraped job description. A short/vague posting (many
real ones are) gives the matching engine very little to work with.
This lets the engine recognize a job's TITLE (e.g. "Software
Developer"), match it to the closest O*NET occupation, and pull in
that occupation's well-established important skills (e.g.
"Programming", "Complex Problem Solving", "Critical Thinking") even
if the actual posting text never spells those out. matching_engine.py
uses this to enrich thin job descriptions before scoring.

Requires FIVE O*NET files (all from the same "download all" package):
    "Occupation Data.csv"   — occupation codes + canonical titles
    "Alternate Titles.csv"  — job title synonyms per occupation
    "Skills.csv"            — importance-rated generic skills
    "Knowledge.csv"         — importance-rated knowledge areas
    "Abilities.csv"         — importance-rated abilities

SETUP (one-time, on your own machine):
    1. Unzip your O*NET database download
    2. Put all five files above in the same folder as this script
       (don't rename them)
    3. Run: python build_occupation_profiles.py
       This creates occupation_profiles.json — matching_engine.py
       picks it up automatically. Run again if you download a newer
       O*NET release later.

If occupation_profiles.json is missing, matching_engine.py still
works fine — it just skips this enrichment step entirely.
"""

import csv
import json
import os
from collections import defaultdict

# How many top-importance elements to keep per occupation, per source
# file. Skills/Knowledge/Abilities each contribute up to this many,
# so an occupation profile can have up to 3x this many total elements
# (fewer after de-duplication, since some names overlap across files).
TOP_N_PER_SOURCE = 6

# Underscore versions checked first (current O*NET CSV package naming),
# space versions kept as fallback for older/differently-packaged downloads.
OCCUPATION_DATA_CANDIDATES = ["Occupation_Data.csv", "Occupation Data.csv", "Occupation_Data.txt", "Occupation Data.txt"]
ALTERNATE_TITLES_CANDIDATES = ["Alternate_Titles.csv", "Alternate Titles.csv", "Alternate_Titles.txt", "Alternate Titles.txt"]
RATING_FILE_GROUPS = {
    "skills": ["Skills.csv", "Skills.txt"],
    "knowledge": ["Knowledge.csv", "Knowledge.txt"],
    "abilities": ["Abilities.csv", "Abilities.txt"],
}

OUTPUT_FILE = "occupation_profiles.json"


def find_file(candidates):
    for filename in candidates:
        if os.path.exists(filename):
            return filename
    return None


def open_reader(filepath):
    """Returns a csv.reader configured for either .csv (comma) or .txt (tab)."""
    delimiter = "," if filepath.lower().endswith(".csv") else "\t"
    f = open(filepath, encoding="utf-8-sig", newline="")
    return f, csv.reader(f, delimiter=delimiter)


def load_occupation_titles():
    """
    Reads Occupation Data.csv -> { soc_code: canonical_title }
    """
    filepath = find_file(OCCUPATION_DATA_CANDIDATES)
    if filepath is None:
        raise FileNotFoundError(
            f"Couldn't find any of: {OCCUPATION_DATA_CANDIDATES}. "
            "This file is required (it provides the canonical title "
            "for every occupation)."
        )

    titles = {}
    f, reader = open_reader(filepath)
    with f:
        header = next(reader)
        soc_idx = header.index("O*NET-SOC Code")
        title_idx = header.index("Title")
        for row in reader:
            if len(row) <= max(soc_idx, title_idx):
                continue
            titles[row[soc_idx].strip()] = row[title_idx].strip()

    print(f"Loaded {len(titles)} occupation titles from {filepath}")
    return titles


def load_alternate_titles():
    """
    Reads Alternate Titles.csv -> { soc_code: [alt_title, alt_title, ...] }
    """
    filepath = find_file(ALTERNATE_TITLES_CANDIDATES)
    if filepath is None:
        print("No Alternate Titles file found — proceeding with canonical titles only")
        return {}

    alt_titles = defaultdict(list)
    f, reader = open_reader(filepath)
    with f:
        header = next(reader)
        soc_idx = header.index("O*NET-SOC Code")
        alt_idx = header.index("Alternate Title")
        for row in reader:
            if len(row) <= max(soc_idx, alt_idx):
                continue
            soc_code = row[soc_idx].strip()
            alt_title = row[alt_idx].strip()
            if alt_title:
                alt_titles[soc_code].append(alt_title)

    print(f"Loaded alternate titles for {len(alt_titles)} occupations from {filepath}")
    return dict(alt_titles)


def load_rating_file(filepath):
    """
    Reads a Skills/Knowledge/Abilities file and returns:
        { soc_code: [(element_name, importance_value), ...] }
    sorted by importance descending, keeping only the top N per
    occupation. Filters to Scale ID == "IM" (the 1-5 Importance
    scale) — these files also contain "LV" (Level) rows, which we
    don't need here.
    """
    ratings = defaultdict(list)
    f, reader = open_reader(filepath)
    with f:
        header = next(reader)
        soc_idx = header.index("O*NET-SOC Code")
        element_idx = header.index("Element Name")
        scale_idx = header.index("Scale ID")
        value_idx = header.index("Data Value")

        for row in reader:
            if len(row) <= max(soc_idx, element_idx, scale_idx, value_idx):
                continue
            if row[scale_idx].strip() != "IM":  # only care about Importance ratings
                continue

            soc_code = row[soc_idx].strip()
            element_name = row[element_idx].strip()
            try:
                value = float(row[value_idx].strip())
            except ValueError:
                continue

            ratings[soc_code].append((element_name, value))

    # Keep only the top N per occupation, sorted by importance descending
    trimmed = {}
    for soc_code, elements in ratings.items():
        elements.sort(key=lambda pair: pair[1], reverse=True)
        trimmed[soc_code] = elements[:TOP_N_PER_SOURCE]

    return trimmed


def build_profiles():
    occupation_titles = load_occupation_titles()
    alt_titles = load_alternate_titles()

    # Merge in whichever of Skills/Knowledge/Abilities are present —
    # at least one is required, but you don't strictly need all three
    combined_elements = defaultdict(list)
    sources_used = []

    for source_name, candidates in RATING_FILE_GROUPS.items():
        filepath = find_file(candidates)
        if filepath is None:
            print(f"Skipping '{source_name}': none of {candidates} found")
            continue
        ratings = load_rating_file(filepath)
        for soc_code, elements in ratings.items():
            combined_elements[soc_code].extend(elements)
        sources_used.append(source_name)
        print(f"Loaded '{source_name}' ratings from {filepath}: {len(ratings)} occupations")

    if not sources_used:
        raise FileNotFoundError(
            "None of Skills/Knowledge/Abilities files were found. "
            "At least one is required to build occupation profiles."
        )

    # Build the final profile per occupation
    profiles = {}
    for soc_code, title in occupation_titles.items():
        elements = combined_elements.get(soc_code, [])
        elements.sort(key=lambda pair: pair[1], reverse=True)

        # De-duplicate element names (Skills/Knowledge/Abilities can
        # overlap in naming) while preserving importance-sorted order
        seen = set()
        top_elements = []
        for name, _value in elements:
            name_lower = name.lower()
            if name_lower not in seen:
                seen.add(name_lower)
                top_elements.append(name)

        profiles[soc_code] = {
            "title": title,
            "alt_titles": alt_titles.get(soc_code, []),
            "top_elements": top_elements[:15],  # cap combined total
        }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

    print(f"\nBuilt profiles for {len(profiles)} occupations")
    print(f"Sources used: {', '.join(sources_used)}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    build_profiles()
