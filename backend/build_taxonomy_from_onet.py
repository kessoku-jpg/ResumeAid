"""Builds skill_taxonomy.json from O*NET skills database."""

import csv
import json
import os
from collections import defaultdict

# Each entry: (filename candidates, column name holding the specific
# skill, column name holding its broader category). O*NET has used
# different names for these same two concepts across releases.
SKILL_FILE_SOURCES = [
    {"filenames": ["software_skills.csv"],
     "skill_column": "Workplace Example", "category_column": "Element Name"},
    {"filenames": ["Software_Skills.csv", "Software Skills.csv",
                   "Technology_Skills.csv", "Technology Skills.csv", "Technology_Skills.txt", "Technology Skills.txt"],
     "skill_column": "Example", "category_column": "Commodity Title"},
    {"filenames": ["tools_used.csv", "Tools_Used.csv", "Tools Used.csv", "Tools_Used.txt", "Tools Used.txt"],
     "skill_column": "Example", "category_column": "Commodity Title"},
]

OUTPUT_FILE = "skill_taxonomy.json"


def find_file(candidates):
    for filename in candidates:
        if os.path.exists(filename):
            return filename
    return None


def merge_file_into_taxonomy(filepath, skill_col, category_col, taxonomy):
    """Merges an O*NET file's skill categories into the taxonomy."""
    is_csv = filepath.lower().endswith(".csv")
    delimiter = "," if is_csv else "\t"

    with open(filepath, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)

        try:
            skill_idx = header.index(skill_col)
            category_idx = header.index(category_col)
        except ValueError:
            raise RuntimeError(
                f"Couldn't find '{skill_col}' and '{category_col}' columns "
                f"in {filepath}'s header row: {header}"
            )

        row_count = 0
        for row in reader:
            if len(row) <= max(skill_idx, category_idx):
                continue

            skill_key = row[skill_idx].strip().lower()
            category = row[category_idx].strip().lower()

            if skill_key and category:
                taxonomy[skill_key].add(category)
                row_count += 1

    return row_count


def build_taxonomy():
    taxonomy = defaultdict(set)
    total_rows = 0
    files_used = []

    for source in SKILL_FILE_SOURCES:
        filepath = find_file(source["filenames"])
        if filepath is None:
            print(f"Skipping: none of {source['filenames']} found")
            continue

        rows = merge_file_into_taxonomy(
            filepath, source["skill_column"], source["category_column"], taxonomy
        )
        total_rows += rows
        files_used.append(filepath)
        print(f"Merged {filepath}: {rows} rows")

    if not files_used:
        raise FileNotFoundError(
            "No O*NET skill files found. Expected one of: "
            + ", ".join(f for source in SKILL_FILE_SOURCES for f in source["filenames"])
        )

    output = {skill: sorted(categories) for skill, categories in taxonomy.items()}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nTotal: {total_rows} rows from {len(files_used)} file(s)")
    print(f"Built taxonomy covering {len(output)} distinct skills/tools")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    build_taxonomy()
