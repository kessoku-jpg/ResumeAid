"""
Builds skill_tree.json — a hierarchical skill tree from the O*NET database.

Unlike skill_taxonomy.json (a flat "tool -> category" lookup), this keeps
O*NET's full Content Model hierarchy so the matching engine can expand a
resume skill into its whole ancestor path, e.g.

    "jira"
      -> Atlassian JIRA                      (example / leaf)
      -> Project management software         (element)
      -> Business function specific software (category)
      -> Software Skills                     (domain)

Tree layout (hierarchy comes from O*NET's dotted element_id codes):

    Domain                2.A Essential Skills | 2.B Transferable Skills |
                          2.C Knowledge | 2.E Software Skills | 4.A Work Activities
      Category            2.B.1 Social Skills, 2.E.1 Business function specific software, 4.A.1 Information Input ...
        Element           2.B.1.a ...,  2.E.1.e Project management software
          Example         (2.E only)  Atlassian JIRA, Microsoft Project ...   <- 23_software_skills.sql
        (4.A only)  Subcategory -> GWA -> IWA -> DWA                          <- content model / 32_gwas_to_iwas_to_dwas.sql

Input : the O*NET MySQL dump in ../onet/*.sql (O*NET 31.0)
Output: backend/skill_tree.json

Usage:
    python build_skill_tree_from_onet.py [--onet-dir ../onet]
"""

import argparse
import json
import os
import re
from collections import defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ONET_DIR = os.path.join(_THIS_DIR, "..", "onet")
OUTPUT_FILE = os.path.join(_THIS_DIR, "skill_tree.json")

CONTENT_MODEL_FILE = "01_content_model_reference.sql"
SOFTWARE_SKILLS_FILE = "23_software_skills.sql"
DWA_FILE = "32_gwas_to_iwas_to_dwas.sql"

# Content Model branches that describe skills a resume can demonstrate.
# Abilities (1.A), interests, education (2.D) and work context (4.C) are
# left out on purpose: they're not things people list as skills.
TREE_DOMAINS = ["2.A", "2.B", "2.C", "2.E", "4.A"]

# Node "type" label by depth below the domain, per domain.
TYPE_BY_DEPTH = {
    "4.A": ["domain", "category", "subcategory", "gwa", "iwa", "dwa"],
    "default": ["domain", "category", "element", "subelement"],
}

# Same vendor-prefix aliasing as resume_improver.py: O*NET says
# "Oracle Java" / "Microsoft Excel" but resumes say "Java" / "Excel".
KNOWN_VENDOR_PREFIXES = [
    "oracle", "microsoft", "ibm", "adobe", "google", "apache", "amazon",
    "sap", "autodesk", "atlassian", "salesforce.com inc", "salesforce.com",
    "red hat", "vmware", "cisco", "intuit", "novell", "sun microsystems",
    "hewlett packard", "apple",
]
AMBIGUOUS_BARE_WORDS = {
    "word", "access", "office", "database", "works", "project", "server",
    "exchange", "explorer", "edge", "teams", "forms", "planner", "stream",
    "flow", "power", "designer", "paint", "notes", "one", "view", "reader",
    "writer", "impress", "draw", "base", "calc", "publisher", "visio",
}


# -----------------------------------------------------------------------
# Minimal parser for the O*NET MySQL dump (one INSERT per line).
# -----------------------------------------------------------------------

_INSERT_RE = re.compile(r"^INSERT INTO (\w+) \(([^)]*)\) VALUES \((.*)\);\s*$", re.DOTALL)
_VALUE_RE = re.compile(r"\s*(?:'((?:[^']|'')*)'|(NULL)|([-+]?\d+(?:\.\d+)?))\s*(?:,|$)", re.DOTALL)


def _parse_values(values_sql):
    values, pos = [], 0
    while pos < len(values_sql):
        m = _VALUE_RE.match(values_sql, pos)
        if not m:
            raise ValueError(f"Unparseable VALUES near: {values_sql[pos:pos + 60]!r}")
        if m.group(1) is not None:
            values.append(m.group(1).replace("''", "'"))
        elif m.group(2):
            values.append(None)
        else:
            values.append(m.group(3))
        pos = m.end()
    return values


def iter_rows(path):
    """Yields each INSERT row of an O*NET .sql file as a {column: value} dict."""
    buffer = ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not buffer and not line.startswith("INSERT INTO"):
                continue
            buffer += line
            if not buffer.rstrip().endswith(");"):
                continue  # value spans multiple lines
            m = _INSERT_RE.match(buffer.strip())
            buffer = ""
            if not m:
                continue
            columns = [c.strip() for c in m.group(2).split(",")]
            yield dict(zip(columns, _parse_values(m.group(3))))


# -----------------------------------------------------------------------
# Tree construction
# -----------------------------------------------------------------------

def _domain_of(element_id):
    for domain in TREE_DOMAINS:
        if element_id == domain or element_id.startswith(domain + "."):
            return domain
    return None


def _node_type(element_id, domain):
    depth = element_id.count(".") - domain.count(".")
    labels = TYPE_BY_DEPTH.get(domain, TYPE_BY_DEPTH["default"])
    return labels[min(depth, len(labels) - 1)]


def _clean_name(name):
    return re.sub(r"\s+", " ", name).strip().rstrip(".")


def _aliases_for(name):
    """Lowercase lookup keys for a node name (full name + resume-style short forms)."""
    key = name.lower().strip()
    aliases = {key}

    for vendor in KNOWN_VENDOR_PREFIXES:
        if key.startswith(vendor + " "):
            bare = key[len(vendor):].strip()
            if bare and bare not in AMBIGUOUS_BARE_WORDS:
                aliases.add(bare)
            break

    # O*NET appends acronyms: "Structured query language SQL" -> "sql".
    # Only for generic lowercase phrases — product names like
    # "SoftRisk Technologies SoftRisk SQL" must not claim the bare "sql".
    words = name.split()
    is_generic_phrase = all(w.islower() for w in words[1:-1])
    if (len(words) >= 3 and is_generic_phrase
            and re.fullmatch(r"[A-Z][A-Z0-9+#./-]{1,9}", words[-1])):
        acronym = words[-1].lower()
        if acronym not in AMBIGUOUS_BARE_WORDS:
            aliases.add(acronym)
            aliases.add(" ".join(words[:-1]).lower())

    return aliases


def build_tree(onet_dir):
    nodes = {}

    # 1) Content Model hierarchy (domains, categories, elements, GWA/IWA/DWA)
    for row in iter_rows(os.path.join(onet_dir, CONTENT_MODEL_FILE)):
        element_id = row["element_id"]
        domain = _domain_of(element_id)
        if domain is None:
            continue
        node = {
            "name": _clean_name(row["element_name"]),
            "type": _node_type(element_id, domain),
            "domain": domain,
            "parent": None,
        }
        if row.get("description"):
            node["description"] = row["description"].strip()
        nodes[element_id] = node

    # DWAs are also listed in their own file; make sure none are missing.
    dwa_path = os.path.join(onet_dir, DWA_FILE)
    if os.path.exists(dwa_path):
        for row in iter_rows(dwa_path):
            dwa_id = row["dwa_element_id"]
            if dwa_id not in nodes:
                nodes[dwa_id] = {
                    "name": _clean_name(row["dwa_element_name"]),
                    "type": "dwa", "domain": "4.A", "parent": None,
                }

    # Parent = nearest existing ancestor by dotted code
    for element_id, node in nodes.items():
        parent_id = element_id
        while "." in parent_id:
            parent_id = parent_id.rsplit(".", 1)[0]
            if parent_id in nodes:
                node["parent"] = parent_id
                break

    # 2) Software examples as leaves under their 2.E element
    example_stats = defaultdict(lambda: {"occupations": set(), "hot": False, "in_demand": False})
    for row in iter_rows(os.path.join(onet_dir, SOFTWARE_SKILLS_FILE)):
        parent_id = row["element_id"]
        if parent_id not in nodes:
            continue
        name = _clean_name(row["workplace_example"])
        stats = example_stats[(parent_id, name)]
        stats["occupations"].add(row["onetsoc_code"])
        stats["hot"] |= row.get("hot_technology") == "Y"
        stats["in_demand"] |= row.get("in_demand") == "Y"

    for (parent_id, name), stats in example_stats.items():
        example_id = f"{parent_id}:{name.lower()}"
        nodes[example_id] = {
            "name": name,
            "type": "example",
            "domain": "2.E",
            "parent": parent_id,
            "occupation_count": len(stats["occupations"]),
            "hot_technology": stats["hot"],
            "in_demand": stats["in_demand"],
        }

    # 3) Children lists + depth
    children = defaultdict(list)
    for node_id, node in nodes.items():
        if node["parent"]:
            children[node["parent"]].append(node_id)
    for node_id, node in nodes.items():
        node["children"] = sorted(children.get(node_id, []))
        depth, cur = 0, node
        while cur["parent"]:
            depth += 1
            cur = nodes[cur["parent"]]
        node["depth"] = depth

    # 4) Alias index: lowercase name / short form -> node ids
    index = defaultdict(set)
    for node_id, node in nodes.items():
        if node["type"] == "domain":
            continue  # "knowledge" etc. are too generic to look up directly
        for alias in _aliases_for(node["name"]):
            if len(alias) >= 2:
                index[alias].add(node_id)

    return nodes, {alias: sorted(ids) for alias, ids in sorted(index.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onet-dir", default=DEFAULT_ONET_DIR)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    nodes, index = build_tree(args.onet_dir)

    type_counts = defaultdict(int)
    for node in nodes.values():
        type_counts[node["type"]] += 1

    output = {
        "meta": {
            "source": "O*NET 31.0 Database (CC BY 4.0) - Content Model + Software Skills",
            "domains": {d: nodes[d]["name"] for d in TREE_DOMAINS if d in nodes},
            "node_count": len(nodes),
            "alias_count": len(index),
            "type_counts": dict(sorted(type_counts.items())),
        },
        "roots": [d for d in TREE_DOMAINS if d in nodes],
        "nodes": nodes,
        "index": index,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))

    print(f"Built skill tree: {len(nodes)} nodes, {len(index)} lookup aliases")
    for node_type, count in output["meta"]["type_counts"].items():
        print(f"  {node_type:12s} {count}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
