#!/usr/bin/env python3
"""
filter_logic_safety.py

Filters the GitHub Advisory Database for Python (PyPI) advisories
with CWE identifiers related to logic security, producing a pure
logic-security benchmark dataset.

Usage:
    python filter_logic_safety.py \
        --db-path advisory-database/advisories \
        --output logic_security_benchmark.jsonl \
        --csv-output logic_security_summary.csv
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, OrderedDict
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Logic-security CWE taxonomy
# ──────────────────────────────────────────────────────────────────────

LOGIC_SECURITY_CWES: dict[str, list[str]] = OrderedDict({
    "Race Conditions": [
        "CWE-362",  # Concurrent Execution Using Shared Resource with Improper Synchronization
        "CWE-364",  # Signal Handler Race Condition
        "CWE-366",  # Race Condition within a Thread
        "CWE-367",  # Time-of-check Time-of-use (TOCTOU) Race Condition
        "CWE-421",  # Race Condition During Access to Alternate Channel
    ],
    "Authorization / Access Control": [
        "CWE-862",  # Missing Authorization
        "CWE-863",  # Incorrect Authorization
        "CWE-285",  # Improper Authorization
        "CWE-284",  # Improper Access Control
        "CWE-639",  # Authorization Bypass Through User-Controlled Key
        "CWE-732",  # Incorrect Permission Assignment for Critical Resource
        "CWE-269",  # Improper Privilege Management
        "CWE-250",  # Execution with Unnecessary Privileges
    ],
    "Authentication": [
        "CWE-287",  # Improper Authentication
        "CWE-306",  # Missing Authentication for Critical Function
        "CWE-522",  # Insufficiently Protected Credentials
        "CWE-521",  # Weak Password Requirements
        "CWE-294",  # Authentication Bypass by Capture-replay
        "CWE-304",  # Missing Critical Step in Authentication
        "CWE-307",  # Improper Restriction of Excessive Authentication Attempts
        "CWE-384",  # Session Fixation
        "CWE-613",  # Insufficient Session Expiration
    ],
    "Business Logic": [
        "CWE-840",  # Business Logic Errors
        "CWE-841",  # Improper Enforcement of Behavioral Workflow
    ],
    "State Management": [
        "CWE-372",  # Incomplete Internal State Distinction
        "CWE-374",  # Passing Mutable Objects to an Untrusted Method
        "CWE-375",  # Returning a Mutable Object to an Untrusted Caller
        "CWE-567",  # Unsynchronized Access to Shared Data in a Multithreaded Context
    ],
    "Information Disclosure": [
        "CWE-200",  # Exposure of Sensitive Information to an Unauthorized Actor
        "CWE-201",  # Insertion of Sensitive Information Into Sent Data
        "CWE-203",  # Observable Discrepancy (timing side-channel)
        "CWE-209",  # Generation of Error Message Containing Sensitive Information
        "CWE-532",  # Insertion of Sensitive Information into Log File
    ],
    "Input Validation (Logic)": [
        "CWE-20",   # Improper Input Validation
        "CWE-179",  # Incorrect Behavior Order: Early Validation
    ],
    "Insufficient Verification": [
        "CWE-345",  # Insufficient Verification of Data Authenticity
        "CWE-346",  # Origin Validation Error
        "CWE-347",  # Improper Verification of Cryptographic Signature
        "CWE-352",  # Cross-Site Request Forgery (CSRF)
    ],
    "Error Handling": [
        "CWE-754",  # Improper Check for Unusual or Exceptional Conditions
        "CWE-755",  # Improper Handling of Exceptional Conditions
        "CWE-390",  # Detection of Error Condition Without Action
        "CWE-391",  # Unchecked Error Condition
        "CWE-252",  # Unchecked Return Value
    ],
    "Improper Control Flow": [
        "CWE-670",  # Always-Incorrect Control Flow Implementation
        "CWE-696",  # Incorrect Behavior Order
        "CWE-691",  # Insufficient Control Flow Management
        "CWE-705",  # Incorrect Control Flow Scoping
    ],
    "Resource Management (Logic)": [
        "CWE-770",  # Allocation of Resources Without Limits or Throttling
        "CWE-771",  # Missing Reference to Active Allocated Resource
        "CWE-772",  # Missing Release of Resource after Effective Lifetime
        "CWE-400",  # Uncontrolled Resource Consumption
        "CWE-404",  # Improper Resource Shutdown or Release
    ],
    "Cryptographic Logic": [
        "CWE-327",  # Use of a Broken or Risky Cryptographic Algorithm
        "CWE-328",  # Use of Weak Hash
        "CWE-330",  # Use of Insufficiently Random Values
        "CWE-338",  # Use of Cryptographically Weak Pseudo-Random Number Generator
    ],
})

# Build a flat lookup: CWE-ID -> category name
_CWE_TO_CATEGORY: dict[str, str] = {}
for _cat, _ids in LOGIC_SECURITY_CWES.items():
    for _cwe in _ids:
        _CWE_TO_CATEGORY[_cwe] = _cat

ALL_LOGIC_CWE_IDS: set[str] = set(_CWE_TO_CATEGORY.keys())


# ──────────────────────────────────────────────────────────────────────
# Advisory parsing helpers
# ──────────────────────────────────────────────────────────────────────

def is_python_advisory(advisory: dict) -> bool:
    """Return True if any affected package belongs to the PyPI ecosystem."""
    for affected in advisory.get("affected", []):
        pkg = affected.get("package", {})
        if pkg.get("ecosystem", "").lower() == "pypi":
            return True
    return False


def get_cwe_ids(advisory: dict) -> list[str]:
    """Extract the CWE IDs from database_specific."""
    return advisory.get("database_specific", {}).get("cwe_ids", [])


def get_matched_logic_cwes(cwe_ids: list[str]) -> list[str]:
    """Return the subset of cwe_ids that are in the logic-security set."""
    return [cid for cid in cwe_ids if cid in ALL_LOGIC_CWE_IDS]


def get_cwe_categories(matched: list[str]) -> list[str]:
    """Map matched CWE IDs to their category names (deduplicated, ordered)."""
    seen = set()
    cats = []
    for cid in matched:
        cat = _CWE_TO_CATEGORY.get(cid, "Unknown")
        if cat not in seen:
            seen.add(cat)
            cats.append(cat)
    return cats


def extract_affected_packages(advisory: dict) -> list[dict]:
    """Extract PyPI package name and version range info."""
    pkgs = []
    for affected in advisory.get("affected", []):
        pkg = affected.get("package", {})
        if pkg.get("ecosystem", "").lower() != "pypi":
            continue
        entry = {"name": pkg.get("name", ""), "ranges": []}
        for r in affected.get("ranges", []):
            events = r.get("events", [])
            range_info = {}
            for ev in events:
                if "introduced" in ev:
                    range_info["introduced"] = ev["introduced"]
                if "fixed" in ev:
                    range_info["fixed"] = ev["fixed"]
            if range_info:
                entry["ranges"].append(range_info)
        # Also capture database_specific last_known_affected_version_range
        db_specific = affected.get("database_specific", {})
        lka = db_specific.get("last_known_affected_version_range")
        if lka:
            entry["last_known_affected_version_range"] = lka
        pkgs.append(entry)
    return pkgs


def extract_severity(advisory: dict) -> dict:
    """Extract severity information."""
    result = {}
    # Quantitative CVSS
    severities = advisory.get("severity", [])
    for s in severities:
        result[s.get("type", "UNKNOWN")] = s.get("score", "")
    # Qualitative severity from database_specific
    db_sev = advisory.get("database_specific", {}).get("severity")
    if db_sev:
        result["qualitative"] = db_sev
    return result


def extract_references(advisory: dict) -> list[str]:
    """Extract reference URLs."""
    return [ref.get("url", "") for ref in advisory.get("references", []) if ref.get("url")]


def build_record(advisory: dict, matched_cwes: list[str], filepath: str) -> dict:
    """Build the output record for a matched advisory."""
    all_cwes = get_cwe_ids(advisory)
    return {
        "ghsa_id": advisory.get("id", ""),
        "cve_aliases": advisory.get("aliases", []),
        "summary": advisory.get("summary", ""),
        "details": advisory.get("details", ""),
        "matched_cwes": matched_cwes,
        "all_cwes": all_cwes,
        "cwe_categories": get_cwe_categories(matched_cwes),
        "severity": extract_severity(advisory),
        "affected_packages": extract_affected_packages(advisory),
        "references": extract_references(advisory),
        "published": advisory.get("published", ""),
        "modified": advisory.get("modified", ""),
        "source_file": filepath,
    }


# ──────────────────────────────────────────────────────────────────────
# Main scanning logic
# ──────────────────────────────────────────────────────────────────────

def scan_advisories(db_path: str):
    """
    Walk the advisory database and yield (advisory_dict, filepath)
    for every valid JSON advisory found.
    """
    db = Path(db_path)
    skipped = 0
    for root, _dirs, files in os.walk(db):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(root, fname)
            # Skip broken symlinks or non-regular files (common on Windows
            # with Git repos containing symlinks)
            if not os.path.isfile(fpath):
                skipped += 1
                continue
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                yield data, fpath
            except Exception as e:
                skipped += 1
                # Only print first few warnings to avoid flooding stderr
                if skipped <= 5:
                    print(f"[WARN] Skipping {fpath}: {e}", file=sys.stderr)
    if skipped > 0:
        print(f"[INFO] Skipped {skipped} unreadable files total.",
              file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────
# Output writers
# ──────────────────────────────────────────────────────────────────────

def write_csv(records: list[dict], output_path: str):
    """Write a summary CSV for quick overview."""
    fieldnames = [
        "ghsa_id", "cve_aliases", "summary", "matched_cwes",
        "cwe_categories", "severity", "affected_packages", "published",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {
                "ghsa_id": rec["ghsa_id"],
                "cve_aliases": "; ".join(rec["cve_aliases"]),
                "summary": rec["summary"],
                "matched_cwes": "; ".join(rec["matched_cwes"]),
                "cwe_categories": "; ".join(rec["cwe_categories"]),
                "severity": rec["severity"].get("qualitative", ""),
                "affected_packages": "; ".join(
                    p["name"] for p in rec["affected_packages"]
                ),
                "published": rec["published"],
            }
            writer.writerow(row)
    print(f"[INFO] Wrote CSV summary to {output_path}")


def print_statistics(stats: dict):
    """Pretty-print filtering statistics."""
    print("\n" + "=" * 60)
    print("  Advisory Database — Logic Security Filter Results")
    print("=" * 60)
    print(f"  Total advisories scanned:   {stats['total_scanned']:>6}")
    print(f"  Python (PyPI) advisories:   {stats['total_python']:>6}")
    print(f"  Logic-security matches:     {stats['total_matched']:>6}")
    print("-" * 60)
    print("  Matches by CWE Category:")
    for cat, count in stats["by_category"].items():
        print(f"    {cat:<40} {count:>4}")
    print("-" * 60)
    print("  Top CWE IDs:")
    for cwe, count in list(stats["by_cwe"].items())[:20]:
        cat = _CWE_TO_CATEGORY.get(cwe, "")
        print(f"    {cwe:<12} ({cat:<35}) {count:>4}")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter the GitHub Advisory Database for Python (PyPI) advisories "
            "with logic-security-related CWE identifiers."
        )
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="advisory-database/advisories",
        help="Path to the advisories directory (default: advisory-database/advisories)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logic_security_benchmark.jsonl",
        help="Output JSONL file path (default: logic_security_benchmark.jsonl)",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Optional CSV summary output file path",
    )
    parser.add_argument(
        "--list-cwes",
        action="store_true",
        help="Print the CWE taxonomy used for filtering and exit",
    )
    args = parser.parse_args()

    # --list-cwes: just print the taxonomy and exit
    if args.list_cwes:
        print("Logic-Security CWE Taxonomy used for filtering:\n")
        for cat, ids in LOGIC_SECURITY_CWES.items():
            print(f"  [{cat}]")
            for cwe in ids:
                print(f"    - {cwe}")
            print()
        print(f"Total: {len(ALL_LOGIC_CWE_IDS)} CWE IDs across "
              f"{len(LOGIC_SECURITY_CWES)} categories")
        return

    # Validate input path
    if not os.path.isdir(args.db_path):
        print(f"[ERROR] Advisory database path not found: {args.db_path}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Scanning advisories in: {args.db_path}")
    
    # Stats trackers
    stats = {
        "total_scanned": 0,
        "total_python": 0,
        "total_matched": 0,
        "by_category": Counter(),
        "by_cwe": Counter(),
    }
    
    # We'll collect records for CSV/Stats but also stream to JSONL
    # Since we want to support CSV output at the end, we might keep them in memory
    # OR we could re-read the JSONL. Given 1438 matches, memory is fine.
    matched_records = []

    try:
        with open(args.output, "w", encoding="utf-8") as f_out:
            for advisory, filepath in scan_advisories(args.db_path):
                stats["total_scanned"] += 1
                
                # Progress report
                if stats["total_scanned"] % 2000 == 0:
                    print(f"[INFO] ... scanned {stats['total_scanned']} advisories so far "
                          f"(Python: {stats['total_python']}, matched: {stats['total_matched']})",
                          flush=True)

                if not is_python_advisory(advisory):
                    continue
                stats["total_python"] += 1

                cwe_ids = get_cwe_ids(advisory)
                matched = get_matched_logic_cwes(cwe_ids)
                if not matched:
                    continue

                stats["total_matched"] += 1
                for cid in matched:
                    stats["by_cwe"][cid] += 1
                    stats["by_category"][_CWE_TO_CATEGORY[cid]] += 1

                record = build_record(advisory, matched, filepath)
                matched_records.append(record)
                
                # Stream to JSONL
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush() # Ensure it's written

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user. Stats so far:")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
    finally:
        print(f"[INFO] Wrote {stats['total_matched']} records to {args.output}")
        
        # Write CSV if requested
        if args.csv_output and matched_records:
            write_csv(matched_records, args.csv_output)
        
        # Print statistics
        print_statistics(stats)

if __name__ == "__main__":
    main()
