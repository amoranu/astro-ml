"""Citation integrity test (spec §13.4 — CI MUST fail on hallucinated citation).

Walks every YAML rule across all schools, verifies:
    1. `source` is non-empty.
    2. `source` matches one of the registered classical text patterns
       OR is explicitly tagged `synthesized` (RAG-generated FiredRules
       only — they're not authored YAML).
    3. If `source_uri` is present, it parses with a known scheme
       (`rag://<corpus>/<path>` or `https?://...`).

Run as:
    python -u -m astroql.tests.test_citation_integrity
Exit code 0 if clean, 1 if any rule fails.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]   # astroql/

# Allowlist of classical-text source-string patterns. Any rule whose
# `source` field doesn't match at least one of these is flagged.
ALLOWED_SOURCE_PATTERNS = [
    re.compile(r"^BPHS\b", re.IGNORECASE),               # Brihat Parashara Hora Shastra
    re.compile(r"^Brihat\s+Parashara", re.IGNORECASE),
    re.compile(r"^Phaladeepika", re.IGNORECASE),
    re.compile(r"^Saravali", re.IGNORECASE),
    re.compile(r"^Jaimini\b", re.IGNORECASE),
    re.compile(r"^KP\s+Reader", re.IGNORECASE),
    re.compile(r"^KP\b", re.IGNORECASE),
    re.compile(r"^Krishnamurti", re.IGNORECASE),
    re.compile(r"^Hora\s+Sara", re.IGNORECASE),
    re.compile(r"^Uttara\s+Kalamrita", re.IGNORECASE),
]

URI_SCHEMES = re.compile(r"^(rag://[a-z]+/|https?://)")


def _check_rule(rule: dict, file: str, errors: list[str]) -> None:
    rid = rule.get("rule_id", "<no-id>")
    src = rule.get("source", "")
    if not src or not src.strip():
        errors.append(f"{file}::{rid} — missing/empty source")
        return
    if not any(p.search(src) for p in ALLOWED_SOURCE_PATTERNS):
        errors.append(
            f"{file}::{rid} — source {src!r} doesn't match any classical "
            f"text pattern. Add pattern to ALLOWED_SOURCE_PATTERNS or fix "
            f"the citation."
        )
    uri = rule.get("source_uri")
    if uri and not URI_SCHEMES.match(uri):
        errors.append(
            f"{file}::{rid} — source_uri {uri!r} doesn't match a known "
            f"scheme (rag://<corpus>/ or https?://)"
        )


def main() -> int:
    rules_root = REPO / "rules"
    yaml_files = sorted(rules_root.glob("**/*.yaml"))
    yaml_files = [f for f in yaml_files
                  if f.name not in ("features_schema.yaml",
                                    "calibrated_strengths.yaml")]
    errors: list[str] = []
    total = 0
    for yf in yaml_files:
        with open(yf, encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
        if not isinstance(data, list):
            errors.append(f"{yf} — not a list of rules")
            continue
        for rule in data:
            total += 1
            _check_rule(rule, str(yf.relative_to(REPO)), errors)

    print(f"Checked {total} rules across {len(yaml_files)} files.")
    if errors:
        print(f"\n[FAIL] {len(errors)} citation issues:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("[OK] all citations match allowed patterns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
