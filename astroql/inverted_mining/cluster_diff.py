"""Diff v2 clusters vs v1 clusters per tradition.

Reports NEW clusters in v2 that were absent (or much rarer) in v1, and
clusters whose frequency moved significantly between runs. The new
clusters are the candidates for v2 rule extraction.

Usage:
  python -m astroql.inverted_mining.cluster_diff \
      --v1 astroql/inverted_mining/data/clusters_v1 \
      --v2 astroql/inverted_mining/data/clusters \
      --out astroql/inverted_mining/data/audit/v2_diff.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load(p: Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for f in p.glob("*.json"):
        with open(f, encoding="utf-8") as fp:
            out[f.stem] = json.load(fp)
    return out


def diff(v1_dir: str, v2_dir: str, out_path: str) -> None:
    v1 = _load(Path(v1_dir))
    v2 = _load(Path(v2_dir))

    lines: List[str] = ["# RAG mining v2 vs v1 — cluster diff", ""]

    for tr in sorted(set(v1) | set(v2)):
        v1_rows = {c["cluster_id"]: c for c in v1.get(tr, [])}
        v2_rows = {c["cluster_id"]: c for c in v2.get(tr, [])}
        new_in_v2 = sorted(
            (c for cid, c in v2_rows.items() if cid not in v1_rows),
            key=lambda c: -c["frequency"],
        )
        gone_in_v2 = sorted(
            (c for cid, c in v1_rows.items() if cid not in v2_rows),
            key=lambda c: -c["frequency"],
        )
        moved: List[tuple] = []
        for cid, c2 in v2_rows.items():
            c1 = v1_rows.get(cid)
            if c1 and abs(c2["frequency"] - c1["frequency"]) >= 5:
                moved.append((cid, c1["frequency"], c2["frequency"], c2))
        moved.sort(key=lambda r: -(r[2] - r[1]))

        lines.append(f"## {tr}")
        lines.append(
            f"- v1: {len(v1_rows)} unique passages | "
            f"v2: {len(v2_rows)} unique passages"
        )
        lines.append(f"- NEW in v2: {len(new_in_v2)}")
        lines.append(f"- gone in v2 (no longer retrieved): {len(gone_in_v2)}")
        lines.append(f"- frequency-shifted ≥5: {len(moved)}")
        lines.append("")

        if new_in_v2:
            lines.append("### NEW in v2 (top 15 by freq)")
            for c in new_in_v2[:15]:
                hints = ", ".join(c.get("feature_hints", [])[:6]) or "-"
                lines.append(
                    f"- **freq={c['frequency']}** `{c['source']}` "
                    f"hints=[{hints}]"
                )
                txt = (c["sample_text"]).replace("\n", " ")[:300]
                lines.append(f"  > {txt}...")
            lines.append("")

        if moved[:10]:
            lines.append("### Frequency shifts (top 10 by gain)")
            for cid, f1, f2, c in moved[:10]:
                d = f2 - f1
                lines.append(
                    f"- `{c['source']}`: {f1} -> {f2} (Δ={d:+d})"
                )
            lines.append("")

        lines.append("---")
        lines.append("")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="astroql/inverted_mining/data/clusters_v1")
    ap.add_argument("--v2", default="astroql/inverted_mining/data/clusters")
    ap.add_argument("--out",
                    default="astroql/inverted_mining/data/audit/v2_diff.md")
    args = ap.parse_args()
    diff(args.v1, args.v2, args.out)


if __name__ == "__main__":
    main()
