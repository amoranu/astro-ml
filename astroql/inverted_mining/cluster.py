"""Passage clustering + audit-markdown writer (Phase D).

After RAG mining (Phase C) writes per-(subject, tradition) passages,
this module aggregates them across all subjects per tradition and
produces:

  1. clusters/{tradition}.json — per-passage frequency + subject list
  2. audit/{tradition}.md — human-readable audit doc with suggested
     rule features extracted from each cluster, ordered by frequency

The audit markdown is the BLOCKING REVIEW STEP before drafting YAML
rules in Phase D.2.

Usage:
  python -m astroql.inverted_mining.cluster
  # or:
  python -m astroql.inverted_mining.cluster --tradition parashari
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Coarse keyword extraction — used to hint at what features a passage
# is talking about, so a human reviewer can quickly judge relevance.
_FEATURE_HINTS = [
    # transit factors
    ("ashtama_shani", r"\b(ashtama|8th from moon|eighth from moon)\b"),
    ("sade_sati",      r"\b(sade ?sati|7\.5 year|saturn over moon)\b"),
    ("kantaka_shani",  r"\b(kantaka|4th from moon)\b"),
    ("saturn_transit", r"\b(saturn (transit|gochar|gochara))\b"),
    ("jupiter_transit",r"\b(jupiter (transit|gochar|gochara))\b"),
    ("mars_transit",   r"\b(mars (transit|gochar|gochara))\b"),
    ("mrityu_bhaga",   r"\b(mrityu ?bhaga|critical degree)\b"),
    # houses
    ("h_2",  r"\b(2nd|second) house\b|\bdhana bhava\b"),
    ("h_7",  r"\b(7th|seventh) house\b|\bjaya bhava\b"),
    ("h_8",  r"\b(8th|eighth) house\b|\brandhra\b"),
    ("h_9",  r"\b(9th|ninth) house\b|\bpitru bhava\b|\bpitri bhava\b"),
    ("h_12", r"\b(12th|twelfth) house\b|\bvyaya bhava\b"),
    # roles
    ("maraka",   r"\bmarakas?\b|\bmaraka(sthan|sthana|esha)\b"),
    ("badhaka",  r"\bbadhakas?\b"),
    ("eighth_lord", r"\b(8th|eighth) lord\b|\b8L\b|\brandhra(esh|adhipathi)\b"),
    ("ninth_lord",  r"\b(9th|ninth) lord\b|\b9L\b|\bpitri lord\b"),
    ("dusthana", r"\bdusthan?a\b|\b6/8/12\b|\b6, 8, (and )?12\b"),
    # karakas
    ("sun_pitri", r"\bsun (is|as) (the )?(karaka|significator) (of |for )?(father|pitri|pitru)\b|\bsurya .*pitri\b"),
    ("saturn_ayur",r"\b(saturn|shani) (is|as) (the )?(karaka|significator) (of |for )?(longevity|ayur|ayurdaya|ayushkaraka)\b|\bsaturn .*ayur\b"),
    # dasha
    ("dasha_chain", r"\b(major period|maha ?dasha|antar ?dasha|pratyantar)\b"),
    ("md_8l_yoga",  r"\bmajor period of (the )?8th lord\b"),
    # exceptions
    ("vipreet",     r"\bvipreet|viparita\b"),
    ("neecha_bhanga", r"\b(neecha ?bhanga|nicha bhanga|debility cancell)\b"),
    ("yogakaraka",  r"\byoga ?karaka\b"),
    # benefic protection
    ("jupiter_benefic", r"\bjupiter\b.{0,40}\b(protect|cancel|relief|benefic)\b"),
    ("aspect_lord", r"\b(aspect|drish?ti)\b.{0,30}\b(lord|signifier)\b"),
    # gender / relationship
    ("father_specific", r"\b(father|pitri|pitru|paternal)\b"),
    ("mother_specific", r"\b(mother|matri|matru|maternal)\b"),
]


def _hash_text(text: str, prefix_len: int = 200) -> str:
    """Stable id for a passage based on first N chars (not full text —
    classical chunks often have small whitespace variants)."""
    norm = re.sub(r"\s+", " ", text.strip().lower())[:prefix_len]
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def _extract_feature_hints(text: str) -> List[str]:
    """Lowercased keyword scan."""
    t = text.lower()
    hits = []
    for label, pat in _FEATURE_HINTS:
        if re.search(pat, t, flags=re.IGNORECASE):
            hits.append(label)
    return hits


def cluster_passages(
    rag_dir: str, snap_dir: str,
    only_tradition: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns: {tradition: [cluster_dict...]} sorted by frequency desc.
    Each cluster_dict:
      - cluster_id (hash)
      - source
      - sample_text (first occurrence)
      - frequency (# of distinct subjects)
      - subjects: [{subject_id, name, gender, age_at_loss, age_bucket, score}]
      - feature_hints: [labels]
    """
    rag_path = Path(rag_dir)
    snap_path = Path(snap_dir)

    # subject metadata index
    subj_meta: Dict[int, Dict[str, Any]] = {}
    for sf in snap_path.glob("*.json"):
        try:
            with open(sf, encoding="utf-8") as f:
                s = json.load(f)
            subj_meta[s["subject_id"]] = {
                "name": s["name"],
                "gender": s["gender"],
                "age_at_loss": s["age_at_loss"],
                "age_bucket": s["age_bucket"],
            }
        except Exception:
            pass

    by_tradition: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for rf in rag_path.glob("*.json"):
        try:
            with open(rf, encoding="utf-8") as f:
                r = json.load(f)
        except Exception:
            continue
        tr = r.get("tradition")
        if not tr or (only_tradition and tr != only_tradition):
            continue
        sid = r.get("subject_id")
        for psg in r.get("passages", []):
            text = psg.get("text", "")
            if not text:
                continue
            cid = _hash_text(text)
            cluster = by_tradition[tr].setdefault(cid, {
                "cluster_id": cid,
                "source": psg.get("source", ""),
                "sample_text": text,
                "subjects": [],
                "subject_ids_seen": set(),
            })
            if sid in cluster["subject_ids_seen"]:
                continue
            cluster["subject_ids_seen"].add(sid)
            meta = subj_meta.get(sid, {})
            cluster["subjects"].append({
                "subject_id": sid,
                "name": meta.get("name", ""),
                "gender": meta.get("gender", ""),
                "age_at_loss": meta.get("age_at_loss"),
                "age_bucket": meta.get("age_bucket", ""),
                "score": psg.get("score", 0.0),
            })

    # finalize: drop the set, add frequency + hints, sort desc
    out: Dict[str, List[Dict[str, Any]]] = {}
    for tr, clusters in by_tradition.items():
        rows = []
        for cid, c in clusters.items():
            c.pop("subject_ids_seen", None)
            c["frequency"] = len(c["subjects"])
            c["feature_hints"] = _extract_feature_hints(c["sample_text"])
            # gender split
            gens = [s["gender"] for s in c["subjects"]]
            c["gender_split"] = {
                "M": gens.count("M"),
                "F": gens.count("F"),
            }
            # age-bucket split
            buckets = [s["age_bucket"] for s in c["subjects"]]
            c["age_bucket_split"] = {b: buckets.count(b) for b in set(buckets)}
            rows.append(c)
        rows.sort(key=lambda r: -r["frequency"])
        out[tr] = rows
    return out


def write_audit_markdown(
    clusters: Dict[str, List[Dict[str, Any]]],
    out_dir: str,
    min_freq: int = 2,
    max_per_tradition: int = 80,
) -> None:
    """Write one .md per tradition. Lists clusters with freq >= min_freq."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for tr, rows in clusters.items():
        kept = [r for r in rows if r["frequency"] >= min_freq][:max_per_tradition]
        lines: List[str] = []
        lines.append(f"# Inverted-mining audit: **{tr}** longevity (father)")
        lines.append("")
        lines.append(
            f"_{len(rows)} unique passages retrieved across all subjects. "
            f"Showing top {len(kept)} clusters with frequency ≥ {min_freq}._"
        )
        lines.append("")
        lines.append("Each cluster = one classical-text chunk that the 8-pass RAG "
                     "retrieved for one or more death-state snapshots. Higher "
                     "frequency = more snapshots' RAG queries matched this chunk "
                     "= higher chance the chunk encodes a generalizable rule.")
        lines.append("")
        lines.append("**Review prompt for each cluster**: is this passage actually "
                     "describing a *rule* for father-death timing (vs. general "
                     "definition / prose / off-topic)? If yes, what AstroQL "
                     "feature paths would encode it?")
        lines.append("")
        lines.append("---")

        for i, c in enumerate(kept, start=1):
            lines.append("")
            lines.append(f"## C{i}. freq={c['frequency']} — `{c['source']}`")
            hints = ", ".join(c["feature_hints"]) or "—"
            lines.append(f"**Feature hints**: {hints}")
            gs = c["gender_split"]
            lines.append(f"**Gender split**: M={gs['M']} F={gs['F']}")
            ab = c["age_bucket_split"]
            ab_str = ", ".join(f"{k}={v}" for k, v in sorted(ab.items()))
            lines.append(f"**Age-bucket split**: {ab_str}")
            lines.append("")
            lines.append("```")
            # cap text to 1.5KB so the audit doc stays readable
            txt = c["sample_text"]
            if len(txt) > 1500:
                txt = txt[:1500] + " […truncated…]"
            lines.append(txt)
            lines.append("```")
            # sample subjects
            samp = sorted(c["subjects"], key=lambda s: s["score"])[:5]
            samp_str = "; ".join(
                f"{s['name']} ({s['gender']}/{s['age_bucket']}, "
                f"score={s['score']:.3f})"
                for s in samp
            )
            lines.append(f"**Sample subjects** (top 5 by score): {samp_str}")
            lines.append("")
            lines.append("**Reviewer notes**:")
            lines.append("- Relevance: ☐ rule  ☐ definition  ☐ prose  ☐ off-topic")
            lines.append("- Feature path(s):")
            lines.append("- Yoga draft:")
            lines.append("")
            lines.append("---")

        md_path = out_path / f"{tr}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"wrote {md_path} ({len(kept)} clusters)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag-results",
                    default="astroql/inverted_mining/data/rag_results")
    ap.add_argument("--snapshots",
                    default="astroql/inverted_mining/data/snapshots")
    ap.add_argument("--out-clusters",
                    default="astroql/inverted_mining/data/clusters")
    ap.add_argument("--out-audit",
                    default="astroql/inverted_mining/data/audit")
    ap.add_argument("--tradition",
                    choices=["parashari", "jaimini", "kp"], default=None)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--max-per-tradition", type=int, default=80)
    args = ap.parse_args()

    clusters = cluster_passages(
        args.rag_results, args.snapshots, only_tradition=args.tradition,
    )

    out_clust = Path(args.out_clusters)
    out_clust.mkdir(parents=True, exist_ok=True)
    for tr, rows in clusters.items():
        with open(out_clust / f"{tr}.json", "w", encoding="utf-8") as f:
            # drop sets / heavy stuff for storage
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"wrote {out_clust / f'{tr}.json'} ({len(rows)} unique)")

    write_audit_markdown(
        clusters, args.out_audit,
        min_freq=args.min_freq,
        max_per_tradition=args.max_per_tradition,
    )


if __name__ == "__main__":
    main()
