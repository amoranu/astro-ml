"""RAGPipeline (spec §6.5, §7).

Thin adapter over astro-prod/rag_engine.py. Graceful no-op if the
corpus config isn't available (offline dev / CI).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schemas.features import FeatureBundle, Passage

log = logging.getLogger(__name__)

_ASTRO_PROD_PATH = Path(
    "C:/Users/ravii/.gemini/antigravity/playground/astro-prod"
)
if str(_ASTRO_PROD_PATH) not in sys.path:
    sys.path.insert(0, str(_ASTRO_PROD_PATH))

try:
    import rag_engine as _rag   # type: ignore
except Exception as e:
    _rag = None
    log.warning("rag_engine unavailable: %s", e)


def _life_area_to_topics(life_area: str) -> List[str]:
    """Map life_area to sub-topic strings for retrieve_for_tradition."""
    m = {
        "longevity": ["longevity maraka death", "ayurdaya lifespan"],
        "marriage":  ["marriage spouse partnership", "kalatra sthana"],
        "career":    ["career profession karma", "dashamsa work"],
        "children":  ["progeny children putra", "santana yoga"],
        "finance":   ["wealth dhana finance", "artha income"],
        "health":    ["disease roga health", "ayurdaya"],
        "nature":    ["nature temperament svabhava"],
    }
    return m.get(life_area, [life_area])


def _natal_context_from_features(features: FeatureBundle) -> Dict[str, Any]:
    """Compact {planets, dasha, lagna} dict for retrieve_for_tradition."""
    ctx: Dict[str, Any] = {"planets": {}}
    for planet, data in features.karaka_data.items():
        ctx["planets"][planet] = {
            "rashi": data.get("sign", ""),
            "house": data.get("house", ""),
        }
    # Take the highest-strength dasha candidate as the "currently relevant"
    # dasha context, if TIMING.
    if features.dasha_candidates:
        c = features.dasha_candidates[0]
        ctx["dasha"] = {
            "md": c.get("md"),
            "ad": c.get("ad"),
        }
    return ctx


class RAGPipeline:
    """Retrieve classical passages relevant to a FeatureBundle."""

    def __init__(self) -> None:
        self._available = _rag is not None and _rag.is_available()

    def is_available(self) -> bool:
        return self._available

    def retrieve(
        self, features: FeatureBundle, top_k: int = 10,
    ) -> List[Passage]:
        if not self._available:
            return []
        school = features.school.value
        focus = features.focus
        life_area = focus.query.life_area.value

        topics = _life_area_to_topics(life_area)
        houses = list(focus.relevant_houses)
        planets = list(set(focus.relation_karakas + focus.domain_karakas))
        # Strip lord-sentinels ("8L") so retrieve gets real planet names.
        planets = [p for p in planets
                   if not (len(p) >= 2 and p[-1] == "L" and p[:-1].isdigit())]
        if "Lagna_Lord" in planets:
            planets.remove("Lagna_Lord")

        query = (
            f"{focus.query.relationship.value} {life_area} "
            f"{focus.query.effect.value} {' '.join(topics[:1])}"
        )
        try:
            raw = _rag.retrieve_for_tradition(
                query=query,
                tradition=school,
                houses=houses,
                planets=planets,
                category=life_area,
                sub_topics=topics,
                natal_context=_natal_context_from_features(features),
            )
        except Exception as e:
            log.warning("RAG retrieval raised: %s", e)
            return []

        out: List[Passage] = []
        for i, row in enumerate(raw[:top_k]):
            out.append(Passage(
                passage_id=f"{school}:{i}",
                text=row.get("text", ""),
                source=row.get("source", ""),
                score=float(row.get("score", 0.0)),
                rule_type=None,
                metadata={"raw_score": row.get("score")},
            ))
        return out
