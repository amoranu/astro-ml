"""Father longevity prediction — Parashari CF-rules application.

Built on top of astroql/engine (CF inference, MYCIN math, dasha
emitter) and astroql/schemas. The rules encode classical Parashari
techniques: Sun-karaka theory, derived-lagna theory (BPHS Ch. 8),
Vimshottari dasha activation, and W-2/W-4 from Sanjay Rath's death-
window selection methodology.

Public API:
  from astroql.applications.father_longevity import RULES
  from astroql.applications.father_longevity.eval import evaluate
"""
from .rules import RULES, ALL_VERSIONS

__all__ = ["RULES", "ALL_VERSIONS"]
