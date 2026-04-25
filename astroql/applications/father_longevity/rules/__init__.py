"""Father longevity rule sets — versioned for ablation testing.

Current canonical ruleset is v15. Earlier versions retained for
historical comparison and ablation runs.

Version summary:
  v12 — Sun-karaka baseline (BPHS Ch. 41 + classical maraka theory)
  v13 — adds 9th/8th lord transit affliction family
  v14 — adds W-2 derived-lagna helpers + W-4 protective-PAD modifier
  v15 — adds derived-F-lord AD rule + standalone PD = F-loss-lord rule

Use:
  from astroql.applications.father_longevity.rules import RULES
  # RULES is a list of CFRuleSpec — current canonical (= v15)
"""
from .v12 import RULES_V12
from .v13 import RULES_V13
from .v14 import RULES_V14
from .v15 import RULES_V15

# Current canonical ruleset.
RULES = RULES_V15

ALL_VERSIONS = {
    "v12": RULES_V12,
    "v13": RULES_V13,
    "v14": RULES_V14,
    "v15": RULES_V15,
}

__all__ = ["RULES", "ALL_VERSIONS",
           "RULES_V12", "RULES_V13", "RULES_V14", "RULES_V15"]
