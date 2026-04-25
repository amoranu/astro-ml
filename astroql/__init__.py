"""AstroQL: per-tradition query planner + RAG-backed reasoning.

See PHASE_F_ARCHITECTURE.md for the per-tradition planner design.

The neuro-symbolic CF engine (CF inference, dasha emitter, MYCIN
math, JSON DSL, Ashtakavarga gating, regression / commit gates,
LLM critic) was spun out into its own repo:

    https://github.com/amoranu/neuro-symbolic-astro

What stays here:
  * `engine.RuleEngine` — legacy clause-based rule evaluator
  * `parser`, `planner`, `rag`, `api`, `benchmark`, `discovery`,
    `inverted_mining`, `sensitivity`, `resolver`, `resolver_engine`,
    `explainer`, `features`, `chart`, `query_plans`, `cli`
  * `schemas/`, `rules/loader.py`, `rules/<school>/*.yaml` —
    DUPLICATED with neuro-symbolic-astro so the new engine is
    standalone. Keep both copies in sync when schema changes land.
"""

__version__ = "0.2.0-dev"
