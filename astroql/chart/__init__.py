"""Chart computation — thin wrapper over astro-prod's AstroEngine.

See memory/astro_engine_source.md for the design decision to delegate to
astro-prod (5444-line class with shadbala, yogas, doshas, KP details,
arudhas etc.) rather than re-implementing.
"""
from .computer import ChartComputer, ChartComputerError

__all__ = ["ChartComputer", "ChartComputerError"]
