"""Feature extraction per school (spec §6.4).

Each school gets its own extractor. All extractors take
(Chart, ResolvedFocus) and return a FeatureBundle.
"""
from .jaimini import JaiminiFeatureExtractor
from .kp import KPFeatureExtractor
from .parashari import ParashariFeatureExtractor

__all__ = [
    "ParashariFeatureExtractor",
    "KPFeatureExtractor",
    "JaiminiFeatureExtractor",
]
