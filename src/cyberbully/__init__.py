"""Cyberbully: Production-ready Cyberbullying Text Detection package.

Detects cyberbullying, toxic harassment, threats, and subtle mockery with
adversarial leetspeak and nuance robustness.
"""

from __future__ import annotations

from typing import Any, Sequence

from cyberbully.detector import CyberbullyingDetector, PredictionResult
from cyberbully.downloader import download_model
from cyberbully.preprocessing import clean_text

__version__ = "0.1.0"
__all__ = [
    "CyberbullyingDetector",
    "PredictionResult",
    "clean_text",
    "download_model",
    "predict",
    "is_cyberbullying",
    "explain",
    "__version__",
]

_DEFAULT_DETECTOR: CyberbullyingDetector | None = None


def get_default_detector() -> CyberbullyingDetector:
    """Retrieve or lazily initialize the shared global detector instance."""
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = CyberbullyingDetector()
    return _DEFAULT_DETECTOR


def predict(
    texts: str | Sequence[str],
    threshold: float | None = None,
    batch_size: int = 32,
) -> PredictionResult | list[PredictionResult]:
    """Classify text using the default detector instance.
    
    Example:
        >>> import cyberbully
        >>> res = cyberbully.predict("you are awesome!")
        >>> print(res.label)
        'not_cyberbullying'
    """
    detector = get_default_detector()
    return detector.predict(texts, threshold=threshold, batch_size=batch_size)


def is_cyberbullying(text: str, threshold: float | None = None) -> bool:
    """Check if a text is cyberbullying using the default detector instance.
    
    Example:
        >>> import cyberbully
        >>> cyberbully.is_cyberbullying("you are so ugly")
        True
    """
    detector = get_default_detector()
    return detector.is_cyberbullying(text, threshold=threshold)


def explain(text: str, threshold: float | None = None) -> dict[str, Any]:
    """Generate detailed breakdown and top category scores for a text."""
    detector = get_default_detector()
    return detector.explain(text, threshold=threshold)
