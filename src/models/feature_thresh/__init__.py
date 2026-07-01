"""PatchCore threshold-generation package.

Heavy PyQt/rembg modules are intentionally not imported at package import time.
The GUI imports ``threshold_page`` explicitly, while the live runtime imports
only the lightweight scoring/config modules it needs.
"""

__all__ = ["FeatureThresholdPage", "calculate_threshold_for_image"]


def __getattr__(name):
    if name == "FeatureThresholdPage":
        from .threshold_page import FeatureThresholdPage

        return FeatureThresholdPage
    if name == "calculate_threshold_for_image":
        from .threshold_service import calculate_threshold_for_image

        return calculate_threshold_for_image
    raise AttributeError(name)
