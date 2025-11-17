"""Text normalization and processing for PXBlendSC-RF strategy."""

import re

from .constants import GENERIC_NOISE_PATTERNS, GENERIC_NUM_LOCATION_PATTERNS


def make_generic_normalizer(use_generic_noise: bool, alias_map: dict):
    """Create a generic text normalizer (returns a picklable class instance)."""
    return GenericNormalizer(use_generic_noise, alias_map)


class GenericNormalizer:
    """Picklable text normalizer for payee names."""

    def __init__(self, use_generic_noise: bool = True, alias_map: dict = None):
        self.use_generic_noise = use_generic_noise
        self.alias_map = alias_map or {}

        # Compile regex patterns
        if use_generic_noise:
            noise_patterns = list(GENERIC_NOISE_PATTERNS) + list(
                GENERIC_NUM_LOCATION_PATTERNS
            )
            self.compiled_patterns = [
                re.compile(pat, re.IGNORECASE) for pat in noise_patterns
            ]
        else:
            self.compiled_patterns = []

    def __call__(self, s: str) -> str:
        """Normalize a string."""
        if not isinstance(s, str):
            return ""

        s = s.lower()

        # Apply noise pattern removal
        if self.use_generic_noise:
            for rx in self.compiled_patterns:
                s = rx.sub(" ", s)

        # Remove special characters
        s = re.sub(r"[\|\[\]\(\)_~`^=:;@<>\"" "']", " ", s)
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()

        # Apply alias replacements
        for pat, repl in self.alias_map.items():
            try:
                s = re.sub(pat, repl, s)
            except re.error:
                if s == pat:
                    s = repl

        return s
