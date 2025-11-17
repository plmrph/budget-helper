"""Constants and configuration patterns for PXBlendSC-RF strategy."""

GENERIC_NOISE_PATTERNS = [
    r"\bpos\b",
    r"\bach\b",
    r"\beft\b",
    r"\bxfer\b",
    r"\bp2p\b",
    r"\bdebit\b",
    r"\bcredit\b",
    r"\bvenmo\b",
    r"\bzelle\b",
    r"\bpaypal\b",
    r"\bstripe\b",
    r"\bsquare\b",
    r"\bcash\s*app\b",
    r"\bauth\b",
    r"\bcapture\b",
    r"\btrx\b",
    r"\btxn\b",
    r"\btransacti?on?\b",
    r"\border\s*id\b",
    r"\bref\w*\b",
    r"\bconf\w*\b",
    r"\bwww\.[a-z0-9\.\-_/]+\b",
    r"\bhttps?://[a-z0-9\.\-_/]+\b",
]

GENERIC_NUM_LOCATION_PATTERNS = [
    r"[#*]{0,2}\d{3,}",
    r"\bno\.\s*\d+\b",
    r"\bstore\s*\d+\b",
    r"\blocation\s*\d+\b",
    r"\bunit\s*\d+\b",
    r"\bapt\s*\d+\b",
    r"\b\d{1,5}\s+[a-z]+\b",
    r"\b[a-z]+,\s*[a-z]{2}\b",
]
