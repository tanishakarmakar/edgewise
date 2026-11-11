"""Hard routing rules: things that keep a request on the device no matter what the policy says."""

from __future__ import annotations

import re
import socket
import time

# Deliberately simple patterns. They favor catching personal data over avoiding false alarms:
# a false alarm only costs a local answer, a miss sends personal data off the device.
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email address": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone number": re.compile(r"(?<![\w.])(?:\+\d{1,3}[\s-]?)?(?:\d[\s-]?){9,11}\d(?![\w.])"),
    "card number": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "Aadhaar number": re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),
    "PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "US social security number": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def detect_pii(text: str) -> list[str]:
    """Kinds of personal data found in `text` (empty list if none)."""
    found = []
    for kind, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text)
        if kind == "card number":
            # Long digit runs are common (order ids, odometer readings); require a valid checksum.
            matches = [m for m in matches if _luhn_ok(re.sub(r"\D", "", m))]
        if matches:
            found.append(kind)
    # A valid card number or Aadhaar number also looks like a phone number; report it once.
    if "phone number" in found and {"card number", "Aadhaar number"} & set(found):
        found.remove("phone number")
    return found


class Connectivity:
    """Cached "can we reach the cloud endpoint?" check, so routing doesn't probe every time."""

    def __init__(
        self, host: str = "api.anthropic.com", port: int = 443, timeout_s: float = 1.5,
        ttl_s: float = 30.0,
    ):  # fmt: skip
        self.host, self.port, self.timeout_s, self.ttl_s = host, port, timeout_s, ttl_s
        self._checked_at = float("-inf")
        self._online = False

    def __call__(self) -> bool:
        now = time.monotonic()
        if now - self._checked_at > self.ttl_s:
            self._online = self._probe()
            self._checked_at = now
        return self._online

    def _probe(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout_s):
                return True
        except OSError:
            return False
