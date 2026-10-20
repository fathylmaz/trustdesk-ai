"""Demonstration-only high-priority keyword screen for synthetic English tickets.

NOT an authorization engine, fraud detector, or production triage policy.
False positives and false negatives are expected. No model or API is modified.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

# Category-specific, explicit regexes; no implied risk based on model confidence.
_RULES: dict[str, tuple[str, ...]] = {
    "lost_or_stolen_card": (
        r"\b(?:lost|missing|misplaced|stolen|stole|stole[n]?|taken)\b.{0,35}\b(?:bank|debit|credit|payment)?\s*card\b",
        r"\b(?:card)\b.{0,35}\b(?:lost|missing|misplaced|stolen|taken)\b",
    ),
    "account_takeover": (
        r"\b(?:account|banking app|bank account)\b.{0,50}\b(?:hacked|compromised|taken over|accessed without|broken into)\b",
        r"\bsomeone\b.{0,50}\b(?:logged into|accessed)\b.{0,40}\b(?:my account|my banking|account)\b",
    ),
    "suspicious_transaction": (
        r"\b(?:unauthori[sz]ed|fraudulent|unrecognized|unrecognised)\b.{0,45}\b(?:transfer|transaction|withdrawal|payment|charge|purchase)\b",
        r"\b(?:transfer|transaction|withdrawal|payment|charge|purchase)\b.{0,40}\b(?:i didn['’]?t make|i did not make|not mine)\b",
    ),
    "phishing_or_credentials": (
        r"\b(?:phishing|scam link|fake bank website|fake banking website)\b",
        r"\b(?:shared|gave|sent)\b.{0,30}\b(?:pin|password|one.time code|otp)\b",
    ),
    "lost_device_access": (
        r"\b(?:lost|stolen|missing)\b.{0,25}\b(?:phone|mobile)\b.{0,45}\b(?:bank|banking|login|account)\b",
    ),
}
_COMPILED = {name: tuple(re.compile(p, re.IGNORECASE) for p in patterns) for name, patterns in _RULES.items()}


def screen_risk(text: str) -> dict:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    found = [name for name, patterns in _COMPILED.items() if any(p.search(text) for p in patterns)]
    return {"flagged": bool(found), "matched_rules": found}
