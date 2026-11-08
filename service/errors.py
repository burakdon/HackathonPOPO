"""Errors the practice engine and its managers raise; service/patient_api.py maps them to HTTP codes."""
from __future__ import annotations


class Conflict(Exception):
    """409: stale presentation, finished session, a reused client_attempt_id, or a state that forbids the call."""


class NotFound(KeyError):
    """404."""


class Invalid(ValueError):
    """422: the request is well formed but breaks a rule of the contract."""
