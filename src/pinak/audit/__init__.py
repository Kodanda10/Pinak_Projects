"""
pinak.audit
Convenience import for the security auditor.
"""
try:
    from pinak.security.auditor import SecurityAuditor as Auditor
except Exception:  # pragma: no cover
    Auditor = None  # type: ignore

__all__ = ["Auditor"]

