"""
pinak.changelog
Helpers for changelog (append-only WORM ledger) operations.
"""
from pinak.ledger.hash_chain import append_entry as log_change, verify_chain

__all__ = [
    "log_change",
    "verify_chain",
]

