import json
from pinak.ledger.hash_chain import append_entry, verify_chain


def test_hash_chain_append_and_verify(tmp_path):
    p = tmp_path / "ledger.jsonl"
    h1 = append_entry(str(p), {"a": 1})
    h2 = append_entry(str(p), {"b": 2}, prev_hash=h1)
    assert h1 != h2
    assert verify_chain(str(p)) is True

