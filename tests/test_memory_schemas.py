from pinak.memory.schemas import (
    json_schema_for_memory, SemanticMemory, ChangelogMemory
)


def test_semantic_memory_model_roundtrip():
    m = SemanticMemory(id="1", layer="semantic", content="hello", tags=["t"]) 
    assert m.layer == "semantic"
    doc = m.model_dump()
    assert doc["content"] == "hello"


def test_json_schema_contains_layers():
    sch = json_schema_for_memory()
    txt = str(sch)
    for name in ("semantic", "working", "event", "changelog", "rag"):
        assert name in txt


def test_changelog_fields_present():
    c = ChangelogMemory(
        id="2", layer="changelog", change_type="create", target_id="1"
    )
    assert c.layer == "changelog"

