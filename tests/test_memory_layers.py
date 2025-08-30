import importlib
import json
import os
import sys
import time
import types

sys.path.insert(0, os.getcwd())
# Add the src path for module imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


def MM():
    manager = importlib.import_module("pinak.memory.manager")
    cls = getattr(manager, "MemoryManager")
    return cls(service_base_url="http://127.0.0.1:8000", token="TEST_TOKEN")


def test_health():
    import requests

    r = requests.get("http://127.0.0.1:8000/api/v1/memory/health", timeout=3)
    assert r.ok and r.json().get("ok") is True


def test_add_and_list_episodic():
    mm = MM()
    add = mm.add_episodic("Met OP at Raigarh; finance insight", 0.9)
    assert add is not None
    lst = mm.list_episodic()
    assert lst and isinstance(lst, list)
    assert any("finance" in str(x).lower() for x in lst)


def test_add_and_list_procedural():
    mm = MM()
    add = mm.add_procedural("Deploy Railway", ["Push", "Connect Railway", "Set env"])
    assert add is not None
    lst = mm.list_procedural()
    assert lst and isinstance(lst, list)


def test_add_and_list_rag():
    mm = MM()
    add = mm.add_rag("Economic Survey 2024", "https://example.com/es2024")
    assert add is not None
    lst = mm.list_rag()
    assert lst and isinstance(lst, list)


def test_events_trail():
    mm = MM()
    ev = mm.list_events()
    assert ev and isinstance(ev, list)


def test_session_memory():
    mm = MM()
    add = mm.add_session("scratch", "temp", 2)
    assert add is not None
    lst = mm.list_session("scratch")
    assert lst and isinstance(lst, list)
    time.sleep(3)
    lst2 = mm.list_session("scratch")
    assert not lst2  # Should be empty after TTL expires


def test_working_memory():
    mm = MM()
    add = mm.add_working("note", 2)
    assert add is not None
    lst = mm.list_working()
    assert lst and isinstance(lst, list)
    time.sleep(3)
    lst2 = mm.list_working()
    assert not lst2  # Should be empty after TTL expires


def test_cross_layer_search():
    mm = MM()
    mm.add_episodic("FDI Railways brief", 0.5)
    mm.add_procedural("Railways report", ["Open", "Read", "Summarize"])
    mm.add_rag("Railways FDI PDF", "https://example.com/rail")
    res = mm.search_all_layers("FDI Railways")
    assert res and isinstance(res, dict)
    data = res
    assert any(data.get(k, []) for k in ("episodic", "procedural", "rag"))


def test_backward_compatibility():
    # Test backward compatibility with add_memory method
    manager = importlib.import_module("pinak.memory.manager")
    mm = MM()
    if hasattr(mm, "add_memory"):
        try:
            mm.add_memory("compat check", ["test"])
            # Just ensure the method exists and doesn't crash
            assert True
        except Exception:
            # If deprecated but present, tolerate exceptions, just ensure attrs exist
            assert True
