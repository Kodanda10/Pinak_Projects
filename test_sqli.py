import sqlite3
from unittest.mock import patch
from Pinak_Services.memory_service.app.core.database import DatabaseManager
import uuid

def test_sqli():
    db = DatabaseManager("test_sqli.db")

    tenant = "t1"
    proj = "p1"

    res = db.add_semantic("hello", [], tenant, proj, 1)
    mid = res["id"]

    # Let's say we want to inject and modify `tenant` for this memory
    try:
        db.update_memory("semantic", mid, {"content='hacked', tenant='t2', content": "dummy"}, tenant, proj)
        print("VULNERABLE!")
    except Exception as e:
        print(f"Error: {e}")

    mem_orig = db.get_memory("semantic", mid, tenant, proj)
    print("Content in t1:", mem_orig)

    mem_new = db.get_memory("semantic", mid, "t2", proj)
    print("Content in t2:", mem_new)

test_sqli()
