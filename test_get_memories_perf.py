import time
import os
import sqlite3
import numpy as np
import json
from Pinak_Services.memory_service.app.core.database import DatabaseManager

def create_mock_data():
    db_path = "test_perf.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    db = DatabaseManager(db_path)

    tenant = "test_tenant"
    project_id = "test_project"

    n_records = 2000

    with db.get_cursor() as conn:
        for i in range(n_records):
            content = f"Test content for record {i}"
            db.add_semantic(content, [], tenant, project_id, embedding_id=i)

    return db, tenant, project_id

def test_perf(db, tenant, project_id):
    # Test performance of get_memories_by_embedding_ids
    embedding_ids = list(range(500))

    start = time.time()
    db.get_memories_by_embedding_ids(embedding_ids, tenant, project_id)
    end = time.time()

    print(f"Time to fetch 500 embedding_ids: {end - start:.4f} seconds")

if __name__ == "__main__":
    db, t, p = create_mock_data()
    test_perf(db, t, p)
    os.remove("test_perf.db")
