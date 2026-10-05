import time
import threading
from app.core.database import DatabaseManager

db = DatabaseManager("test_perf_pool.db")

class ThreadLocalDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self._local = threading.local()

    def get_conn(self):
        if not hasattr(self._local, 'conn'):
            self._local.conn = __import__('sqlite3').connect(self.db_path)
            self._local.conn.row_factory = __import__('sqlite3').Row
        return self._local.conn

tl_db = ThreadLocalDB("test_perf_pool.db")

start = time.time()
for _ in range(1000):
    with db.get_cursor() as cur:
        cur.execute("SELECT 1")
end = time.time()
print("Time without pooling:", end - start)

start = time.time()
for _ in range(1000):
    conn = tl_db.get_conn()
    cur = conn.cursor()
    cur.execute("SELECT 1")
    conn.commit()
    cur.close()
end = time.time()
print("Time WITH pooling:", end - start)

import os
if os.path.exists("test_perf_pool.db"): os.remove("test_perf_pool.db")
