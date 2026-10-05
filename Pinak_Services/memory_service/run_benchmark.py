import time
import os
from app.core.database import DatabaseManager

if os.path.exists("test_bench.db"):
    os.remove("test_bench.db")

db = DatabaseManager("test_bench.db")

start = time.time()
for _ in range(2000):
    with db.get_cursor() as cur:
        cur.execute("SELECT 1")
end = time.time()

print("Time for 2000 operations:", end - start)

if os.path.exists("test_bench.db"):
    os.remove("test_bench.db")
