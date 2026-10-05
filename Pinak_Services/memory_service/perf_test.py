import time
from app.core.database import DatabaseManager

db = DatabaseManager("test_perf.db")

start = time.time()
for _ in range(1000):
    with db.get_cursor() as cur:
        cur.execute("SELECT 1")
end = time.time()
print("Time without pooling:", end - start)
