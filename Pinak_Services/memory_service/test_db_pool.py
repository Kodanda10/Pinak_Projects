import threading
from app.core.database import DatabaseManager

db = DatabaseManager(":memory:")

def worker():
    with db.get_cursor() as cur:
        cur.execute("SELECT 1")

threads = [threading.Thread(target=worker) for _ in range(10)]
for t in threads: t.start()
for t in threads: t.join()
print("Done")
