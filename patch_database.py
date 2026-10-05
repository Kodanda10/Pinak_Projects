import re

filepath = "Pinak_Services/memory_service/app/core/database.py"
with open(filepath, "r") as f:
    content = f.read()

if "import threading" not in content:
    content = content.replace("import logging", "import logging\nimport threading")

old_init = """    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()"""

new_init = """    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._local = threading.local()
        self._init_db()"""

content = content.replace(old_init, new_init)

old_cursor = """    @contextmanager
    def get_cursor(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()"""

new_cursor = """    @contextmanager
    def get_cursor(self):
        if self.db_path == ":memory:":
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            try:
                yield cur
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
            return

        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row

        conn = self._local.conn
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise"""

content = content.replace(old_cursor, new_cursor)

with open(filepath, "w") as f:
    f.write(content)
