from app.core.database import DatabaseManager

db = DatabaseManager(":memory:")
db.init_db()
db.add_memory("semantic", "123", {"content": "test"}, "t1", "p1")
db.update_memory("semantic", "123", {"content": "new_content", "content = 'hacked'; --": "vuln"}, "t1", "p1")
print(db.get_memory("semantic", "123", "t1", "p1"))
