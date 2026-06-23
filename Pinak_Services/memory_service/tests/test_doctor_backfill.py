import sqlite3

from cli import doctor


def _init_db(path: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE memories_semantic (
                id TEXT,
                content TEXT,
                client_id TEXT,
                client_name TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO memories_semantic (id, content, client_id, client_name) VALUES (?, ?, ?, ?)",
            ("1", "hello", None, None),
        )
        conn.commit()


def test_doctor_backfills_missing_client_ids(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    _init_db(str(db_path))
    monkeypatch.setattr(doctor, "_get_db_path", lambda: str(db_path))

    report = doctor.run_doctor(fix=True, allow_heavy=False)
    assert any("backfilled client_id" in action for action in report.actions)

    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT client_id, client_name FROM memories_semantic WHERE id = '1'")
        row = cur.fetchone()
        assert row == ("unknown", "unknown")
