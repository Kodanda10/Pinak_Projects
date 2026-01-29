import sqlite3
import json
import logging
import os
import uuid
import datetime
from typing import List, Dict, Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages SQLite connection, schema initialization, and CRUD operations.
    Supports Multi-tenancy via 'tenant' and 'project_id' columns.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize the database schema."""
        conn = self._get_connection()
        try:
            # Enable WAL mode for concurrency
            conn.execute("PRAGMA journal_mode=WAL;")

            with conn:
                # 1. Semantic Memory (Knowledge)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories_semantic (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        tags TEXT, -- JSON array
                        embedding_id INTEGER, -- Link to FAISS
                        tenant TEXT NOT NULL,
                        project_id TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                """)
                # FTS Index for Semantic
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_semantic_fts
                    USING fts5(content, content='memories_semantic', content_rowid='rowid');
                """)
                # Trigger to keep FTS updated
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS memories_semantic_ai AFTER INSERT ON memories_semantic BEGIN
                      INSERT INTO memories_semantic_fts(rowid, content) VALUES (new.rowid, new.content);
                    END;
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS memories_semantic_ad AFTER DELETE ON memories_semantic BEGIN
                      INSERT INTO memories_semantic_fts(memories_semantic_fts, rowid, content) VALUES('delete', old.rowid, old.content);
                    END;
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS memories_semantic_au AFTER UPDATE ON memories_semantic BEGIN
                      INSERT INTO memories_semantic_fts(memories_semantic_fts, rowid, content) VALUES('delete', old.rowid, old.content);
                      INSERT INTO memories_semantic_fts(rowid, content) VALUES (new.rowid, new.content);
                    END;
                """)

                # 2. Episodic Memory (Experiences)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories_episodic (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        goal TEXT,
                        plan TEXT, -- JSON
                        outcome TEXT,
                        tool_logs TEXT, -- JSON
                        salience INTEGER,
                        tenant TEXT NOT NULL,
                        project_id TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                """)
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_episodic_fts
                    USING fts5(content, goal, outcome, content='memories_episodic', content_rowid='rowid');
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS memories_episodic_ai AFTER INSERT ON memories_episodic BEGIN
                      INSERT INTO memories_episodic_fts(rowid, content, goal, outcome) VALUES (new.rowid, new.content, new.goal, new.outcome);
                    END;
                """)

                # 3. Procedural Memory (Skills)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories_procedural (
                        id TEXT PRIMARY KEY,
                        skill_name TEXT NOT NULL,
                        trigger TEXT,
                        steps TEXT, -- JSON
                        code_snippet TEXT,
                        tenant TEXT NOT NULL,
                        project_id TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                """)
                 # FTS for Procedural
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_procedural_fts
                    USING fts5(skill_name, trigger, steps, content='memories_procedural', content_rowid='rowid');
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS memories_procedural_ai AFTER INSERT ON memories_procedural BEGIN
                      INSERT INTO memories_procedural_fts(rowid, skill_name, trigger, steps) VALUES (new.rowid, new.skill_name, new.trigger, new.steps);
                    END;
                """)

                # 4. RAG Memory (External Source)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories_rag (
                        id TEXT PRIMARY KEY,
                        query TEXT,
                        external_source TEXT,
                        content TEXT,
                        tenant TEXT NOT NULL,
                        project_id TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                """)

                # 5. Events (Audit Log)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS logs_events (
                        id TEXT PRIMARY KEY,
                        event_type TEXT,
                        payload TEXT, -- JSON
                        tenant TEXT NOT NULL,
                        project_id TEXT NOT NULL,
                        ts TEXT NOT NULL
                    );
                """)

                # 6. Session
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS logs_session (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        content TEXT,
                        role TEXT, -- user/agent
                        tenant TEXT NOT NULL,
                        project_id TEXT NOT NULL,
                        ts TEXT NOT NULL,
                        expires_at TEXT
                    );
                """)

                # 7. Working Memory
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS working_memory (
                        id TEXT PRIMARY KEY,
                        content TEXT,
                        tenant TEXT NOT NULL,
                        project_id TEXT NOT NULL,
                        ts TEXT NOT NULL,
                        expires_at TEXT
                    );
                """)

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
        finally:
            conn.close()

    @contextmanager
    def get_cursor(self):
        conn = self._get_connection()
        try:
            yield conn.cursor()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # --- CRUD Operations ---

    def add_semantic(self, content: str, tags: List[str], tenant: str, project_id: str, embedding_id: int) -> Dict[str, Any]:
        memory_id = str(uuid.uuid4())
        created_at = datetime.datetime.utcnow().isoformat()
        with self.get_cursor() as cur:
            cur.execute("""
                INSERT INTO memories_semantic (id, content, tags, embedding_id, tenant, project_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (memory_id, content, json.dumps(tags), embedding_id, tenant, project_id, created_at))
        return {
            "id": memory_id, "content": content, "tags": tags, "tenant": tenant, "project_id": project_id, "created_at": created_at
        }

    def add_episodic(self, content: str, tenant: str, project_id: str,
                     goal: str = None, plan: List[str] = None, outcome: str = None,
                     tool_logs: List[Dict] = None, salience: int = 0) -> Dict[str, Any]:
        memory_id = str(uuid.uuid4())
        created_at = datetime.datetime.utcnow().isoformat()
        with self.get_cursor() as cur:
            cur.execute("""
                INSERT INTO memories_episodic (id, content, goal, plan, outcome, tool_logs, salience, tenant, project_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (memory_id, content, goal, json.dumps(plan) if plan else None, outcome,
                  json.dumps(tool_logs) if tool_logs else None, salience, tenant, project_id, created_at))
        return {
            "id": memory_id, "content": content, "goal": goal, "outcome": outcome, "tenant": tenant, "project_id": project_id, "created_at": created_at
        }

    def add_procedural(self, skill_name: str, steps: List[str], tenant: str, project_id: str,
                       trigger: str = None, code_snippet: str = None) -> Dict[str, Any]:
        memory_id = str(uuid.uuid4())
        created_at = datetime.datetime.utcnow().isoformat()
        with self.get_cursor() as cur:
            cur.execute("""
                INSERT INTO memories_procedural (id, skill_name, trigger, steps, code_snippet, tenant, project_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (memory_id, skill_name, trigger, json.dumps(steps), code_snippet, tenant, project_id, created_at))
        return {
            "id": memory_id, "skill_name": skill_name, "tenant": tenant, "project_id": project_id, "created_at": created_at
        }

    def add_rag(self, query: str, external_source: str, content: str, tenant: str, project_id: str) -> Dict[str, Any]:
        memory_id = str(uuid.uuid4())
        created_at = datetime.datetime.utcnow().isoformat()
        with self.get_cursor() as cur:
            cur.execute("""
                INSERT INTO memories_rag (id, query, external_source, content, tenant, project_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (memory_id, query, external_source, content, tenant, project_id, created_at))
        return {"id": memory_id, "query": query, "created_at": created_at}

    def get_memory(self, layer: str, memory_id: str, tenant: str, project_id: str) -> Optional[Dict[str, Any]]:
        table_map = {
            "semantic": "memories_semantic",
            "episodic": "memories_episodic",
            "procedural": "memories_procedural",
            "rag": "memories_rag",
            "working": "working_memory"
        }
        if layer not in table_map:
            return None
        table = table_map[layer]
        with self.get_cursor() as cur:
            cur.execute(f"SELECT * FROM {table} WHERE id = ? AND tenant = ? AND project_id = ?", (memory_id, tenant, project_id))
            row = cur.fetchone()
            if row:
                d = dict(row)
                if layer == "semantic" and d.get('tags'): d['tags'] = json.loads(d['tags'])
                if layer == "episodic":
                    if d.get('plan'): d['plan'] = json.loads(d['plan'])
                    if d.get('tool_logs'): d['tool_logs'] = json.loads(d['tool_logs'])
                if layer == "procedural" and d.get('steps'): d['steps'] = json.loads(d['steps'])
                return d
            return None

    def update_memory(self, layer: str, memory_id: str, updates: Dict[str, Any], tenant: str, project_id: str) -> bool:
        """Updates a memory record."""
        table_map = {
            "semantic": "memories_semantic",
            "episodic": "memories_episodic",
            "procedural": "memories_procedural",
            "rag": "memories_rag",
            "working": "working_memory"
        }
        if layer not in table_map:
            raise ValueError(f"Invalid layer: {layer}")

        table = table_map[layer]

        # Serialize JSON fields
        json_fields = ['tags', 'plan', 'tool_logs', 'steps']
        for k in json_fields:
            if k in updates and isinstance(updates[k], (list, dict)):
                updates[k] = json.dumps(updates[k])

        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values())
        values.extend([memory_id, tenant, project_id])

        with self.get_cursor() as cur:
            cur.execute(f"UPDATE {table} SET {set_clause} WHERE id = ? AND tenant = ? AND project_id = ?", values)
            return cur.rowcount > 0

    def delete_memory(self, layer: str, memory_id: str, tenant: str, project_id: str) -> bool:
        """Deletes a memory from the specified layer."""
        table_map = {
            "semantic": "memories_semantic",
            "episodic": "memories_episodic",
            "procedural": "memories_procedural",
            "rag": "memories_rag",
            "working": "working_memory"
        }
        if layer not in table_map:
            raise ValueError(f"Invalid layer: {layer}")

        table = table_map[layer]
        with self.get_cursor() as cur:
            cur.execute(f"DELETE FROM {table} WHERE id = ? AND tenant = ? AND project_id = ?", (memory_id, tenant, project_id))
            return cur.rowcount > 0

    # --- Search Operations (FTS) ---

    def search_keyword(self, query: str, tenant: str, project_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Performs FTS5 search across Semantic, Episodic, and Procedural layers.
        Returns a unified list of results with a 'type' field.
        """
        results = []

        # Clean query for FTS5 (remove special chars usually, but sqlite handles some)
        # Using simple tokenizer pattern
        safe_query = f'"{query}"'

        with self.get_cursor() as cur:
            # Semantic Search
            cur.execute("""
                SELECT memories_semantic.id, memories_semantic.content, memories_semantic.tags, memories_semantic.created_at, 'semantic' as type, memories_semantic.tenant, memories_semantic.project_id, memories_semantic_fts.rank
                FROM memories_semantic
                JOIN memories_semantic_fts ON memories_semantic.rowid = memories_semantic_fts.rowid
                WHERE memories_semantic_fts MATCH ? AND tenant = ? AND project_id = ?
                ORDER BY rank LIMIT ?
            """, (safe_query, tenant, project_id, limit))
            for row in cur.fetchall():
                d = dict(row)
                if d.get('tags'): d['tags'] = json.loads(d['tags'])
                results.append(d)

            # Episodic Search
            cur.execute("""
                SELECT memories_episodic.id, memories_episodic.content, memories_episodic.plan, memories_episodic.tool_logs, memories_episodic.created_at, 'episodic' as type, memories_episodic.tenant, memories_episodic.project_id, memories_episodic_fts.rank
                FROM memories_episodic
                JOIN memories_episodic_fts ON memories_episodic.rowid = memories_episodic_fts.rowid
                WHERE memories_episodic_fts MATCH ? AND tenant = ? AND project_id = ?
                ORDER BY rank LIMIT ?
            """, (safe_query, tenant, project_id, limit))
            for row in cur.fetchall():
                d = dict(row)
                if d.get('plan'): d['plan'] = json.loads(d['plan'])
                if d.get('tool_logs'): d['tool_logs'] = json.loads(d['tool_logs'])
                results.append(d)

            # Procedural Search
            cur.execute("""
                SELECT memories_procedural.id, memories_procedural.skill_name as content, memories_procedural.steps, memories_procedural.created_at, 'procedural' as type, memories_procedural.tenant, memories_procedural.project_id, memories_procedural_fts.rank
                FROM memories_procedural
                JOIN memories_procedural_fts ON memories_procedural.rowid = memories_procedural_fts.rowid
                WHERE memories_procedural_fts MATCH ? AND tenant = ? AND project_id = ?
                ORDER BY rank LIMIT ?
            """, (safe_query, tenant, project_id, limit))
            for row in cur.fetchall():
                d = dict(row)
                if d.get('steps'): d['steps'] = json.loads(d['steps'])
                results.append(d)

        return results

    def get_semantic_by_embedding_ids(self, embedding_ids: List[int], tenant: str, project_id: str) -> List[Dict[str, Any]]:
        if not embedding_ids:
            return []
        placeholders = ','.join('?' for _ in embedding_ids)
        with self.get_cursor() as cur:
            cur.execute(f"""
                SELECT * FROM memories_semantic
                WHERE embedding_id IN ({placeholders}) AND tenant = ? AND project_id = ?
            """, (*embedding_ids, tenant, project_id))
            rows = []
            for row in cur.fetchall():
                d = dict(row)
                if d.get('tags'): d['tags'] = json.loads(d['tags'])
                rows.append(d)
            return rows

    def add_event(self, event_type: str, payload: Dict, tenant: str, project_id: str) -> Dict[str, Any]:
        event_id = str(uuid.uuid4())
        ts = datetime.datetime.utcnow().isoformat()
        with self.get_cursor() as cur:
            cur.execute("""
                INSERT INTO logs_events (id, event_type, payload, tenant, project_id, ts)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (event_id, event_type, json.dumps(payload), tenant, project_id, ts))
        return {"id": event_id, "ts": ts}

    def list_events(self, tenant: str, project_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self.get_cursor() as cur:
            cur.execute("""
                SELECT * FROM logs_events WHERE tenant = ? AND project_id = ? ORDER BY ts DESC LIMIT ?
            """, (tenant, project_id, limit))
            return [dict(row) for row in cur.fetchall()]

    def add_session(self, session_id: str, content: str, role: str, tenant: str, project_id: str) -> Dict[str, Any]:
        log_id = str(uuid.uuid4())
        ts = datetime.datetime.utcnow().isoformat()
        with self.get_cursor() as cur:
            cur.execute("""
                INSERT INTO logs_session (id, session_id, content, role, tenant, project_id, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (log_id, session_id, content, role, tenant, project_id, ts))
        return {"id": log_id, "ts": ts}

    def list_session(self, session_id: str, tenant: str, project_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self.get_cursor() as cur:
            cur.execute("""
                SELECT * FROM logs_session WHERE session_id = ? AND tenant = ? AND project_id = ? ORDER BY ts ASC LIMIT ?
            """, (session_id, tenant, project_id, limit))
            return [dict(row) for row in cur.fetchall()]

    def add_working(self, content: str, tenant: str, project_id: str) -> Dict[str, Any]:
        memory_id = str(uuid.uuid4())
        ts = datetime.datetime.utcnow().isoformat()
        with self.get_cursor() as cur:
            cur.execute("""
                INSERT INTO working_memory (id, content, tenant, project_id, ts)
                VALUES (?, ?, ?, ?, ?)
            """, (memory_id, content, tenant, project_id, ts))
        return {"id": memory_id, "ts": ts}

    def list_working(self, tenant: str, project_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self.get_cursor() as cur:
            cur.execute("""
                SELECT * FROM working_memory WHERE tenant = ? AND project_id = ? ORDER BY ts DESC LIMIT ?
            """, (tenant, project_id, limit))
            return [dict(row) for row in cur.fetchall()]
