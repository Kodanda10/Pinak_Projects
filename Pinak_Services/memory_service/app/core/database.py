import os
import json
import uuid
import datetime
import logging
import re
from typing import List, Dict, Any, Optional
from sqlalchemy import select, update, delete, text, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.async_db import AsyncSessionLocal, engine
from app.core.models import (
    Base, SemanticMemory, EpisodicMemory, ProceduralMemory, RAGMemory, WorkingMemory,
    EventLog, SessionLog, ClientRegistry, AgentLog, AccessLog, Quarantine, AuditLog, ClientIssue
)

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        # Initialization is now async and explicit via init_db()

    async def init_db(self):
        """Async initialization of database tables and FTS."""
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

            # FTS Setup (Raw SQL required for Virtual Tables)
            # 1. Semantic
            await conn.execute(text("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_semantic_fts 
                USING fts5(content, tags, content='memories_semantic', content_rowid='rowid');
            """))
            await conn.execute(text("""
                CREATE TRIGGER IF NOT EXISTS memories_semantic_ai AFTER INSERT ON memories_semantic BEGIN
                  INSERT INTO memories_semantic_fts(rowid, content, tags) VALUES (new.rowid, new.content, new.tags);
                END;
            """))

            # 2. Episodic
            await conn.execute(text("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_episodic_fts
                USING fts5(content, goal, outcome, content='memories_episodic', content_rowid='rowid');
            """))
            await conn.execute(text("""
                CREATE TRIGGER IF NOT EXISTS memories_episodic_ai AFTER INSERT ON memories_episodic BEGIN
                  INSERT INTO memories_episodic_fts(rowid, content, goal, outcome) VALUES (new.rowid, new.content, new.goal, new.outcome);
                END;
            """))

            # 3. Procedural
            await conn.execute(text("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_procedural_fts
                USING fts5(skill_name, trigger, steps, description, content='memories_procedural', content_rowid='rowid');
            """))
            await conn.execute(text("""
                CREATE TRIGGER IF NOT EXISTS memories_procedural_ai AFTER INSERT ON memories_procedural BEGIN
                  INSERT INTO memories_procedural_fts(rowid, skill_name, trigger, steps, description) VALUES (new.rowid, new.skill_name, new.trigger, new.steps, new.description);
                END;
            """))

    def _sanitize_fts_query(self, query: str) -> str:
        terms = []
        for raw in (query or "").split():
            token = raw.strip().replace('"', "").replace("'", "")
            if not token:
                continue
            if re.search(r"[^A-Za-z0-9_]", token):
                terms.append(f"\"{token}\"")
            else:
                terms.append(token)
        return " ".join(terms) if terms else (query or "")

    async def add_semantic(self, content: str, tags: list, tenant: str, project_id: str, embedding_id: int,
                     agent_id: Optional[str] = None, client_id: Optional[str] = None,
                     client_name: Optional[str] = None) -> Dict[str, Any]:
        mid = str(uuid.uuid4())
        created_at = datetime.datetime.now().isoformat()

        async with AsyncSessionLocal() as session:
            item = SemanticMemory(
                id=mid, content=content, tags=json.dumps(tags), embedding_id=embedding_id,
                agent_id=agent_id, client_id=client_id, client_name=client_name,
                tenant=tenant, project_id=project_id, created_at=created_at
            )
            session.add(item)
            await session.commit()

        return {
            "id": mid,
            "content": content,
            "tags": tags,
            "tenant": tenant,
            "project_id": project_id,
            "created_at": created_at,
            "agent_id": agent_id,
            "client_id": client_id,
            "client_name": client_name,
        }

    async def search_keyword(self, query: str, tenant: str, project_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        fts_query = self._sanitize_fts_query(query)
        results = []

        async with AsyncSessionLocal() as session:
            # Semantic FTS
            res = await session.execute(text("""
                SELECT m.id, m.content, m.tags, 'semantic' as type
                FROM memories_semantic m
                JOIN memories_semantic_fts f ON m.rowid = f.rowid
                WHERE memories_semantic_fts MATCH :q AND m.tenant = :t AND m.project_id = :p
                ORDER BY f.rank LIMIT :l
            """), {"q": fts_query, "t": tenant, "p": project_id, "l": limit})

            for row in res.fetchall():
                d = dict(row._mapping)
                if d.get("tags"):
                    try:
                        d["tags"] = json.loads(d["tags"])
                    except:
                        pass
                results.append(d)

            # Episodic FTS
            res = await session.execute(text("""
                SELECT m.id, m.content, m.goal, m.outcome, m.plan, m.steps, 'episodic' as type
                FROM memories_episodic m
                JOIN memories_episodic_fts f ON m.rowid = f.rowid
                WHERE memories_episodic_fts MATCH :q AND m.tenant = :t AND m.project_id = :p
                ORDER BY f.rank LIMIT :l
            """), {"q": fts_query, "t": tenant, "p": project_id, "l": limit})

            for row in res.fetchall():
                d = dict(row._mapping)
                if d.get("plan"):
                    try:
                        d["plan"] = json.loads(d["plan"])
                    except:
                        pass
                if d.get("steps"):
                    try:
                        d["steps"] = json.loads(d["steps"])
                        d["tool_logs"] = d["steps"]
                    except:
                        pass
                results.append(d)

            # Procedural FTS
            res = await session.execute(text("""
                SELECT m.id, m.skill_name as content, m.description, m.steps, 'procedural' as type 
                FROM memories_procedural m
                JOIN memories_procedural_fts f ON m.rowid = f.rowid
                WHERE memories_procedural_fts MATCH :q AND m.tenant = :t AND m.project_id = :p
                ORDER BY f.rank LIMIT :l
            """), {"q": fts_query, "t": tenant, "p": project_id, "l": limit})

            for row in res.fetchall():
                d = dict(row._mapping)
                if d.get("steps"):
                    try:
                        d["steps"] = json.loads(d["steps"])
                    except:
                        pass
                results.append(d)

        return results

    async def add_episodic(self, content: str, tenant: str, project_id: str,
                     salience: int = 0, goal: Optional[str] = None,
                     plan: Optional[list] = None, tool_logs: Optional[list] = None,
                     outcome: Optional[str] = None, embedding_id: Optional[int] = None,
                     agent_id: Optional[str] = None, client_id: Optional[str] = None,
                     client_name: Optional[str] = None) -> Dict[str, Any]:
        mid = str(uuid.uuid4())
        created_at = datetime.datetime.now().isoformat()
        steps = tool_logs or []

        async with AsyncSessionLocal() as session:
            item = EpisodicMemory(
                id=mid, content=content, goal=goal, outcome=outcome,
                plan=json.dumps(plan), steps=json.dumps(steps), salience=salience,
                embedding_id=embedding_id, agent_id=agent_id, client_id=client_id,
                client_name=client_name, tenant=tenant, project_id=project_id,
                created_at=created_at
            )
            session.add(item)
            await session.commit()

        return {
            "id": mid,
            "goal": goal,
            "agent_id": agent_id,
            "client_id": client_id,
            "client_name": client_name,
        }

    async def upsert_agent(self, agent_id: str, client_name: str, status: str, tenant: str, project_id: str,
                     hostname: Optional[str] = None, pid: Optional[str] = None, meta: Optional[Dict[str, Any]] = None,
                     client_id: Optional[str] = None, parent_client_id: Optional[str] = None) -> Dict[str, Any]:
        mid = str(uuid.uuid4())
        last_seen = datetime.datetime.now().isoformat()
        meta_json = json.dumps(meta or {})

        async with AsyncSessionLocal() as session:
            # Upsert logic for SQLite/Postgres
            # Check exist first
            stmt = select(AgentLog).where(
                AgentLog.agent_id == agent_id,
                AgentLog.client_name == client_name,
                AgentLog.tenant == tenant,
                AgentLog.project_id == project_id
            )
            res = await session.execute(stmt)
            existing = res.scalar_one_or_none()

            if existing:
                existing.hostname = hostname
                existing.pid = pid
                existing.status = status
                existing.meta = meta_json
                existing.client_id = client_id
                existing.parent_client_id = parent_client_id
                existing.last_seen = last_seen
            else:
                session.add(AgentLog(
                    id=mid, agent_id=agent_id, client_name=client_name, status=status,
                    hostname=hostname, pid=pid, meta=meta_json, client_id=client_id,
                    parent_client_id=parent_client_id, tenant=tenant, project_id=project_id,
                    last_seen=last_seen
                ))
            await session.commit()

        return {"agent_id": agent_id, "client_name": client_name, "status": status, "last_seen": last_seen, "client_id": client_id, "parent_client_id": parent_client_id}

    async def list_agents(self, tenant: str, project_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            stmt = select(AgentLog).where(
                AgentLog.tenant == tenant, AgentLog.project_id == project_id
            ).order_by(AgentLog.last_seen.desc()).limit(limit)
            res = await session.execute(stmt)
            rows = []
            for item in res.scalars().all():
                d = {c.name: getattr(item, c.name) for c in item.__table__.columns}
                if d.get("meta"):
                    try:
                        d["meta"] = json.loads(d["meta"])
                    except:
                        d["meta"] = {}
                rows.append(d)
            return rows

    async def add_access_event(self, event_type: str, status: str, tenant: str, project_id: str,
                         agent_id: Optional[str] = None, client_name: Optional[str] = None,
                         client_id: Optional[str] = None, parent_client_id: Optional[str] = None,
                         child_client_id: Optional[str] = None, target_layer: Optional[str] = None,
                         query: Optional[str] = None, memory_id: Optional[str] = None,
                         result_count: Optional[int] = None, detail: Optional[str] = None) -> Dict[str, Any]:
        mid = str(uuid.uuid4())
        ts = datetime.datetime.now().isoformat()

        async with AsyncSessionLocal() as session:
            session.add(AccessLog(
                id=mid, agent_id=agent_id, client_name=client_name, client_id=client_id,
                parent_client_id=parent_client_id, child_client_id=child_client_id,
                event_type=event_type, target_layer=target_layer, query=query,
                memory_id=memory_id, result_count=result_count, status=status, detail=detail,
                tenant=tenant, project_id=project_id, ts=ts
            ))
            await session.commit()

        await self.add_audit_event(
            event_type=f"access:{event_type}",
            payload={
                "agent_id": agent_id,
                "client_name": client_name,
                "client_id": client_id,
                "parent_client_id": parent_client_id,
                "child_client_id": child_client_id,
                "target_layer": target_layer,
                "query": query,
                "memory_id": memory_id,
                "result_count": result_count,
                "status": status,
                "detail": detail,
                "tenant": tenant,
                "project_id": project_id,
                "ts": ts,
            },
        )
        return {"id": mid, "event_type": event_type, "status": status, "ts": ts}

    async def list_access_events(self, tenant: str, project_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            stmt = select(AccessLog).where(
                AccessLog.tenant == tenant, AccessLog.project_id == project_id
            ).order_by(AccessLog.ts.desc()).limit(limit)
            res = await session.execute(stmt)
            return [{c.name: getattr(i, c.name) for c in i.__table__.columns} for i in res.scalars().all()]

    async def add_quarantine(self, layer: str, payload: Dict[str, Any], tenant: str, project_id: str,
                       agent_id: Optional[str] = None, client_id: Optional[str] = None,
                       client_name: Optional[str] = None, validation_errors: Optional[List[str]] = None) -> Dict[str, Any]:
        mid = str(uuid.uuid4())
        created_at = datetime.datetime.now().isoformat()

        async with AsyncSessionLocal() as session:
            session.add(Quarantine(
                id=mid, layer=layer, payload=json.dumps(payload), status='pending',
                agent_id=agent_id, client_id=client_id, client_name=client_name,
                validation_errors=json.dumps(validation_errors or []),
                tenant=tenant, project_id=project_id, created_at=created_at
            ))
            await session.commit()

        await self.add_audit_event(
            event_type="quarantine:create",
            payload={"id": mid, "layer": layer, "tenant": tenant, "project_id": project_id},
        )
        return {"id": mid, "status": "pending", "layer": layer}

    async def list_quarantine(self, tenant: str, project_id: str, status: str = "pending", limit: int = 100) -> List[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            stmt = select(Quarantine).where(
                Quarantine.tenant == tenant, Quarantine.project_id == project_id, Quarantine.status == status
            ).order_by(Quarantine.created_at.desc()).limit(limit)
            res = await session.execute(stmt)
            rows = []
            for item in res.scalars().all():
                d = {c.name: getattr(item, c.name) for c in item.__table__.columns}
                if d.get("payload"):
                    try:
                        d["payload"] = json.loads(d["payload"])
                    except:
                        d["payload"] = {}
                if d.get("validation_errors"):
                    try:
                        d["validation_errors"] = json.loads(d["validation_errors"])
                    except:
                        d["validation_errors"] = []
                rows.append(d)
            return rows

    async def resolve_quarantine(self, item_id: str, status: str, reviewer: str) -> Optional[Dict[str, Any]]:
        reviewed_at = datetime.datetime.now().isoformat()

        async with AsyncSessionLocal() as session:
            stmt = select(Quarantine).where(Quarantine.id == item_id)
            res = await session.execute(stmt)
            item = res.scalar_one_or_none()
            if not item:
                return None

            item.status = status
            item.reviewed_at = reviewed_at
            item.reviewed_by = reviewer
            await session.commit()

            d = {c.name: getattr(item, c.name) for c in item.__table__.columns}
            if d.get("payload"):
                try:
                    d["payload"] = json.loads(d["payload"])
                except:
                    d["payload"] = {}

        await self.add_audit_event(
            event_type=f"quarantine:{status}",
            payload={"id": item_id, "reviewed_by": reviewer},
        )
        return d

    async def add_audit_event(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        mid = str(uuid.uuid4())
        ts = datetime.datetime.now().isoformat()
        payload_json = json.dumps(payload, sort_keys=True)
        import hashlib

        async with AsyncSessionLocal() as session:
            stmt = select(AuditLog).order_by(AuditLog.ts.desc()).limit(1)
            res = await session.execute(stmt)
            prev = res.scalar_one_or_none()
            prev_hash = prev.hash if prev else ""

            h = hashlib.sha256(f"{prev_hash}|{event_type}|{payload_json}|{ts}".encode("utf-8")).hexdigest()

            session.add(AuditLog(
                id=mid, event_type=event_type, payload=payload_json,
                prev_hash=prev_hash, hash=h, ts=ts
            ))
            await session.commit()

        return {"id": mid, "hash": h, "ts": ts}

    async def add_procedural(self, skill_name: str, steps: list, tenant: str, project_id: str,
                       description: Optional[str] = None, trigger: Optional[str] = None,
                       code_snippet: Optional[str] = None, embedding_id: Optional[int] = None,
                       agent_id: Optional[str] = None, client_id: Optional[str] = None,
                       client_name: Optional[str] = None) -> Dict[str, Any]:
        mid = str(uuid.uuid4())
        created_at = datetime.datetime.now().isoformat()

        async with AsyncSessionLocal() as session:
            item = ProceduralMemory(
                id=mid, skill_name=skill_name, trigger=trigger, steps=json.dumps(steps),
                description=description, code_snippet=code_snippet, embedding_id=embedding_id,
                agent_id=agent_id, client_id=client_id, client_name=client_name,
                tenant=tenant, project_id=project_id, created_at=created_at
            )
            session.add(item)
            await session.commit()

        return {"id": mid, "skill_name": skill_name}

    async def add_rag(self, query: str, external_source: str, content: str, tenant: str, project_id: str,
                agent_id: Optional[str] = None, client_id: Optional[str] = None,
                client_name: Optional[str] = None) -> Dict[str, Any]:
        mid = str(uuid.uuid4())
        created_at = datetime.datetime.now().isoformat()

        async with AsyncSessionLocal() as session:
            session.add(RAGMemory(
                id=mid, query=query, external_source=external_source, content=content,
                agent_id=agent_id, client_id=client_id, client_name=client_name,
                tenant=tenant, project_id=project_id, created_at=created_at
            ))
            await session.commit()

        return {"id": mid, "query": query}

    async def get_memories_by_embedding_ids(self, embedding_ids: List[int], tenant: str, project_id: str) -> List[Dict[str, Any]]:
        if not embedding_ids:
            return []
        
        results = []
        async with AsyncSessionLocal() as session:
            # Semantic
            stmt = select(SemanticMemory).where(
                SemanticMemory.embedding_id.in_(embedding_ids),
                SemanticMemory.tenant == tenant,
                SemanticMemory.project_id == project_id
            )
            res = await session.execute(stmt)
            for r in res.scalars().all():
                d = {c.name: getattr(r, c.name) for c in r.__table__.columns}
                d['type'] = 'semantic'
                if d.get('tags'):
                    try:
                        d['tags'] = json.loads(d['tags'])
                    except:
                        pass
                results.append(d)

            # Episodic
            stmt = select(EpisodicMemory).where(
                EpisodicMemory.embedding_id.in_(embedding_ids),
                EpisodicMemory.tenant == tenant,
                EpisodicMemory.project_id == project_id
            )
            res = await session.execute(stmt)
            for r in res.scalars().all():
                d = {c.name: getattr(r, c.name) for c in r.__table__.columns}
                d['type'] = 'episodic'
                if d.get('plan'):
                    try:
                        d['plan'] = json.loads(d['plan'])
                    except:
                        pass
                if d.get('steps'):
                    try:
                        d['steps'] = json.loads(d['steps'])
                    except:
                        pass
                results.append(d)

            # Procedural
            stmt = select(ProceduralMemory).where(
                ProceduralMemory.embedding_id.in_(embedding_ids),
                ProceduralMemory.tenant == tenant,
                ProceduralMemory.project_id == project_id
            )
            res = await session.execute(stmt)
            for r in res.scalars().all():
                d = {c.name: getattr(r, c.name) for c in r.__table__.columns}
                d['type'] = 'procedural'
                d['content'] = d['skill_name']
                if d.get('steps'):
                    try:
                        d['steps'] = json.loads(d['steps'])
                    except:
                        pass
                results.append(d)

        return results

    async def get_memory(self, layer: str, memory_id: str, tenant: str, project_id: str) -> Optional[Dict[str, Any]]:
        model_map = {
            "semantic": SemanticMemory,
            "episodic": EpisodicMemory,
            "procedural": ProceduralMemory,
            "rag": RAGMemory,
            "working": WorkingMemory,
        }
        Model = model_map.get(layer)
        if not Model:
            return None

        async with AsyncSessionLocal() as session:
            stmt = select(Model).where(Model.id == memory_id, Model.tenant == tenant, Model.project_id == project_id)
            res = await session.execute(stmt)
            item = res.scalar_one_or_none()
            if not item: return None

            d = {c.name: getattr(item, c.name) for c in item.__table__.columns}
            # Deserialize
            if d.get("tags"):
                try:
                    d["tags"] = json.loads(d["tags"])
                except:
                    pass
            if d.get("plan"):
                try:
                    d["plan"] = json.loads(d["plan"])
                except:
                    pass
            if d.get("steps"):
                try:
                    d["steps"] = json.loads(d["steps"])
                    d["tool_logs"] = d["steps"]
                except:
                    pass
            return d

    async def update_memory(self, layer: str, memory_id: str, updates: Dict[str, Any], tenant: str, project_id: str) -> bool:
        model_map = {
            "semantic": SemanticMemory,
            "episodic": EpisodicMemory,
            "procedural": ProceduralMemory,
            "rag": RAGMemory,
        }
        Model = model_map.get(layer)
        if not Model: raise ValueError("Invalid layer")

        serialized = {}
        for key, value in updates.items():
            if key in ("tags", "plan", "steps") and value is not None:
                serialized[key] = json.dumps(value)
            else:
                serialized[key] = value

        async with AsyncSessionLocal() as session:
            stmt = update(Model).where(Model.id == memory_id, Model.tenant == tenant, Model.project_id == project_id).values(**serialized)
            res = await session.execute(stmt)
            await session.commit()
            return res.rowcount > 0

    async def delete_memory(self, layer: str, memory_id: str, tenant: str, project_id: str) -> bool:
        model_map = {
            "semantic": SemanticMemory,
            "episodic": EpisodicMemory,
            "procedural": ProceduralMemory,
            "rag": RAGMemory,
        }
        Model = model_map.get(layer)
        if not Model: raise ValueError("Invalid layer")

        async with AsyncSessionLocal() as session:
            stmt = delete(Model).where(Model.id == memory_id, Model.tenant == tenant, Model.project_id == project_id)
            res = await session.execute(stmt)
            await session.commit()
            return res.rowcount > 0

    async def list_working(self, tenant: str, project_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            stmt = select(WorkingMemory).where(WorkingMemory.tenant == tenant, WorkingMemory.project_id == project_id).order_by(WorkingMemory.updated_at.desc()).limit(limit)
            res = await session.execute(stmt)
            return [{c.name: getattr(i, c.name) for c in i.__table__.columns} for i in res.scalars().all()]

    async def add_event(self, event_type: str, payload: Dict, tenant: str, project_id: str) -> Dict[str, Any]:
        event_id = str(uuid.uuid4())
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        async with AsyncSessionLocal() as session:
            session.add(EventLog(
                id=event_id, event_type=event_type, payload=json.dumps(payload),
                tenant=tenant, project_id=project_id, ts=ts
            ))
            await session.commit()
        return {"id": event_id, "ts": ts}

    async def list_events(self, tenant: str, project_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            stmt = select(EventLog).where(EventLog.tenant == tenant, EventLog.project_id == project_id).order_by(EventLog.ts.desc()).limit(limit)
            res = await session.execute(stmt)
            return [{c.name: getattr(i, c.name) for c in i.__table__.columns} for i in res.scalars().all()]

    async def add_session(self, session_id: str, content: str, role: str, tenant: str, project_id: str,
                    agent_id: Optional[str] = None, client_id: Optional[str] = None,
                    client_name: Optional[str] = None, parent_client_id: Optional[str] = None,
                    child_client_id: Optional[str] = None) -> Dict[str, Any]:
        log_id = str(uuid.uuid4())
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        async with AsyncSessionLocal() as session:
            session.add(SessionLog(
                id=log_id, session_id=session_id, content=content, role=role,
                agent_id=agent_id, client_id=client_id, client_name=client_name,
                parent_client_id=parent_client_id, child_client_id=child_client_id,
                tenant=tenant, project_id=project_id, ts=ts
            ))
            await session.commit()
        return {"id": log_id, "ts": ts}

    async def list_session(self, session_id: str, tenant: str, project_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            stmt = select(SessionLog).where(
                SessionLog.session_id == session_id, SessionLog.tenant == tenant, SessionLog.project_id == project_id
            ).order_by(SessionLog.ts.asc()).limit(limit)
            res = await session.execute(stmt)
            return [{c.name: getattr(i, c.name) for c in i.__table__.columns} for i in res.scalars().all()]

    async def observe_client(self, client_id: str, tenant: str, project_id: str,
                       client_name: Optional[str] = None, parent_client_id: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cid = str(uuid.uuid4())
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {})

        async with AsyncSessionLocal() as session:
            stmt = select(ClientRegistry).where(
                ClientRegistry.client_id == client_id,
                ClientRegistry.tenant == tenant,
                ClientRegistry.project_id == project_id
            )
            res = await session.execute(stmt)
            existing = res.scalar_one_or_none()

            if existing:
                if client_name: existing.client_name = client_name
                if parent_client_id: existing.parent_client_id = parent_client_id
                if metadata: existing.metadata_ = meta_json
                existing.updated_at = now
                existing.last_seen = now
            else:
                session.add(ClientRegistry(
                    id=cid, client_id=client_id, client_name=client_name, parent_client_id=parent_client_id,
                    status='observed', metadata_=meta_json, tenant=tenant, project_id=project_id,
                    created_at=now, updated_at=now, last_seen=now
                ))
            await session.commit()

        return {"client_id": client_id, "status": "observed", "updated_at": now}

    async def register_client(self, client_id: str, tenant: str, project_id: str,
                        client_name: Optional[str] = None, parent_client_id: Optional[str] = None,
                        status: str = "registered", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cid = str(uuid.uuid4())
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {})

        async with AsyncSessionLocal() as session:
            stmt = select(ClientRegistry).where(
                ClientRegistry.client_id == client_id,
                ClientRegistry.tenant == tenant,
                ClientRegistry.project_id == project_id
            )
            res = await session.execute(stmt)
            existing = res.scalar_one_or_none()

            if existing:
                if client_name: existing.client_name = client_name
                if parent_client_id: existing.parent_client_id = parent_client_id
                existing.status = status
                existing.metadata_ = meta_json
                existing.updated_at = now
                existing.last_seen = now
            else:
                session.add(ClientRegistry(
                    id=cid, client_id=client_id, client_name=client_name, parent_client_id=parent_client_id,
                    status=status, metadata_=meta_json, tenant=tenant, project_id=project_id,
                    created_at=now, updated_at=now, last_seen=now
                ))
            await session.commit()

        return {"client_id": client_id, "status": status, "updated_at": now}

    async def get_client(self, client_id: str, tenant: str, project_id: str) -> Optional[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            stmt = select(ClientRegistry).where(
                ClientRegistry.client_id == client_id,
                ClientRegistry.tenant == tenant,
                ClientRegistry.project_id == project_id
            )
            res = await session.execute(stmt)
            item = res.scalar_one_or_none()
            if not item: return None

            d = {}
            for c in item.__table__.columns:
                attr = "metadata_" if c.name == "metadata" else c.name
                d[c.name] = getattr(item, attr)

            if d.get("metadata"): # it is stored as 'metadata' key in dict now (string value)
                try:
                    d["metadata"] = json.loads(d["metadata"])
                except:
                    d["metadata"] = {}
            return d

    async def list_clients(self, tenant: str, project_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            stmt = select(ClientRegistry).where(
                ClientRegistry.tenant == tenant, ClientRegistry.project_id == project_id
            ).order_by(ClientRegistry.last_seen.desc()).limit(limit)
            res = await session.execute(stmt)
            rows = []
            for item in res.scalars().all():
                d = {}
                for c in item.__table__.columns:
                    attr = "metadata_" if c.name == "metadata" else c.name
                    d[c.name] = getattr(item, attr)

                if d.get("metadata"):
                    try:
                        d["metadata"] = json.loads(d["metadata"])
                    except:
                        d["metadata"] = {}
                rows.append(d)
            return rows

    async def list_child_clients(self, parent_client_id: str, tenant: str, project_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            stmt = select(ClientRegistry).where(
                ClientRegistry.parent_client_id == parent_client_id,
                ClientRegistry.tenant == tenant,
                ClientRegistry.project_id == project_id
            ).order_by(ClientRegistry.last_seen.desc()).limit(limit)
            res = await session.execute(stmt)
            rows = []
            for item in res.scalars().all():
                d = {}
                for c in item.__table__.columns:
                    attr = "metadata_" if c.name == "metadata" else c.name
                    d[c.name] = getattr(item, attr)

                if d.get("metadata"):
                    try:
                        d["metadata"] = json.loads(d["metadata"])
                    except:
                        d["metadata"] = {}
                rows.append(d)
            return rows

    async def get_client_layer_stats(self, client_id: str, tenant: str, project_id: str) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        last_write: Dict[str, Optional[str]] = {}

        tables = {
            "semantic": (SemanticMemory, "created_at"),
            "episodic": (EpisodicMemory, "created_at"),
            "procedural": (ProceduralMemory, "created_at"),
            "rag": (RAGMemory, "created_at"),
            "working": (WorkingMemory, "updated_at"),
        }

        async with AsyncSessionLocal() as session:
            for layer, (Model, ts_col_name) in tables.items():
                ts_col = getattr(Model, ts_col_name)
                stmt = select(func.count(), func.max(ts_col)).where(
                    Model.tenant == tenant, Model.project_id == project_id, Model.client_id == client_id
                )
                res = await session.execute(stmt)
                row = res.fetchone()
                counts[layer] = int(row[0] or 0)
                last_write[layer] = row[1]

        return {
            "counts": counts,
            "last_write": last_write,
            "total": sum(counts.values()),
        }

    async def count_client_issues(self, client_id: str, tenant: str, project_id: str, status: str = "open") -> int:
        async with AsyncSessionLocal() as session:
            stmt = select(func.count()).where(
                ClientIssue.client_id == client_id,
                ClientIssue.tenant == tenant,
                ClientIssue.project_id == project_id,
                ClientIssue.status == status
            )
            res = await session.execute(stmt)
            return res.scalar() or 0

    async def count_quarantine(self, client_id: str, tenant: str, project_id: str, status: str = "pending") -> int:
        async with AsyncSessionLocal() as session:
            stmt = select(func.count()).where(
                Quarantine.client_id == client_id,
                Quarantine.tenant == tenant,
                Quarantine.project_id == project_id,
                Quarantine.status == status
            )
            res = await session.execute(stmt)
            return res.scalar() or 0

    async def add_working(self, content: str, tenant: str, project_id: str,
                    agent_id: Optional[str] = None, client_id: Optional[str] = None,
                    client_name: Optional[str] = None) -> Dict[str, Any]:
        wid = str(uuid.uuid4())
        updated_at = datetime.datetime.now().isoformat()
        key = "current_context"

        async with AsyncSessionLocal() as session:
            # Check exist
            stmt = select(WorkingMemory).where(
                WorkingMemory.session_id == 'global',
                WorkingMemory.key == key,
                WorkingMemory.tenant == tenant,
                WorkingMemory.project_id == project_id
            )
            res = await session.execute(stmt)
            existing = res.scalar_one_or_none()

            if existing:
                existing.value = content
                existing.agent_id = agent_id
                existing.client_id = client_id
                existing.client_name = client_name
                existing.updated_at = updated_at
                wid = existing.id
            else:
                session.add(WorkingMemory(
                    id=wid, session_id='global', key=key, value=content,
                    agent_id=agent_id, client_id=client_id, client_name=client_name,
                    tenant=tenant, project_id=project_id, updated_at=updated_at
                ))
            await session.commit()

        return {
            "id": wid,
            "content": content,
            "tenant": tenant,
            "project_id": project_id,
            "created_at": updated_at,
        }

    async def add_client_issue(self, client_id: str, message: str, tenant: str, project_id: str,
                         error_code: str, client_name: Optional[str] = None,
                         agent_id: Optional[str] = None, parent_client_id: Optional[str] = None,
                         child_client_id: Optional[str] = None, layer: Optional[str] = None,
                         payload: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        issue_id = str(uuid.uuid4())
        created_at = datetime.datetime.now().isoformat()

        async with AsyncSessionLocal() as session:
            session.add(ClientIssue(
                id=issue_id, client_id=client_id, client_name=client_name, agent_id=agent_id,
                parent_client_id=parent_client_id, child_client_id=child_client_id,
                layer=layer, error_code=error_code, message=message,
                payload=json.dumps(payload) if payload else None,
                metadata_=json.dumps(metadata) if metadata else None,
                status='open', tenant=tenant, project_id=project_id, created_at=created_at
            ))
            await session.commit()

        await self.add_audit_event(
            event_type="client_issue:create",
            payload={
                "id": issue_id,
                "client_id": client_id,
                "client_name": client_name,
                "agent_id": agent_id,
                "error_code": error_code,
                "tenant": tenant,
                "project_id": project_id,
                "ts": created_at,
            },
        )
        return {
            "id": issue_id,
            "client_id": client_id,
            "client_name": client_name,
            "agent_id": agent_id,
            "parent_client_id": parent_client_id,
            "child_client_id": child_client_id,
            "layer": layer,
            "error_code": error_code,
            "message": message,
            "payload": payload,
            "metadata": metadata,
            "status": "open",
            "tenant": tenant,
            "project_id": project_id,
            "created_at": created_at,
            "resolved_at": None,
            "resolved_by": None,
            "resolution": None,
        }

    async def list_client_issues(self, tenant: str, project_id: str, status: str = "open", limit: int = 200) -> List[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            stmt = select(ClientIssue).where(
                ClientIssue.tenant == tenant, ClientIssue.project_id == project_id, ClientIssue.status == status
            ).order_by(ClientIssue.created_at.desc()).limit(limit)
            res = await session.execute(stmt)
            rows = []
            for item in res.scalars().all():
                d = {}
                for c in item.__table__.columns:
                    attr = "metadata_" if c.name == "metadata" else c.name
                    d[c.name] = getattr(item, attr)

                if d.get("payload"):
                    try:
                        d["payload"] = json.loads(d["payload"])
                    except:
                        d["payload"] = {}
                if d.get("metadata"):
                    try:
                        d["metadata"] = json.loads(d["metadata"])
                    except:
                        d["metadata"] = {}
                rows.append(d)
            return rows

    async def resolve_client_issue(self, issue_id: str, resolution: str, reviewer: str) -> Optional[Dict[str, Any]]:
        resolved_at = datetime.datetime.now().isoformat()

        async with AsyncSessionLocal() as session:
            stmt = select(ClientIssue).where(ClientIssue.id == issue_id)
            res = await session.execute(stmt)
            item = res.scalar_one_or_none()
            if not item: return None

            item.status = 'resolved'
            item.resolved_at = resolved_at
            item.resolved_by = reviewer
            item.resolution = resolution
            await session.commit()

            d = {}
            for c in item.__table__.columns:
                attr = "metadata_" if c.name == "metadata" else c.name
                d[c.name] = getattr(item, attr)

            if d.get("payload"):
                try:
                    d["payload"] = json.loads(d["payload"])
                except:
                    d["payload"] = {}
            if d.get("metadata"):
                try:
                    d["metadata"] = json.loads(d["metadata"])
                except:
                    d["metadata"] = {}

        await self.add_audit_event(
            event_type=f"client_issue:resolved",
            payload={"id": issue_id, "reviewer": reviewer, "resolution": resolution, "ts": resolved_at},
        )
        return d

    async def get_working(self, tenant: str, project_id: str) -> Dict[str, Any]:
        async with AsyncSessionLocal() as session:
            stmt = select(WorkingMemory).where(
                WorkingMemory.session_id == 'global',
                WorkingMemory.key == 'current_context',
                WorkingMemory.tenant == tenant,
                WorkingMemory.project_id == project_id
            )
            res = await session.execute(stmt)
            item = res.scalar_one_or_none()
            if item:
                return {"content": item.value, "updated_at": item.updated_at}
            return None

    async def delete_expired_memories(self, now_iso: str) -> Dict[str, int]:
        async with AsyncSessionLocal() as session:
            # logs_session
            stmt1 = delete(SessionLog).where(
                and_(SessionLog.expires_at.is_not(None), SessionLog.expires_at < now_iso)
            )
            res1 = await session.execute(stmt1)
            session_deleted = res1.rowcount

            # working_memory
            stmt2 = delete(WorkingMemory).where(
                and_(WorkingMemory.expires_at.is_not(None), WorkingMemory.expires_at < now_iso)
            )
            res2 = await session.execute(stmt2)
            working_deleted = res2.rowcount

            await session.commit()

        return {"session": session_deleted, "working": working_deleted}
