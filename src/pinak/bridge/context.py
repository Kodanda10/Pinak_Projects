from __future__ import annotations

import json
import os
import uuid
import hashlib
import random
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

PINAK_DIR = ".pinak"
PINAK_CONFIG = "pinak.json"
PINAK_TOKEN_FILE = "token"
KEYRING_SERVICE = "pinak"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid7() -> str:
    try:
        return str(uuid.uuid7())  # type: ignore[attr-defined]
    except Exception:
        ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        rand80 = random.getrandbits(80)
        val = (ts_ms << 80) | rand80
        return str(uuid.UUID(hex=f"{val:032x}"[-32:]))


def _walk_up_for(start: Path, name: str) -> Optional[Path]:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    while True:
        cand = cur / PINAK_DIR / name
        if cand.exists():
            return cand
        if cur.parent == cur:
            return None
        cur = cur.parent


@dataclass
class ProjectContext:
    project_id: str
    project_name: Optional[str] = None
    tenant: Optional[str] = None
    memory_url: Optional[str] = None
    created_at: Optional[str] = None
    version: int = 1
    root_dir: Optional[Path] = None
    identity_fingerprint: Optional[str] = None

    @staticmethod
    def find(start: Optional[Path] = None) -> Optional["ProjectContext"]:
        start = start or Path.cwd()
        cfg_path = _walk_up_for(start, PINAK_CONFIG)
        if not cfg_path:
            return None
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        base = {
            "project_id": data.get("project_id"),
            "project_name": data.get("project_name"),
            "tenant": data.get("tenant"),
            "memory_url": data.get("memory_url"),
            "version": int(data.get("version", 1)),
        }
        fp = hashlib.sha256(json.dumps(base, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
        return ProjectContext(
            project_id=base["project_id"],
            project_name=base["project_name"],
            tenant=base["tenant"],
            memory_url=base["memory_url"],
            created_at=data.get("created_at"),
            version=base["version"],
            root_dir=cfg_path.parent.parent,
            identity_fingerprint=data.get("identity_fingerprint") or fp,
        )

    @staticmethod
    def init_new(project_name: str, memory_url: str, tenant: Optional[str] = None, root: Optional[Path] = None) -> "ProjectContext":
        pid = f"Pnk-{_uuid7()}"
        root = root or Path.cwd()
        pinak_dir = root / PINAK_DIR
        pinak_dir.mkdir(parents=True, exist_ok=True)
        base = {
            "project_id": pid,
            "project_name": project_name,
            "tenant": tenant,
            "memory_url": memory_url,
            "version": 1,
        }
        fp = hashlib.sha256(json.dumps(base, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
        cfg = dict(base)
        cfg.update({"created_at": _now_iso(), "identity_fingerprint": fp})
        (pinak_dir / PINAK_CONFIG).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return ProjectContext(project_id=pid, project_name=project_name, tenant=tenant, memory_url=memory_url, created_at=cfg["created_at"], root_dir=root, identity_fingerprint=fp)

    def _kr(self) -> Tuple[str, str]:
        return KEYRING_SERVICE, f"project:{self.project_id}"

    def set_token(self, token: str, fallback_to_file: bool = True) -> None:
        try:
            import keyring
            s,u = self._kr()
            keyring.set_password(s,u,token)
            return
        except Exception:
            if not fallback_to_file:
                raise
        token_path = (self.root_dir or Path.cwd()) / PINAK_DIR / PINAK_TOKEN_FILE
        token_path.write_text(token, encoding="utf-8")
        try:
            token_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass

    def get_token(self) -> Optional[str]:
        env = os.getenv("PINAK_TOKEN")
        if env:
            return env
        try:
            import keyring
            s,u = self._kr()
            tok = keyring.get_password(s,u)
            if tok:
                return tok
        except Exception:
            pass
        t = _walk_up_for((self.root_dir or Path.cwd()), PINAK_TOKEN_FILE)
        return t.read_text(encoding="utf-8").strip() if t and t.exists() else None

    # Rotation helper: mint a short-lived JWT and store it
    def rotate_token(self, minutes: int = 60, secret: Optional[str] = None, sub: str = "analyst", role: Optional[str] = None) -> str:
        try:
            from jose import jwt
        except Exception as e:
            raise RuntimeError("python-jose required for token rotation") from e
        import datetime
        secret = secret or os.getenv("SECRET_KEY", "change-me-in-prod")
        exp_ts = int((datetime.datetime.utcnow() + datetime.timedelta(minutes=int(minutes))).timestamp())
        claims = {"sub": sub, "pid": self.project_id, "exp": exp_ts}
        if role:
            claims["role"] = role
        token = jwt.encode(claims, secret, algorithm="HS256")
        self.set_token(token)
        return token
