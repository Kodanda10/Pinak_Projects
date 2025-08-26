from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
import hashlib
import stat
import random


PINAK_DIR_NAME = ".pinak"
PINAK_CONFIG_NAME = "pinak.json"
PINAK_TOKEN_FILE = "token"
KEYRING_SERVICE = "pinak"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _walk_up_for(path: Path, name: str) -> Optional[Path]:
    cur = path.resolve()
    if cur.is_file():
        cur = cur.parent
    while True:
        candidate = cur / PINAK_DIR_NAME / name
        if candidate.exists():
            return candidate
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
        cfg_path = _walk_up_for(start, PINAK_CONFIG_NAME)
        if not cfg_path:
            return None
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            # Compute fingerprint from stable fields
            fp_src = json.dumps({
                "project_id": data.get("project_id"),
                "project_name": data.get("project_name"),
                "tenant": data.get("tenant"),
                "memory_url": data.get("memory_url"),
                "version": int(data.get("version", 1)),
            }, separators=(",", ":"), sort_keys=True)
            fingerprint = hashlib.sha256(fp_src.encode("utf-8")).hexdigest()
            ctx = ProjectContext(
                project_id=data.get("project_id"),
                project_name=data.get("project_name"),
                tenant=data.get("tenant"),
                memory_url=data.get("memory_url"),
                created_at=data.get("created_at"),
                version=int(data.get("version", 1)),
                root_dir=cfg_path.parent.parent,  # .pinak/ -> project root
                identity_fingerprint=data.get("identity_fingerprint") or fingerprint,
            )
            return ctx
        except Exception:
            return None

    @staticmethod
    def init_new(project_name: str, memory_url: str, tenant: Optional[str] = None, root: Optional[Path] = None) -> "ProjectContext":
        project_id = f"Pnk-{_uuid7()}"
        root = root or Path.cwd()
        pinak_dir = root / PINAK_DIR_NAME
        _safe_mkdir(pinak_dir)
        cfg_base = {
            "project_id": project_id,
            "project_name": project_name,
            "tenant": tenant,
            "memory_url": memory_url,
            "version": 1,
        }
        fp_src = json.dumps(cfg_base, separators=(",", ":"), sort_keys=True)
        fingerprint = hashlib.sha256(fp_src.encode("utf-8")).hexdigest()
        cfg = dict(cfg_base)
        cfg.update({
            "created_at": _now_iso(),
            "identity_fingerprint": fingerprint,
        })
        cfg_path = (pinak_dir / PINAK_CONFIG_NAME)
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return ProjectContext(
            project_id=project_id,
            project_name=project_name,
            tenant=tenant,
            memory_url=memory_url,
            created_at=cfg["created_at"],
            root_dir=root,
            identity_fingerprint=fingerprint,
        )

    # ---- Token storage ----
    def _token_keyring_key(self) -> Tuple[str, str]:
        # service, username
        return KEYRING_SERVICE, f"project:{self.project_id}"

    def set_token(self, token: str, fallback_to_file: bool = True) -> None:
        # Try keyring first for secure storage
        try:
            import keyring  # type: ignore

            service, username = self._token_keyring_key()
            keyring.set_password(service, username, token)
            return
        except Exception:
            if not fallback_to_file:
                raise
        # Fallback to .pinak/token
        if not self.root_dir:
            root = ProjectContext.find()
            self.root_dir = root.root_dir if root else Path.cwd()
        pinak_dir = (self.root_dir or Path.cwd()) / PINAK_DIR_NAME
        _safe_mkdir(pinak_dir)
        token_path = (pinak_dir / PINAK_TOKEN_FILE)
        token_path.write_text(token, encoding="utf-8")
        try:
            token_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600
        except Exception:
            pass

    def get_token(self) -> Optional[str]:
        # Env override always wins
        env_token = os.getenv("PINAK_TOKEN")
        if env_token:
            return env_token
        # Keyring
        try:
            import keyring  # type: ignore

            service, username = self._token_keyring_key()
            token = keyring.get_password(service, username)
            if token:
                return token
        except Exception:
            pass
        # Fallback to file
        cfg_path = _walk_up_for(self.root_dir or Path.cwd(), PINAK_TOKEN_FILE)
        if cfg_path and cfg_path.exists():
            try:
                return cfg_path.read_text(encoding="utf-8").strip()
            except Exception:
                return None
        return None

# --------- helpers ---------

def _uuid7() -> str:
    """Generate a UUIDv7-like identifier. If Python has uuid.uuid7 use it; else approximate.

    Note: For local dev identity, monotonicity is sufficient; real UUIDv7 format not strictly required.
    """
    try:
        return str(uuid.uuid7())  # type: ignore[attr-defined]
    except Exception:
        # Approximate: 48 bits millis + 80 bits randomness
        ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        rand80 = random.getrandbits(80)
        val = (ts_ms << 80) | rand80
        # Pack into 128 bits hex (trim/pad)
        hex128 = f"{val:032x}"[-32:]
        return str(uuid.UUID(hex=hex128))
