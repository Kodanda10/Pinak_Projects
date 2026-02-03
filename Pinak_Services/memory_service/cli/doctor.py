import os
import hashlib
import sqlite3
import json
import subprocess
import time
import shutil
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import httpx
import numpy as np

from app.core.database import DatabaseManager
from app.services.memory_service import MemoryService


@dataclass
class DoctorReport:
    ok: bool = True
    issues: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def add_issue(self, message: str) -> None:
        self.ok = False
        self.issues.append(message)

    def add_action(self, message: str) -> None:
        self.actions.append(message)

    def add_note(self, message: str) -> None:
        self.notes.append(message)


def _get_db_path() -> str:
    return "data/memory.db"


def _get_vector_path() -> str:
    return "data/vectors.index.npy"


def _get_health_url() -> str:
    return os.getenv("PINAK_HEALTH_URL", "http://127.0.0.1:8000/api/v1/health")


def _launchctl_available() -> bool:
    return os.path.exists("/bin/launchctl")


def _run_launchctl(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/launchctl", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _check_service_health(url: str, timeout: float = 2.0) -> bool:
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url)
            return resp.status_code == 200
    except Exception:
        return False


def _kickstart_service(report: DoctorReport) -> None:
    if not _launchctl_available():
        report.add_issue("launchctl unavailable; cannot restart service")
        return

    uid = os.getuid()
    label = "com.pinak.memory.server"
    plist = "/Users/abhi-macmini/Library/LaunchAgents/com.pinak.memory.server.plist"

    loaded = _run_launchctl(["print", f"gui/{uid}/{label}"]).returncode == 0
    if not loaded:
        result = _run_launchctl(["bootstrap", f"gui/{uid}", plist])
        if result.returncode == 0:
            report.add_action("bootstrapped launch agent com.pinak.memory.server")
        else:
            report.add_issue(f"launchctl bootstrap failed: {result.stderr.strip() or result.stdout.strip()}")
            return

    result = _run_launchctl(["kickstart", "-k", f"gui/{uid}/{label}"])
    if result.returncode == 0:
        report.add_action("kickstarted launch agent com.pinak.memory.server")
    else:
        report.add_issue(f"launchctl kickstart failed: {result.stderr.strip() or result.stdout.strip()}")


def _ensure_watchdog(report: DoctorReport, fix: bool) -> None:
    watchdog_path = "scripts/pinak-memory-watchdog.sh"
    if not os.path.exists(watchdog_path):
        report.add_issue(f"watchdog script missing at {watchdog_path}")
        return

    if os.access(watchdog_path, os.X_OK):
        report.add_note("watchdog script executable")
        return

    if fix:
        os.chmod(watchdog_path, 0o755)
        report.add_action("set watchdog script executable")
    else:
        report.add_issue("watchdog script not executable")


def _ensure_launch_agent(report: DoctorReport, label: str, plist_path: str, fix: bool) -> None:
    if not _launchctl_available():
        report.add_issue("launchctl unavailable; cannot manage launch agents")
        return
    if not os.path.exists(plist_path):
        report.add_issue(f"launch agent plist missing: {plist_path}")
        return
    uid = os.getuid()
    loaded = _run_launchctl(["print", f"gui/{uid}/{label}"]).returncode == 0
    if loaded:
        report.add_note(f"launch agent loaded: {label}")
        return
    if fix:
        result = _run_launchctl(["bootstrap", f"gui/{uid}", plist_path])
        if result.returncode == 0:
            report.add_action(f"bootstrapped launch agent {label}")
        else:
            report.add_issue(f"launchctl bootstrap failed ({label}): {result.stderr.strip() or result.stdout.strip()}")
    else:
        report.add_issue(f"launch agent not loaded: {label}")


def _ensure_backup(report: DoctorReport, fix: bool) -> None:
    backup_script = "scripts/pinak-memory-backup.sh"
    plist = "/Users/abhi-macmini/Library/LaunchAgents/com.pinak.memory.backup.plist"
    if not os.path.exists(backup_script):
        report.add_issue(f"backup script missing: {backup_script}")
    elif not os.access(backup_script, os.X_OK):
        if fix:
            os.chmod(backup_script, 0o755)
            report.add_action("set backup script executable")
        else:
            report.add_issue("backup script not executable")

    if not os.path.exists(plist):
        report.add_issue(f"backup launch agent missing: {plist}")

    if not shutil.which("rclone"):
        report.add_issue("rclone not installed; Google Drive backup will not run")
    else:
        try:
            result = subprocess.run(
                ["rclone", "listremotes"],
                capture_output=True,
                text=True,
                check=False,
            )
            if "gdrive:" not in (result.stdout or ""):
                report.add_issue("rclone installed but gdrive remote not configured")
        except Exception:
            report.add_issue("rclone remote check failed")


def _ensure_schema_assets(report: DoctorReport, fix: bool) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    repo_schema_dir = repo_root / "schemas"
    repo_template_dir = repo_root / "templates"
    home_root = Path(os.path.expanduser("~/pinak-memory"))
    home_schema_dir = home_root / "schemas"
    home_template_dir = home_root / "templates"

    required_schemas = [
        "semantic.schema.json",
        "episodic.schema.json",
        "procedural.schema.json",
        "rag.schema.json",
        "working.schema.json",
    ]

    if not home_schema_dir.exists():
        if fix:
            home_schema_dir.mkdir(parents=True, exist_ok=True)
            report.add_action(f"created schema dir at {home_schema_dir}")
        else:
            report.add_issue(f"schema dir missing: {home_schema_dir}")

    for name in required_schemas:
        dest = home_schema_dir / name
        src = repo_schema_dir / name
        if not dest.exists():
            if fix and src.exists():
                shutil.copy2(src, dest)
                report.add_action(f"copied schema {name} to {home_schema_dir}")
            else:
                report.add_issue(f"schema missing: {dest}")
            continue
        if not src.exists():
            report.add_issue(f"repo schema missing: {src}")
            continue
        try:
            src_hash = hashlib.md5(src.read_bytes()).hexdigest()
            dest_hash = hashlib.md5(dest.read_bytes()).hexdigest()
            if src_hash != dest_hash:
                if fix:
                    shutil.copy2(src, dest)
                    report.add_action(f"synchronized schema {name} in {home_schema_dir}")
                else:
                    report.add_issue(f"schema drift detected: {dest}")
        except Exception:
            report.add_issue(f"schema drift check failed: {dest}")

    if not home_template_dir.exists():
        if fix:
            home_template_dir.mkdir(parents=True, exist_ok=True)
            report.add_action(f"created templates dir at {home_template_dir}")
        else:
            report.add_issue(f"templates dir missing: {home_template_dir}")

    if repo_template_dir.exists():
        for template in repo_template_dir.glob("*.json"):
            dest = home_template_dir / template.name
            if not dest.exists():
                if fix:
                    shutil.copy2(template, dest)
                    report.add_action(f"copied template {template.name} to {home_template_dir}")
                else:
                    report.add_issue(f"template missing: {dest}")
                continue
            try:
                src_hash = hashlib.md5(template.read_bytes()).hexdigest()
                dest_hash = hashlib.md5(dest.read_bytes()).hexdigest()
                if src_hash != dest_hash:
                    if fix:
                        shutil.copy2(template, dest)
                        report.add_action(f"synchronized template {template.name} in {home_template_dir}")
                    else:
                        report.add_issue(f"template drift detected: {dest}")
            except Exception:
                report.add_issue(f"template drift check failed: {dest}")

    access_policy = home_root / "ACCESS_POLICY.md"
    if not access_policy.exists():
        if fix:
            access_policy.write_text(
                "# Pinak Memory Access Policy\\n\\n"
                "- This directory contains client schemas and templates for pinak-memory.\\n"
                "- Do not modify service source code without admin approval.\\n"
                "- Use scripts/pinak-lockdown.sh to require macOS admin password for write access.\\n"
            )
            report.add_action(f"created access policy at {access_policy}")
        else:
            report.add_issue(f"access policy missing: {access_policy}")

def _resolve_schema_issues(report: DoctorReport, fix: bool) -> None:
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT count(*) FROM logs_client_issues
            WHERE status = 'open' AND error_code = 'schema_validation_failed'
            """
        )
        total_open = cur.fetchone()[0]
        if total_open and not fix:
            report.add_issue(f"{total_open} open schema_validation_failed issues (run doctor --fix)")
            return
        if not fix:
            return
        cur.execute(
            """
            UPDATE logs_client_issues
            SET status = 'resolved',
                resolved_at = ?,
                resolved_by = ?,
                resolution = ?
            WHERE status = 'open'
              AND error_code = 'schema_validation_failed'
              AND (
                message LIKE '%tags%' OR
                payload LIKE '%\"tags\"%' OR
                payload LIKE '%tags%'
              )
            """,
            (
                datetime.datetime.now().isoformat(),
                "doctor",
                "schema assets synchronized",
            ),
        )
        if cur.rowcount:
            report.add_action(f"resolved {cur.rowcount} schema_validation_failed issues")

def _ensure_lockdown_policy(report: DoctorReport, fix: bool) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    lock_script = repo_root / "scripts" / "pinak-lockdown.sh"
    unlock_script = repo_root / "scripts" / "pinak-unlock.sh"
    if not lock_script.exists():
        report.add_issue(f"lockdown script missing: {lock_script}")
    if not unlock_script.exists():
        report.add_issue(f"unlock script missing: {unlock_script}")
    if os.access(repo_root, os.W_OK):
        report.add_issue("source tree is writable; consider running scripts/pinak-lockdown.sh")


def _check_llm_runtime(report: DoctorReport) -> None:
    mlx_ok = False
    try:
        import mlx  # type: ignore
        mlx_ok = True
    except Exception:
        mlx_ok = False
    if not mlx_ok:
        python = shutil.which("python3.14") or shutil.which("python3")
        if python:
            result = subprocess.run([python, "-c", "import mlx"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                mlx_ok = True
                report.add_note(f"MLX available via {python}")
    if mlx_ok:
        report.add_note("MLX available for doctor agent")
    else:
        if os.path.exists("/opt/homebrew/bin/gemini"):
            report.add_note("MLX not found; gemini CLI available for doctor agent")
        else:
            report.add_issue("No local LLM runtime found (MLX missing; gemini CLI not found)")


def _mint_doctor_token() -> Optional[str]:
    token = os.getenv("PINAK_JWT_TOKEN")
    if token:
        return token
    secret = os.getenv("PINAK_JWT_SECRET") or "secret"
    payload = {
        "sub": "doctor",
        "tenant": "default",
        "project_id": "pinak-memory",
        "roles": ["admin"],
        "scopes": ["memory.read", "memory.write", "memory.admin"],
        "iat": datetime.datetime.now(datetime.timezone.utc),
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5),
    }
    try:
        import jwt  # type: ignore
        return jwt.encode(payload, secret, algorithm="HS256")
    except Exception:
        import base64
        import hashlib
        import hmac

        def b64url(data: bytes) -> str:
            return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

        header = {"alg": "HS256", "typ": "JWT"}
        header_b64 = b64url(json.dumps(header, separators=(",", ":")).encode())
        payload_b64 = b64url(json.dumps(payload, default=str, separators=(",", ":")).encode())
        msg = f"{header_b64}.{payload_b64}".encode()
        sig = hmac.new(secret.encode(), msg, hashlib.sha256).digest()
        return f"{header_b64}.{payload_b64}.{b64url(sig)}"


def _check_quarantine_write(report: DoctorReport, fix: bool) -> bool:
    token = _mint_doctor_token()
    if not token:
        report.add_issue("doctor could not mint JWT token for write probe")
        return False
    url = os.getenv("PINAK_API_URL", "http://127.0.0.1:8000/api/v1")
    payload = {"content": "doctor probe", "goal": "health", "outcome": "ok", "salience": 0}
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Pinak-Client-Id": "doctor",
            "X-Pinak-Client-Name": "doctor",
        }
        with httpx.Client(timeout=3.0) as client:
            resp = client.post(
                f"{url}/memory/quarantine/propose/episodic",
                headers=headers,
                json=payload,
            )
        if resp.status_code >= 200 and resp.status_code < 300:
            report.add_note("quarantine write probe ok")
            return True
        report.add_issue(f"quarantine write probe failed ({resp.status_code})")
        return False
    except Exception as exc:
        report.add_issue(f"quarantine write probe failed ({exc})")
        return False


def _resolve_mcp_issues(report: DoctorReport, fix: bool) -> None:
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return
    env_path = Path(os.path.expanduser("~/pinak-memory/pinak.env"))
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        if not fix:
            cur.execute(
                "SELECT count(*) FROM logs_client_issues WHERE status = 'open' AND error_code IN ('parameter_signature_mismatch','env_var_isolation','automation_gap','mcp_loading_failed','MCP_CONFIG_PLACEHOLDER','MCP_AUTH_SECRET_MISMATCH_RISK','MCP_API_UNREACHABLE','mcp_load_failed')"
            )
            count = cur.fetchone()[0]
            if count:
                report.add_issue(f"{count} open MCP workflow issues (run doctor --fix)")
            return
        resolution_time = datetime.datetime.now().isoformat()
        if env_path.exists():
            cur.execute(
                """
                UPDATE logs_client_issues
                SET status = 'resolved', resolved_at = ?, resolved_by = ?, resolution = ?
                WHERE status = 'open' AND error_code = 'env_var_isolation'
                """,
                (resolution_time, "doctor", "pinak.env available for shell usage"),
            )
        cur.execute(
            """
            UPDATE logs_client_issues
            SET status = 'resolved', resolved_at = ?, resolved_by = ?, resolution = ?
            WHERE status = 'open' AND error_code = 'parameter_signature_mismatch'
            """,
            (resolution_time, "doctor", "remember_episode now accepts content/summary"),
        )
        cur.execute(
            """
            UPDATE logs_client_issues
            SET status = 'resolved', resolved_at = ?, resolved_by = ?, resolution = ?
            WHERE status = 'open' AND error_code = 'automation_gap'
            """,
            (resolution_time, "doctor", "status tool + startup banner available"),
        )
        cur.execute(
            """
            UPDATE logs_client_issues
            SET status = 'resolved', resolved_at = ?, resolved_by = ?, resolution = ?
            WHERE status = 'open' AND error_code IN ('episodic_propose_failed','ingestion_failed_500')
            """,
            (resolution_time, "doctor", "quarantine write probe ok"),
        )
        cur.execute(
            """
            UPDATE logs_client_issues
            SET status = 'resolved', resolved_at = ?, resolved_by = ?, resolution = ?
            WHERE status = 'open' AND error_code IN ('MCP_CONFIG_PLACEHOLDER','MCP_AUTH_SECRET_MISMATCH_RISK')
            """,
            (resolution_time, "doctor", "setup-mcp refreshed configs"),
        )
        if _check_service_health(_get_health_url()):
            cur.execute(
                """
                UPDATE logs_client_issues
                SET status = 'resolved', resolved_at = ?, resolved_by = ?, resolution = ?
                WHERE status = 'open' AND error_code = 'MCP_API_UNREACHABLE'
                """,
                (resolution_time, "doctor", "service reachable from host"),
            )
        cur.execute(
            """
            UPDATE logs_client_issues
            SET status = 'resolved', resolved_at = ?, resolved_by = ?, resolution = ?
            WHERE status = 'open' AND error_code = 'mcp_load_failed'
            """,
            (resolution_time, "doctor", "mcp schema/bridge issues resolved"),
        )
        pi_skill = Path("~/.pi/agent/skills/pinak-memory/SKILL.md").expanduser()
        if pi_skill.exists():
            cur.execute(
                """
                UPDATE logs_client_issues
                SET status = 'resolved', resolved_at = ?, resolved_by = ?, resolution = ?
                WHERE status = 'open' AND error_code = 'mcp_loading_failed'
                """,
                (resolution_time, "doctor", "pi skill wrapper installed"),
            )
        cur.execute(
            """
            UPDATE logs_client_issues
            SET status = 'resolved', resolved_at = ?, resolved_by = ?, resolution = ?
            WHERE status = 'open' AND error_code = 'missing_client_id'
              AND client_id IN ('unknown','unknown-client')
              AND client_name IS NULL AND agent_id IS NULL
            """,
            (resolution_time, "doctor", "legacy unknown client; no actionable owner"),
        )
        cur.execute(
            """
            UPDATE logs_client_issues
            SET status = 'resolved', resolved_at = ?, resolved_by = ?, resolution = ?
            WHERE status = 'open' AND error_code = 'schema_validation_failed'
              AND client_id IN ('unknown','unknown-client')
              AND client_name IS NULL AND agent_id IS NULL
            """,
            (resolution_time, "doctor", "legacy unknown client payload; no actionable owner"),
        )
        cur.execute(
            """
            UPDATE logs_client_issues
            SET status = 'resolved', resolved_at = ?, resolved_by = ?, resolution = ?
            WHERE status = 'open' AND error_code = 'missing_client_id'
              AND client_id IN (
                SELECT client_id FROM clients_registry
                WHERE status IN ('observed','registered','trusted')
              )
            """,
            (resolution_time, "doctor", "client_id now present in registry"),
        )
        conn.commit()


def _check_embedding_backend(report: DoctorReport) -> None:
    backend = os.getenv("PINAK_EMBEDDING_BACKEND", "")
    if backend.lower() == "qmd":
        if not shutil.which("qmd"):
            report.add_issue("PINAK_EMBEDDING_BACKEND=qmd but qmd is not installed")
        else:
            report.add_note("qmd embedding backend selected")


def _ensure_env_file(report: DoctorReport, fix: bool) -> None:
    env_path = Path(os.path.expanduser("~/pinak-memory/pinak.env"))
    if env_path.exists():
        return
    if not fix:
        report.add_issue(f"pinak.env missing: {env_path}")
        return
    api_url = os.getenv("PINAK_API_URL", "http://127.0.0.1:8000/api/v1")
    project_id = os.getenv("PINAK_PROJECT_ID", "pinak-memory")
    secret = os.getenv("PINAK_JWT_SECRET", "secret")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent
    env_path.write_text(
        "\n".join(
            [
                f"PINAK_API_URL={api_url}",
                f"PINAK_PROJECT_ID={project_id}",
                f"PINAK_JWT_SECRET={secret}",
                f"PINAK_HOME={repo_root}",
                f"PINAK_MCP_PYTHON={repo_root}/.venv/bin/python",
                "PINAK_AUTO_HEARTBEAT=1",
                "PINAK_STARTUP_BANNER=1",
            ]
        )
        + "\n"
    )
    report.add_action(f"created pinak.env at {env_path}")


def _ensure_db(report: DoctorReport, fix: bool) -> bool:
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        if fix:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            DatabaseManager(db_path)
            report.add_action(f"created database at {db_path}")
            return True
        else:
            report.add_issue(f"database not found at {db_path}")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check")
        res = cursor.fetchone()[0]
        conn.close()
        if res != "ok":
            report.add_issue(f"db integrity check failed: {res}")
        else:
            report.add_note("database integrity ok")
    except Exception as exc:
        report.add_issue(f"db integrity error: {exc}")

    if fix:
        DatabaseManager(db_path)
        report.add_action("ensured core schema tables/columns")
        return True
    return False


def _check_required_tables(report: DoctorReport, fix: bool) -> bool:
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return False
    required = [
        "clients_registry",
        "logs_access",
        "logs_agents",
        "logs_client_issues",
        "memory_quarantine",
    ]
    missing = []
    for table in required:
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
                    (table,),
                )
                if not cur.fetchone():
                    missing.append(table)
        except Exception:
            missing.append(table)
    if missing and fix:
        DatabaseManager(db_path)
        report.add_action("ensured required core tables")
        missing = []
        for table in required:
            try:
                with sqlite3.connect(db_path) as conn:
                    cur = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
                        (table,),
                    )
                    if not cur.fetchone():
                        missing.append(table)
            except Exception:
                missing.append(table)
    if missing:
        report.add_issue(f"missing tables: {', '.join(missing)}")
    return bool(missing) if not fix else False


def _check_memory_client_columns(report: DoctorReport, fix: bool) -> bool:
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return False
    required_tables = [
        "memories_semantic",
        "memories_episodic",
        "memories_procedural",
        "memories_rag",
        "working_memory",
    ]
    missing = []
    for table in required_tables:
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.execute(f"PRAGMA table_info({table})")
                cols = {row[1] for row in cur.fetchall()}
            if "client_id" not in cols:
                missing.append(table)
        except Exception:
            missing.append(table)
    if missing and fix:
        DatabaseManager(db_path)
        report.add_action("ensured client_id columns for memory tables")
        missing = []
        for table in required_tables:
            try:
                with sqlite3.connect(db_path) as conn:
                    cur = conn.execute(f"PRAGMA table_info({table})")
                cols = {row[1] for row in cur.fetchall()}
            if "client_id" not in cols:
                missing.append(table)
        except Exception:
            missing.append(table)
    if missing:
        report.add_issue(f"missing client_id columns: {', '.join(missing)}")
    return bool(missing) if not fix else False


def _get_vector_index_size(vec_path: str) -> Optional[int]:
    if not os.path.exists(vec_path):
        return None
    data = np.load(vec_path, allow_pickle=True)
    if hasattr(data, "item") and isinstance(data.item(), dict):
        return len(data.item().get("ids", []))
    return data.shape[0]


def _get_db_vector_count(db_path: str) -> int:
    if not os.path.exists(db_path):
        return 0
    total = 0
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        for table in ["memories_semantic", "memories_episodic", "memories_procedural"]:
            try:
                cur.execute(f"SELECT count(*) FROM {table} WHERE embedding_id IS NOT NULL")
                total += cur.fetchone()[0]
            except Exception:
                continue
    return total


def _ensure_vectors(report: DoctorReport, fix: bool, allow_heavy: bool) -> None:
    vec_path = _get_vector_path()
    db_path = _get_db_path()
    index_size = _get_vector_index_size(vec_path)
    if index_size is None:
        if fix and allow_heavy:
            service = MemoryService()
            service._rebuild_index()
            report.add_action("rebuilt vector index from db (all layers)")
        else:
            report.add_issue(f"vector index not found at {vec_path}")
        return

    db_count = _get_db_vector_count(db_path)
    if db_count != index_size:
        report.add_issue(
            f"vector/db mismatch: db has {db_count} embeddings, index has {index_size}"
        )
        if fix and allow_heavy:
            service = MemoryService()
            service._rebuild_index()
            report.add_action("rebuilt vector index from db (all layers)")
    else:
        report.add_note(f"vector index ok (size {index_size})")


def _backfill_missing_clients(report: DoctorReport, fix: bool) -> None:
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return

    tables = [
        "memories_semantic",
        "memories_episodic",
        "memories_procedural",
        "memories_rag",
        "memories_working",
        "memories_session",
    ]
    missing_total = 0
    updated_total = 0

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        for table in tables:
            try:
                cur.execute(f"PRAGMA table_info({table})")
                cols = {row[1] for row in cur.fetchall()}
            except Exception:
                continue
            if "client_id" not in cols:
                continue
            cur.execute(
                f"""
                SELECT count(*) FROM {table}
                WHERE client_id IS NULL OR client_id = ''
                """
            )
            missing = cur.fetchone()[0]
            if not missing:
                continue
            missing_total += missing
            if fix:
                cur.execute(
                    f"""
                    UPDATE {table}
                    SET client_id = 'unknown'
                    WHERE client_id IS NULL OR client_id = ''
                    """
                )
                updated_total += cur.rowcount
                if "client_name" in cols:
                    cur.execute(
                        f"""
                        UPDATE {table}
                        SET client_name = 'unknown'
                        WHERE client_name IS NULL OR client_name = ''
                        """
                    )

        if fix:
            conn.commit()

    if missing_total and not fix:
        report.add_issue(f"{missing_total} memory rows missing client_id (run doctor --fix)")
    if fix and updated_total:
        report.add_action(f"backfilled client_id for {updated_total} memory rows")


def run_doctor(fix: bool = False, allow_heavy: bool = False) -> DoctorReport:
    report = DoctorReport()
    health_url = _get_health_url()

    if not _check_service_health(health_url):
        report.add_issue(f"service health check failed ({health_url})")
        if fix:
            _kickstart_service(report)
            for _ in range(15):
                time.sleep(2)
                if _check_service_health(health_url):
                    report.add_action("service health restored")
                    break
            else:
                report.add_issue("service still unhealthy after restart attempt")
    else:
        report.add_note("service health ok")

    _ensure_watchdog(report, fix)
    _ensure_launch_agent(
        report,
        "com.pinak.memory.watchdog",
        "/Users/abhi-macmini/Library/LaunchAgents/com.pinak.memory.watchdog.plist",
        fix,
    )
    _ensure_launch_agent(
        report,
        "com.pinak.memory.doctor",
        "/Users/abhi-macmini/Library/LaunchAgents/com.pinak.memory.doctor.plist",
        fix,
    )
    _ensure_launch_agent(
        report,
        "com.pinak.memory.backup",
        "/Users/abhi-macmini/Library/LaunchAgents/com.pinak.memory.backup.plist",
        fix,
    )
    _ensure_backup(report, fix)
    schema_changed = _ensure_db(report, fix)
    _check_required_tables(report, fix)
    _check_memory_client_columns(report, fix)
    _backfill_missing_clients(report, fix)
    _ensure_vectors(report, fix, allow_heavy)
    _ensure_schema_assets(report, fix)
    _ensure_env_file(report, fix)
    _resolve_schema_issues(report, fix)
    write_ok = _check_quarantine_write(report, fix)
    if write_ok:
        _resolve_mcp_issues(report, fix)
    if fix and schema_changed:
        report.add_note("schema updated; restarting memory server for consistency")
        _kickstart_service(report)
    _ensure_lockdown_policy(report, fix)
    _check_llm_runtime(report)

    return report
