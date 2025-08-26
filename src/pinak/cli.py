from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run(cmd: list[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)


def ensure_docker(timeout: int = 120) -> bool:
    if not have("docker"):
        print("Docker not installed. Please install Docker Desktop or Colima.")
        return False
    try:
        run(["docker", "info"], check=True)
        return True
    except Exception:
        # macOS: try to start Docker Desktop
        if sys.platform == "darwin" and have("open") and Path("/Applications/Docker.app").exists():
            print("Starting Docker Desktop…")
            try:
                run(["open", "-a", "Docker"], check=False)
            except Exception:
                pass
            import time
            waited = 0
            while waited < timeout:
                try:
                    run(["docker", "info"], check=True)
                    print("Docker engine is ready.")
                    return True
                except Exception:
                    time.sleep(3)
                    waited += 3
        print("Docker engine unavailable. Please start Docker.")
        return False


def try_up_services() -> bool:
    """Bring up services via docker compose in a known workspace layout.

    Prefers /Users/<user>/production/infra + docker-compose.yml. Falls back to ./Pinak_Services.
    """
    root = Path.cwd()
    prod = Path("/Users") / Path.home().name / "production"
    if (prod / "docker-compose.yml").exists():
        try:
            run(["docker", "network", "create", "prodnet"], check=False)
            run(["docker", "compose", "-f", str(prod / "infra" / "docker-compose.yml"), "up", "-d"], check=True)
            run(["docker", "compose", "-f", str(prod / "docker-compose.yml"), "up", "-d", "--build"], check=True)
            return True
        except Exception as e:
            print(f"Compose up (production) failed: {e}")
    # Fallback to local services folder if present
    local_services = root / "Pinak_Services"
    if (local_services / "docker-compose.yml").exists():
        try:
            run(["docker", "network", "create", "prodnet"], check=False)
            run(["docker", "compose", "-f", str(local_services / "docker-compose.yml"), "up", "-d", "--build"], check=True)
            return True
        except Exception as e:
            print(f"Compose up (local) failed: {e}")
    print("No compose files found. Please open the production repo or Pinak_Services.")
    return False


def cmd_doctor(args: argparse.Namespace) -> int:
    ok = True
    # Security baseline check
    base_files = [
        Path("security/SECURITY-IRONCLAD.md"),
        Path("security/policy/ci-security-gates.yaml"),
        Path("SECURITY.md"),
        Path(".well-known/security.txt"),
    ]
    for p in base_files:
        if not p.exists():
            ok = False
            print(f"Missing security file: {p}")
    # Bridge config check
    pinak_json = Path(".pinak/pinak.json")
    if not pinak_json.exists():
        print("No project identity found (.pinak/pinak.json). Run: pinak bridge init …")
        ok = False
    # Docker availability check
    if not have("docker"):
        print("Docker not installed (required for one-click services).")
        ok = False
    print("Doctor: ", "OK" if ok else "Issues found")
    return 0 if ok else 2


def cmd_quickstart(args: argparse.Namespace) -> int:
    # 1) Ensure Docker
    if not ensure_docker():
        return 2
    # 2) Bridge init (best-effort)
    try:
        from .bridge.cli import main as bridge_main
        bridge_args = [
            "init", "--name", args.name or Path.cwd().name,
            "--url", args.url or os.getenv("PINAK_MEMORY_URL", "http://localhost:8011"),
        ]
        if args.tenant:
            bridge_args += ["--tenant", args.tenant]
        if args.token:
            bridge_args += ["--token", args.token]
        bridge_main(bridge_args)
    except SystemExit:
        pass
    except Exception as e:
        print(f"Bridge init skipped: {e}")
    # 3) Bring up services
    if not try_up_services():
        return 2
    # 4) Health check via pinak-memory
    try:
        from .memory.cli import main as mem_main
        print("Checking memory health…")
        rc = mem_main(["health"])  # relies on bridge context
        if rc == 0:
            print("Memory API healthy.")
        else:
            print("Memory API unhealthy; verify compose up and ports.")
    except Exception as e:
        print(f"Health check skipped: {e}")
    print("Quickstart complete.")
    return 0


def cmd_bridge(args: argparse.Namespace) -> int:
    from .bridge.cli import main as bridge_main
    return bridge_main(args.rest)


def cmd_memory(args: argparse.Namespace) -> int:
    from .memory.cli import main as mem_main
    return mem_main(args.rest)


def cmd_token(args: argparse.Namespace) -> int:
    # Convenience wrapper for dev tokens
    script = Path(__file__).parent.parent.parent / "scripts" / "dev_token.sh"
    if not script.exists():
        print("dev_token.sh not found; ensure Pinak_Package/scripts present.")
        return 2
    cmd = ["bash", str(script)]
    if args.sub:
        cmd += ["--sub", args.sub]
    if args.secret:
        cmd += ["--secret", args.secret]
    if args.set:
        cmd += ["--set"]
    run(cmd, check=False)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="pinak", description="Pinak CLI — one-click local-first setup")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("quickstart", help="One-click: bridge init + services up + health")
    q.add_argument("--name", default=None)
    q.add_argument("--url", default=None)
    q.add_argument("--tenant", default=None)
    q.add_argument("--token", default=None)
    q.set_defaults(func=cmd_quickstart)

    d = sub.add_parser("doctor", help="Check security baseline, bridge, and docker availability")
    d.set_defaults(func=cmd_doctor)

    b = sub.add_parser("bridge", help="Bridge subcommands passthrough")
    b.add_argument("rest", nargs=argparse.REMAINDER)
    b.set_defaults(func=cmd_bridge)

    m = sub.add_parser("memory", help="Memory CLI passthrough")
    m.add_argument("rest", nargs=argparse.REMAINDER)
    m.set_defaults(func=cmd_memory)

    t = sub.add_parser("token", help="Mint dev tokens (helper)")
    t.add_argument("--sub", default="analyst")
    t.add_argument("--secret", default=os.getenv("SECRET_KEY", "change-me-in-prod"))
    t.add_argument("--set", action="store_true")
    t.set_defaults(func=cmd_token)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

