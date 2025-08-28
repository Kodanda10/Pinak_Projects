from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
import socket


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run(cmd: list[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)


def ensure_docker(timeout: int = 120) -> bool:
    if not have("docker"):
        print("Docker not installed.")
        return False
    try:
        run(["docker", "info"], check=True)
        return True
    except Exception:
        # 1) Try Docker Desktop on macOS
        if sys.platform == "darwin" and have("open") and Path("/Applications/Docker.app").exists():
            print("Starting Docker Desktop…")
            run(["open","-a","Docker"], check=False)
            waited = 0
            while waited < timeout:
                try:
                    run(["docker","info"], check=True)
                    print("Docker engine is ready (Docker Desktop).")
                    return True
                except Exception:
                    time.sleep(3); waited += 3
        # 2) Fallback to Colima (macOS/Linux) for BCM
        if have("colima"):
            print("Starting Colima (fallback)…")
            # Start with modest defaults; user can override externally
            try:
                run(["colima","start","--cpu","2","--memory","4","--disk","20"], check=False)
            except Exception:
                pass
            waited = 0
            while waited < timeout:
                try:
                    run(["docker","info"], check=True)
                    print("Docker engine is ready (Colima).")
                    return True
                except Exception:
                    time.sleep(3); waited += 3
        print("Docker engine unavailable. Install Docker Desktop or Colima.")
        return False


def try_up_services() -> bool:
    base = Path.home()/"production"
    if (base/"docker-compose.yml").exists():
        try:
            run(["docker","network","create","prodnet"], check=False)
            run(["docker","compose","-f", str(base/"infra"/"docker-compose.yml"), "up","-d"], check=True)
            run(["docker","compose","-f", str(base/"docker-compose.yml"), "up","-d","--build"], check=True)
            return True
        except Exception as e:
            print(f"Compose up (production) failed: {e}")
    local = Path.cwd()/"Pinak_Services"
    if (local/"docker-compose.yml").exists():
        try:
            # Ensure dev TLS certs exist for HTTPS compose
            cert_script = Path.cwd()/"scripts"/"dev_certs.sh"
            cert_dir = local/"certs"
            if cert_script.exists() and not cert_dir.exists():
                print("Generating dev TLS certificates…")
                run(["bash", str(cert_script)], check=True)
            run(["docker","network","create","prodnet"], check=False)
            run(["docker","compose","-f", str(local/"docker-compose.yml"), "up","-d","--build"], check=True)
            return True
        except Exception as e:
            print(f"Compose up (local) failed: {e}")
    # Minimal fallback using published images
    import tempfile
    tmp = Path(tempfile.gettempdir())/"pinak-quickstart-compose.yml"
    tmp.write_text('''
services:
  memory-api:
    image: ${PINAK_IMAGE_REGISTRY:-pinak}/pinak-memory-api:latest
    ports: ["8011:8000"]
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REQUIRE_PROJECT_HEADER=true
  redis:
    image: redis:alpine
''')
    try:
        run(["docker","compose","-f", str(tmp), "pull"], check=False)
        run(["docker","compose","-f", str(tmp), "up","-d"], check=True)
        print("Started minimal Pinak services from images (if available).")
        return True
    except Exception as e:
        print(f"Minimal compose failed: {e}")
        return False


def cmd_up(args: argparse.Namespace) -> int:
    # Security preflight
    print("Running security preflight (doctor)…")
    drc = cmd_doctor(argparse.Namespace())
    if drc != 0:
        print("Preflight failed; fix issues before starting services.")
        return drc
    # Ensure data volume paths exist
    try:
        base = Path.cwd()/"Pinak_Services"/"memory_service"/"data"
        base.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: could not ensure data volume path: {e}")
    # Port availability hints (non-fatal)
    def port_in_use(port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.2)
                return s.connect_ex(("127.0.0.1", port)) == 0
        except Exception:
            return False
    for pnum, name in [(8001, 'memory-api'), (8880, 'gov-gateway'), (8800, 'parlant')]:
        if port_in_use(pnum):
            print(f"Note: port {pnum} appears in use (possibly {name}).")
    if not ensure_docker():
        return 2
    ok = try_up_services()
    if not ok:
        return 2
    # Health check retries for memory-api
    try:
        from .memory.cli import main as mem_main
        healthy = False
        for _ in range(10):
            if mem_main(["health"]) == 0:
                healthy = True
                break
            time.sleep(1)
        print("Memory API healthy." if healthy else "Memory API unhealthy; verify compose up and ports.")
    except Exception:
        pass
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    if not have("docker"):
        print("Docker not installed.")
        return 2
    # Try local compose first
    local = Path.cwd()/"Pinak_Services"/"docker-compose.yml"
    if local.exists():
        try:
            run(["docker","compose","-f", str(local), "down"], check=False)
        except Exception as e:
            print(f"Compose down (local) warning: {e}")
    # Try production locations if present
    base = Path.home()/"production"/"docker-compose.yml"
    infra = Path.home()/"production"/"infra"/"docker-compose.yml"
    for f in [infra, base]:
        if f.exists():
            try:
                run(["docker","compose","-f", str(f), "down"], check=False)
            except Exception:
                pass
    print("Services stopped (where applicable).")
    return 0


def cmd_services_status(args: argparse.Namespace) -> int:
    # Memory API health via pinak memory health
    try:
        from .memory.cli import main as mem_main
        rc = mem_main(["health"])
    except Exception:
        rc = 1
    print({"memory_api_ok": rc == 0})
    return 0 if rc == 0 else 1


def cmd_quickstart(args: argparse.Namespace) -> int:
    if not ensure_docker():
        return 2
    # Bridge init best-effort
        try:
            from .bridge.cli import main as bridge_main
            bargs = ["init","--name", args.name or Path.cwd().name, "--url", args.url or os.getenv("PINAK_MEMORY_URL","http://localhost:8011")]
            if args.tenant: bargs += ["--tenant", args.tenant]
            if args.token: bargs += ["--token", args.token]
            bridge_main(bargs)
        except SystemExit:
            pass
        except Exception as e:
            print(f"Bridge init skipped: {e}")
    if not try_up_services():
        return 2
    # If dev CA present, set default for memory client
    try:
        ca = Path.cwd()/"Pinak_Services"/"certs"/"ca.crt"
        if ca.exists():
            os.environ.setdefault("PINAK_MEMORY_CA", str(ca))
    except Exception:
        pass
    # Health check retries
    try:
        from .memory.cli import main as mem_main
        ok = False
        for _ in range(10):
            rc = mem_main(["health"])
            if rc == 0: ok = True; break
            time.sleep(1)
        print("Memory API healthy." if ok else "Memory API unhealthy; verify compose up and ports.")
    except Exception as e:
        print(f"Health check skipped: {e}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    ok = True
    # baseline
    for p in [Path("security/SECURITY-IRONCLAD.md"), Path("security/policy/ci-security-gates.yaml"), Path("SECURITY.md"), Path(".well-known/security.txt")]:
        if not p.exists():
            ok = False; print(f"Missing security file: {p}")
    if not (Path.cwd()/".pinak"/"pinak.json").exists():
        ok = False; print("No project identity found (.pinak/pinak.json)")
    # ensure .pinak is not tracked by git
    if (Path.cwd()/".git").exists():
        try:
            out = subprocess.run(["git","ls-files","--error-unmatch",".pinak"], capture_output=True)
            if out.returncode == 0:
                ok = False; print("Security: .pinak directory must not be tracked by git")
        except Exception:
            pass
    if not have("docker"):
        ok = False; print("Docker not installed")
    print("Doctor:", "OK" if ok else "Issues found")
    return 0 if ok else 2


def cmd_bridge(args: argparse.Namespace) -> int:
    from .bridge.cli import main as bridge_main
    return bridge_main(args.rest)


def cmd_memory(args: argparse.Namespace) -> int:
    from .memory.cli import main as mem_main
    return mem_main(args.rest)

def cmd_governance(args: argparse.Namespace) -> int:
    # Simple passthrough using httpx to the gateway
    import httpx, json
    base = os.getenv('PINAK_GOV_URL', 'http://localhost:8880')
    # Bridge context auto-discovery
    pid = os.getenv('PINAK_PROJECT_ID')
    tok = os.getenv('PINAK_TOKEN')
    try:
        from .bridge.context import ProjectContext
        if not pid or not tok:
            ctx = ProjectContext.find()
            if ctx:
                pid = pid or ctx.project_id
                tok = tok or ctx.get_token()
    except Exception:
        pass
    headers = {}
    if pid:
        headers['X-Pinak-Project'] = pid
    if tok:
        headers['Authorization'] = f'Bearer {tok}'
    rest = args.rest or []
    method = rest[0].upper() if rest else 'GET'
    path = rest[1] if len(rest)>1 else 'health'
    body = None
    if len(rest)>2:
        try:
            body = json.loads(' '.join(rest[2:]))
        except Exception:
            print('Body must be JSON if provided', file=sys.stderr)
            return 2
    with httpx.Client(timeout=15.0) as client:
        r = client.request(method, f"{base}/{path.lstrip('/')}", headers=headers, json=body)
        print(r.text)
        return 0 if r.status_code < 400 else 1

def cmd_token(args: argparse.Namespace) -> int:
    # Mint a JWT with pid bound to current project and optional role, then optionally store
    try:
        from jose import jwt
    except Exception:
        print("Please install 'python-jose' to use token minting", file=sys.stderr)
        return 2
    pid = os.getenv('PINAK_PROJECT_ID')
    try:
        from .bridge.context import ProjectContext
        if not pid:
            ctx = ProjectContext.find()
            if ctx:
                pid = ctx.project_id
    except Exception:
        pass
    if not pid:
        print("No project identity found. Run 'pinak bridge init' first.", file=sys.stderr)
        return 2
    import time, datetime
    claims = {"sub": args.sub or "analyst", "pid": pid}
    if args.role:
        claims["role"] = args.role
    # Optional short-lived expiry in minutes
    if args.exp is not None:
        try:
            mins = int(args.exp)
            exp_ts = int((datetime.datetime.utcnow() + datetime.timedelta(minutes=mins)).timestamp())
            claims["exp"] = exp_ts
        except Exception:
            print("Invalid --exp value; must be integer minutes", file=sys.stderr)
            return 2
    token = jwt.encode(claims, args.secret, algorithm="HS256")
    print(token)
    if args.set:
        try:
            from .bridge.context import ProjectContext
            ctx = ProjectContext.find()
            if ctx:
                ctx.set_token(token)
                print("Token stored in keyring/.pinak/token")
        except Exception:
            pass
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="pinak", description="Pinak CLI — one-click local-first setup")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quickstart"); q.add_argument("--name", default=None); q.add_argument("--url", default=None); q.add_argument("--tenant", default=None); q.add_argument("--token", default=None); q.set_defaults(func=cmd_quickstart)
    d = sub.add_parser("doctor"); d.set_defaults(func=cmd_doctor)
    b = sub.add_parser("bridge"); b.add_argument("rest", nargs=argparse.REMAINDER); b.set_defaults(func=cmd_bridge)
    m = sub.add_parser("memory"); m.add_argument("rest", nargs=argparse.REMAINDER); m.set_defaults(func=cmd_memory)
    g = sub.add_parser("governance", help="Governance passthrough to Pinak_Gov gateway")
    g.add_argument("rest", nargs=argparse.REMAINDER)
    g.set_defaults(func=cmd_governance)
    t = sub.add_parser("token"); t.add_argument("--sub", default="analyst"); t.add_argument("--role", default=None); t.add_argument("--exp", type=int, default=None, help="Expiry in minutes (optional)"); t.add_argument("--secret", default=os.getenv("SECRET_KEY","change-me-in-prod")); t.add_argument("--set", action="store_true"); t.set_defaults(func=cmd_token)

    # one-click orchestration
    up = sub.add_parser("up", help="Start Memory + Gov + Parlant (compose)"); up.set_defaults(func=cmd_up)
    dn = sub.add_parser("down", help="Stop services (compose)"); dn.set_defaults(func=cmd_down)
    st = sub.add_parser("status", help="Health check for services")
    st.set_defaults(func=cmd_services_status)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
