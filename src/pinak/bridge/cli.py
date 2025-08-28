from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Optional
import os

from .context import ProjectContext, PINAK_DIR, PINAK_CONFIG


def _git_is_tracked_pinak(root: Path) -> bool:
    try:
        res = subprocess.run(["git","ls-files","--error-unmatch", str(root/ PINAK_DIR)], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0
    except Exception:
        return False

def _git_ignores_pinak(root: Path) -> bool:
    try:
        res = subprocess.run(["git","check-ignore", str(root/ PINAK_DIR)], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0:
            return True
    except Exception:
        pass
    gi = root / ".gitignore"
    return gi.exists() and PINAK_DIR in gi.read_text(encoding="utf-8")


def cmd_init(args: argparse.Namespace) -> int:
    root = Path.cwd()
    ctx = ProjectContext.find(root)
    if ctx and not args.force:
        print(f"Found existing Pinak project: {ctx.project_name} ({ctx.project_id}) at {ctx.root_dir}")
        return 0
    if _git_is_tracked_pinak(root):
        print("Error: .pinak/ is tracked by git. Untrack and add to .gitignore first.")
        return 2
    ctx = ProjectContext.init_new(project_name=args.name, memory_url=args.url, tenant=args.tenant, root=root)
    if args.token:
        ctx.set_token(args.token)
        print("Stored token in keyring/.pinak/token.")
    print(json.dumps({
        "project_id": ctx.project_id,
        "project_name": ctx.project_name,
        "tenant": ctx.tenant,
        "memory_url": ctx.memory_url,
    }, indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    ctx = ProjectContext.find()
    if not ctx:
        print("No Pinak project found (.pinak/pinak.json missing)")
        return 1
    out = {
        "project_id": ctx.project_id,
        "project_name": ctx.project_name,
        "tenant": ctx.tenant,
        "memory_url": ctx.memory_url,
        "token_present": bool(ctx.get_token()),
        "fingerprint": ctx.identity_fingerprint,
        "config_path": str((ctx.root_dir or Path.cwd())/ PINAK_DIR/ PINAK_CONFIG),
    }
    print(json.dumps(out if args.json else out, indent=2 if not args.json else None))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    root = Path.cwd()
    ctx = ProjectContext.find(root)
    ok = True
    checks = {"has_context": bool(ctx)}
    if not ctx:
        ok = False
    gi_ok = _git_ignores_pinak(root)
    checks["git_ignores_.pinak"] = gi_ok
    if not gi_ok:
        ok = False
    tok = ctx.get_token() if ctx else None
    checks["token_present"] = bool(tok)
    # Token expiry diagnostics (non-fatal): warn if expired or expiring soon
    try:
        if tok:
            try:
                from jose import jwt
                claims = jwt.get_unverified_claims(tok)  # type: ignore[attr-defined]
                exp = int(claims.get("exp")) if claims.get("exp") is not None else None
            except Exception:
                exp = None
            import time
            now = int(time.time())
            if exp is not None:
                checks["token_expired"] = exp <= now
                checks["token_expires_in_s"] = max(0, exp - now)
                # Consider expired as a failed verify
                if exp <= now:
                    ok = False
    except Exception:
        pass
    print(json.dumps({"ok": ok, "checks": checks}, indent=2))
    return 0 if ok else 2


def cmd_token_set(args: argparse.Namespace) -> int:
    ctx = ProjectContext.find()
    if not ctx:
        print("No Pinak project found.")
        return 1
    ctx.set_token(args.token)
    print("Token stored.")
    return 0


def cmd_token_print(args: argparse.Namespace) -> int:
    ctx = ProjectContext.find()
    if not ctx:
        print("No Pinak project found.")
        return 1
    tok = ctx.get_token()
    if not tok:
        print("No token found.")
        return 2
    print(tok)
    return 0


def cmd_token_revoke(args: argparse.Namespace) -> int:
    ctx = ProjectContext.find()
    if not ctx:
        print("No Pinak project found.")
        return 1
    # Remove keyring entry and fallback file
    try:
        import keyring
        s,u = ctx._kr()
        try:
            keyring.delete_password(s,u)
        except Exception:
            pass
    except Exception:
        pass
    tf = (ctx.root_dir or Path.cwd())/ PINAK_DIR/ 'token'
    try:
        if tf.exists():
            tf.unlink()
    except Exception:
        pass
    print("Token revoked.")
    return 0


def cmd_token_rotate(args: argparse.Namespace) -> int:
    ctx = ProjectContext.find()
    if not ctx:
        print("No Pinak project found.")
        return 1
    try:
        tok = ctx.rotate_token(minutes=int(args.exp or 60), secret=args.secret, sub=args.sub or "analyst", role=args.role)
        print(tok)
        return 0
    except Exception as e:
        print(f"Rotation failed: {e}", file=sys.stderr)
        return 2


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="pinak-bridge", description="Pinak Bridge: project identity + token management")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="Initialize .pinak/pinak.json")
    pi.add_argument("--name", required=True)
    pi.add_argument("--url", required=True)
    pi.add_argument("--tenant", default=None)
    pi.add_argument("--token", default=None)
    pi.add_argument("--force", action="store_true")
    pi.set_defaults(func=cmd_init)

    ps = sub.add_parser("status", help="Show config")
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(func=cmd_status)

    pv = sub.add_parser("verify", help="Verify config + gitignore")
    pv.set_defaults(func=cmd_verify)

    pt = sub.add_parser("token", help="Token ops")
    tsub = pt.add_subparsers(dest="tcmd", required=True)
    pts = tsub.add_parser("set"); pts.add_argument("--token", required=True); pts.set_defaults(func=cmd_token_set)
    ptp = tsub.add_parser("print"); ptp.set_defaults(func=cmd_token_print)
    ptr = tsub.add_parser("revoke"); ptr.set_defaults(func=cmd_token_revoke)
    pto = tsub.add_parser("rotate"); pto.add_argument("--exp", type=int, default=60); pto.add_argument("--secret", default=os.getenv("SECRET_KEY","change-me-in-prod")); pto.add_argument("--sub", default="analyst"); pto.add_argument("--role", default=None); pto.set_defaults(func=cmd_token_rotate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
