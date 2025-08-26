from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from .context import ProjectContext, PINAK_DIR_NAME, PINAK_CONFIG_NAME
import subprocess


def cmd_init(args: argparse.Namespace) -> int:
    root = Path.cwd()
    ctx = ProjectContext.find(root)
    if ctx and not args.force:
        print(f"Found existing Pinak project: {ctx.project_name} ({ctx.project_id}) at {ctx.root_dir}")
        return 0
    # Hard-fail if .pinak is tracked by git
    if _git_is_tracked_pinak(root):
        print("Error: .pinak/ is tracked by git. Remove from VCS and add to .gitignore before initializing.", file=sys.stderr)
        return 2
    ctx = ProjectContext.init_new(project_name=args.name, memory_url=args.url, tenant=args.tenant, root=root)
    if args.token:
        ctx.set_token(args.token)
        print("Stored token in keyring (.pinak/token if keyring unavailable).")
    pinak_dir = root / PINAK_DIR_NAME
    print(f"Initialized Pinak project at {pinak_dir / PINAK_CONFIG_NAME}")
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
        print("No Pinak project found (missing .pinak/pinak.json).")
        return 1
    has_token = bool(ctx.get_token())
    out = {
        "project_id": ctx.project_id,
        "project_name": ctx.project_name,
        "tenant": ctx.tenant,
        "memory_url": ctx.memory_url,
        "token_present": has_token,
        "fingerprint": ctx.identity_fingerprint,
        "config_path": str(Path(ctx.root_dir or Path.cwd()) / PINAK_DIR_NAME / PINAK_CONFIG_NAME)
    }
    if getattr(args, "json", False):
        print(json.dumps(out))
    else:
        print(json.dumps(out, indent=2))
    return 0


def _git_ignores_pinak(root: Path) -> bool:
    # Returns True if .pinak is not tracked and is ignored by git
    try:
        # If tracked, error-unmatch returns 0; we want False in that case
        res = subprocess.run(["git", "ls-files", "--error-unmatch", str(root/ ".pinak")], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0:
            return False  # tracked
    except Exception:
        # If git not available, best-effort: check .gitignore contains .pinak
        gi = root / ".gitignore"
        return gi.exists() and ".pinak" in gi.read_text(encoding="utf-8")
    # Check if ignored
    try:
        res = subprocess.run(["git", "check-ignore", str(root/ ".pinak")], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0
    except Exception:
        gi = root / ".gitignore"
        return gi.exists() and ".pinak" in gi.read_text(encoding="utf-8")


def _git_is_tracked_pinak(root: Path) -> bool:
    try:
        res = subprocess.run(["git", "ls-files", "--error-unmatch", str(root/ ".pinak")], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0
    except Exception:
        return False


def cmd_verify(args: argparse.Namespace) -> int:
    root = Path.cwd()
    ctx = ProjectContext.find(root)
    ok = True
    checks = {}
    if not ctx:
        checks["has_context"] = False
        ok = False
    else:
        checks["has_context"] = True
        # fingerprint recompute
        try:
            loaded_fp = ctx.identity_fingerprint
            # reload file minimal to recompute
            data = json.loads((root/ PINAK_DIR_NAME/ PINAK_CONFIG_NAME).read_text(encoding="utf-8"))
            fp_src = json.dumps({k: data[k] for k in ("project_id","project_name","tenant","memory_url","version")}, separators=(",", ":"), sort_keys=True)
            import hashlib
            fp2 = hashlib.sha256(fp_src.encode()).hexdigest()
            checks["fingerprint_match"] = (fp2 == loaded_fp)
            if fp2 != loaded_fp:
                ok = False
        except Exception:
            checks["fingerprint_match"] = False
            ok = False
    # git ignore
    gi_ok = _git_ignores_pinak(root)
    checks["git_ignores_.pinak"] = gi_ok
    if not gi_ok:
        ok = False
    # token presence
    checks["token_present"] = bool(ctx and ctx.get_token())
    result = {"ok": ok, "checks": checks}
    if getattr(args, "json", False):
        print(json.dumps(result))
    else:
        print(json.dumps(result, indent=2))
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
        service, username = ctx._token_keyring_key()
        try:
            keyring.delete_password(service, username)
        except Exception:
            pass
    except Exception:
        pass
    token_file = (Path(ctx.root_dir or Path.cwd())/ PINAK_DIR_NAME/ "token")
    try:
        if token_file.exists():
            token_file.unlink()
    except Exception:
        pass
    print("Token revoked.")
    return 0


def cmd_token_rotate(args: argparse.Namespace) -> int:
    # Local rotate is equivalent to set; server-driven rotate would be added later
    args.token = args.token or ""
    if not args.token:
        print("Provide --token for local rotation.")
        return 2
    return cmd_token_set(args)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="pinak-bridge", description="Pinak Bridge: project identity and token management")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Initialize .pinak/pinak.json in current project")
    p_init.add_argument("--name", required=True, help="Project name")
    p_init.add_argument("--url", required=True, help="Memory API base URL (e.g., http://localhost:8011)")
    p_init.add_argument("--tenant", default=None, help="Tenant/org (optional)")
    p_init.add_argument("--token", default=None, help="Store a bearer token in keyring/.pinak/token")
    p_init.add_argument("--force", action="store_true", help="Overwrite if already initialized")
    p_init.set_defaults(func=cmd_init)

    p_status = sub.add_parser("status", help="Show current project Pinak configuration")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    p_verify = sub.add_parser("verify", help="Verify .pinak config, fingerprint, and gitignore guards")
    p_verify.add_argument("--json", action="store_true")
    p_verify.set_defaults(func=cmd_verify)

    p_tset = sub.add_parser("token", help="Token management commands")
    tsub = p_tset.add_subparsers(dest="token_cmd", required=True)

    p_ts = tsub.add_parser("set", help="Store a token in keyring/.pinak/token")
    p_ts.add_argument("--token", required=True)
    p_ts.set_defaults(func=cmd_token_set)

    p_tp = tsub.add_parser("print", help="Print current token")
    p_tp.set_defaults(func=cmd_token_print)

    p_tr = tsub.add_parser("revoke", help="Revoke token from keyring/.pinak/token")
    p_tr.set_defaults(func=cmd_token_revoke)

    p_trot = tsub.add_parser("rotate", help="Rotate token (local set)")
    p_trot.add_argument("--token", required=True)
    p_trot.set_defaults(func=cmd_token_rotate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
