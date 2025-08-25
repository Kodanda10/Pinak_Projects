# Agent Bootstrap (Generic)

Before any task, the agent must read and honor:
1) `/security/SECURITY-IRONCLAD.md`
2) `/security/policy/ci-security-gates.yaml`
3) Repo-level `SECURITY.md` or `/.well-known/security.txt` (if present)

**Mandatory checks**
- If files are missing or outdated vs. baseline, halt and request remediation.
- When proposing changes, align with CI gates and ASVS controls referenced here.
