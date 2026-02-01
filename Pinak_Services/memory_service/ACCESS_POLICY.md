# Pinak Memory Access Policy

- This repository contains protected pinak-memory source code.
- By default the repo is locked (immutable + owner-only permissions).
- Changes require macOS admin password via `scripts/pinak-unlock.sh` (admin only).
- Keep `~/pinak-memory` for shared schemas/templates; do not store secrets here.
- Use `scripts/pinak-lockdown.sh` to re-lock after changes.
