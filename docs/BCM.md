# Business Continuity for Pinak (Docker/Colima)

To ensure builds/run don’t block when Docker Desktop is unavailable, the Pinak CLI falls back to Colima automatically.

## Behavior
- pinak quickstart calls an internal ensure_docker():
  - If Docker is running, continues normally.
  - On macOS: tries to start Docker Desktop and waits up to 120s.
  - Fallback: If `colima` is installed, starts Colima (`colima start --cpu 2 --memory 4 --disk 20`) and waits for `docker info`.
  - If neither is available, exits with a clear message.

## Requirements
- macOS/Linux: install Colima (`brew install colima`) if Docker Desktop isn’t available.
- Windows: use Docker Desktop or WSL2 Docker.

## Notes
- Colima uses its own VM and socket; `docker` CLI will point to the Colima socket after `colima start`.
- Resources are modest by default; adjust via `colima start` flags.

