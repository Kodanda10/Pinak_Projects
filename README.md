# Pinak_Project: An Integrated AI Development Ecosystem

Welcome to the `Pinak_Project`. This repository contains a suite of powerful, local-first tools designed to accelerate and secure the development of AI-powered applications. The architecture is designed to be modular, scalable, and adhere to modern, FANG-level engineering principles.

## Core Components

The ecosystem is composed of two primary parts that work together:

1.  **`Pinak_Package` (The Core SDK):** This is a lightweight Python package that you install in your projects (`pip install -e .`). It acts as a client or SDK, providing a simple, high-level interface to the powerful backend services. It contains:
    *   `pinak.memory`: A client for interacting with the Memory Service.
    *   `pinak.security`: A client for the Security Service.
    *   `pinak.env_manager` (Planned): A client for the Environment Management Service.

2.  **`Pinak_Services` (The Microservices Backend):** This is a containerized, high-performance backend built using Docker. It runs the heavy-lifting services that the `Pinak_Package` communicates with.
    *   **`memory-api`:** A FastAPI service that manages the agent's long-term memory, using a vector database for semantic search.
    *   **`redis`:** A Redis instance for caching and short-term working memory.

---

## Architecture Evolution

This project has evolved through distinct phases to achieve its current robust architecture.

*   **Phase 1: The Monolithic Library:** The project began as a standard Python library where all logic (memory, security) was contained within the installable package. While simple, this approach was not scalable or decoupled.

*   **Phase 2: Evolution to Microservices:** To achieve FANG-level scalability and modularity, the system was refactored into a client-server model. The heavy logic was moved into the standalone, containerized `Pinak_Services`, and the `Pinak_Package` was transformed into a lightweight API client. This decouples the agent's application logic from the backend services, allowing them to be scaled, updated, and maintained independently.

---

## Getting Started

To use the Pinak ecosystem, you need Git and Docker Desktop installed.

**1. Run the Backend Services:**

First, start the containerized services. From the `Pinak_Package` root directory, navigate to the services directory and use `docker-compose`:

```bash
cd Pinak_Services/
docker-compose up --build
```

This will launch the Memory API and Redis. The API will be available at `http://localhost:8001`.

### Using the SDK with Auth (recommended)

The Memory API in this repo requires a Bearer token. Pass it explicitly or via environment variables.

```python
from pinak.memory.manager import MemoryManager

# Option A: provide token explicitly (e.g., from OIDC/JWT flow)
mm = MemoryManager(service_base_url="http://localhost:8001", token="<your-jwt>")

# Option B: env vars
# export PINAK_MEMORY_URL=http://localhost:8001
# export PINAK_TOKEN=eyJhbGciOi...
# mm = MemoryManager()

mm.add_memory("test memory", ["test"]) 
print(mm.search_memory("test"))
```

In dev, you can mint a test-only JWT signed with the API `SECRET_KEY` (HS256). In production, obtain real tokens via your IdP (OIDC).

**2. Install the SDK in Your Project:**

In any other project where you want to build an agent, activate its virtual environment and install the `Pinak_Project` package in editable mode:

```bash
pip install -e /path/to/your/Pinak_Package
```

You can now import the clients and use them in your code.

---

## CLI Quickstart

Install the package (editable or from PyPI when published), then use the CLI:

```
export PINAK_MEMORY_URL=http://localhost:8001
export PINAK_TOKEN=<your-jwt-if-required>

pinak-memory health         # -> {"ok": true/false}
pinak-memory add "hello" --tags demo
pinak-memory search "hello"
```

The CLI uses the same auth model as the SDK: pass `--url`/`--token` flags or rely on `PINAK_MEMORY_URL`/`PINAK_TOKEN`.

---

## Roadmap: Phase 3 - The Pinak-Env Service

To address the complexity of managing development environments (running servers, terminals, etc.), the next major capability is the **Pinak-Env** service.

### Vision

The Pinak-Env service will be a new module, `pinak.env_manager`, that allows an agent (or a non-technical user) to manage a full development workflow using simple, high-level commands, abstracting away the underlying terminal operations.

### Core Features

1.  **Configuration-Driven:** The `EnvManager` will read a `pinak_project.json` file in the root of any project to understand its structure (e.g., what the `frontend` and `backend` start commands are).
2.  **Autonomous Process Management:** It will use Python's `subprocess` module to run processes (like dev servers, build scripts) in the background, completely detached from the user's terminal.
3.  **Stateful Tracking:** It will maintain a state of all running processes, including their Process ID (PID), status (`running`, `stopped`, `error`), and start time.
4.  **Real-time Log Capture:** It will capture the `stdout` and `stderr` streams from all background processes, allowing the user to request logs for a specific service at any time.
5.  **Graceful Termination:** It will provide commands to stop specific services or shut down the entire environment gracefully.

### Example Usage (Conceptual)

Once built, a user could interact with their project like this:

```python
from pinak.env_manager.agent import EnvManager

# EnvManager loads the project's pinak_project.json
env = EnvManager("/path/to/my/new/webapp")

# User gives a high-level command
print("Starting backend server...")
env.run("backend") # -> runs 'python manage.py runserver' from config

print("Starting frontend dev server...")
env.run("frontend") # -> runs 'npm run dev' from config

# Later, check the status
status = env.get_status()
# -> {'backend': {'status': 'running', ...}, 'frontend': {'status': 'running', ...}}

# Get logs from the backend
logs = env.get_logs("backend")

# Shut everything down
env.stop_all()
```

This will provide a powerful, automated layer for project management, truly simplifying the developer experience.

