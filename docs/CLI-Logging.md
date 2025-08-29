# CLI Interaction Logging

## Overview

CLI Interaction Logging is a new core feature of Pinak designed to provide developers with unparalleled context and auditability of their command-line activities. By capturing every command a user runs (with context) without disrupting their shell, Pinak will store these as "CLI session events" in its memory for fast search and replay.

This feature is built with a strong emphasis on security, performance, and a clean user experience, aligning with Pinak's vision of a world-class, local-first macOS application.

## Key Features

*   **Comprehensive Capture:** Records command, arguments, timestamps, duration, exit code, current working directory, TTY, shell, and hostname.
*   **Output Management:** Captures first N lines of output and full output to artifact files with SHA256 for integrity.
*   **Sessionization:** Organizes commands into logical sessions based on terminal and idle time.
*   **Secure Storage:** Events are stored with a defined schema, with large outputs managed as separate artifacts.
*   **Intelligent Retrieval:** Provides CLI commands for searching, showing, and replaying past CLI sessions.
*   **High-Volume Handling:** Designed to handle high volumes of CLI events with buffered writing, batch uploading, and backpressure mechanisms.
*   **Privacy-First Redaction:** Implements a robust redaction pipeline to prevent sensitive information (e.g., JWTs, API keys, passwords) from being stored, using configurable patterns and secret scanning CI gates.
*   **Performance Optimization:** Utilizes a non-blocking local capture agent (`pinak-clips`) with IPC queues, batching, and resource guardrails.
*   **Configurable Retention:** Supports daily directory partitioning, compression, and configurable retention windows for artifacts.
*   **Seamless UX:** Integrates with the Pinak Bridge App for easy toggling, status indication, and quick search access.

## Security & Privacy Considerations

*   **Redaction Pipeline:** Sensitive patterns are masked with `[REDACTED]` to preserve search utility without leakage.
*   **Secret Scanning CI Gate:** Pre-commit hooks and CI gates validate redaction before packaging.
*   **ASVS Alignment:** Adheres to ASVS principles for authentication (tokens never logged), access (project scoping, RBAC for log retrieval), data at rest (file permissions, optional encryption), and error handling.
*   **Default Privacy:** Logging is off by default (opt-in), with configurable redaction levels.

## Usage (CLI)

*   `pinak session log on|off|status`: Enable, disable, or check the status of CLI logging.
*   `pinak session log search --query "<text>" --since --until --session <id> --limit`: Search for past CLI sessions.
*   `pinak session log show --id <event_id> [--full-output]`: Display details of a specific CLI event.
*   `pinak session log replay --id <event_id>`: Best-effort re-execution of a past command in a subshell.

## Configuration

Configuration options will be available via environment variables and the Pinak Bridge App preferences, allowing control over logging status, redaction levels, artifact caps, and retention days.
