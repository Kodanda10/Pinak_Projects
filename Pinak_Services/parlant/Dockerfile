FROM python:3.11-slim

#
# Parlant server with optional Pinak memory integration
# - Non-root user for compliance with image policy gates
# - Optional build-arg INSTALL_PINAK=1 to install local Pinak package
# - Expect to run with --read-only and a writable /data volume
#

ENV PIP_NO_CACHE_DIR=1
WORKDIR /app

# System prep and Python tooling
RUN pip install --upgrade pip

# Install Parlant server
RUN pip install parlant==3.0.1

# Optionally install Pinak from the build context (repo root)
# Build with:  docker build -f Pinak_Services/parlant/Dockerfile --build-arg INSTALL_PINAK=1 .
# Note: The build context must be the repository root so COPY . works as expected.
ARG INSTALL_PINAK=0
COPY . /opt/pinak_src
RUN if [ "$INSTALL_PINAK" = "1" ]; then \
        echo "Installing Pinak package from /opt/pinak_src" && \
        pip install /opt/pinak_src; \
    else \
        echo "Skipping Pinak package install (INSTALL_PINAK!=1)"; \
    fi

# Provide a default data dir
ENV PARLANT_DATA_DIR=/data
RUN mkdir -p /data

# Create and switch to an unprivileged user (policy: user != root)
RUN groupadd -r app && useradd --no-log-init -r -g app app && \
    chown -R app:app /app /data /opt/pinak_src
USER app

# Expose default port
EXPOSE 8800

# Run the official server entrypoint
CMD ["parlant-server"]

# Basic container healthcheck
USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
USER app
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 CMD curl -fsS http://127.0.0.1:8800/ || exit 1
