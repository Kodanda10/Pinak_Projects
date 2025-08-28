FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt \
    && useradd --no-log-init -r -m -d /home/app -s /usr/sbin/nologin app \
    && mkdir -p /app/app \
    && chown -R app:app /app /home/app
COPY app /app/app
USER app
ENV GOV_UPSTREAM=http://parlant:8800 \
    SECRET_KEY=change-me-in-prod \
    REQUIRE_PROJECT_HEADER=true
EXPOSE 8880
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8880"]

# Basic container healthcheck
USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
USER app
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 CMD curl -fsS http://127.0.0.1:8880/health || exit 1
