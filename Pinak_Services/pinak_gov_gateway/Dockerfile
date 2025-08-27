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
