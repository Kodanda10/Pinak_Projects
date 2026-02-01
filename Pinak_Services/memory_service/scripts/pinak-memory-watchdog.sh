#!/bin/bash

set -euo pipefail

SERVER_URL="${PINAK_API_URL:-http://127.0.0.1:8000}"
HEALTH_URL="${PINAK_HEALTH_URL:-${SERVER_URL%/}/api/v1/health}"
PLIST="/Users/abhi-macmini/Library/LaunchAgents/com.pinak.memory.server.plist"
LABEL="com.pinak.memory.server"
LOG_FILE="/Users/abhi-macmini/Library/Logs/pinak-memory-watchdog.log"
STATE_FILE="/tmp/pinak-memory-watchdog.state"
COOLDOWN_SECONDS=300
STARTUP_GRACE_SECONDS=180
FAILURE_THRESHOLD=3
FALLBACK_SCRIPT="/Users/abhi-macmini/clawd-simba/Pinak_Projects/Pinak_Services/memory_service/scripts/pinak-memory-server.sh"

log() {
  /bin/echo "[$(/bin/date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

read_state() {
  LAST_RESTART_EPOCH=0
  FAILURES=0
  FALLBACK_PID=0
  if [ -f "$STATE_FILE" ]; then
    LAST_RESTART_EPOCH=$(/usr/bin/awk -F= '/^last_restart=/{print $2}' "$STATE_FILE" 2>/dev/null || echo 0)
    FAILURES=$(/usr/bin/awk -F= '/^failures=/{print $2}' "$STATE_FILE" 2>/dev/null || echo 0)
    FALLBACK_PID=$(/usr/bin/awk -F= '/^fallback_pid=/{print $2}' "$STATE_FILE" 2>/dev/null || echo 0)
  fi
}

write_state() {
  /bin/echo "last_restart=$LAST_RESTART_EPOCH" > "$STATE_FILE"
  /bin/echo "failures=$FAILURES" >> "$STATE_FILE"
  /bin/echo "fallback_pid=$FALLBACK_PID" >> "$STATE_FILE"
}

check_health() {
  local http_code curl_status
  set +e
  http_code=$(/usr/bin/curl -s -o /dev/null -w "%{http_code}" --max-time 2 "$HEALTH_URL" 2>>"$LOG_FILE")
  curl_status=$?
  set -e
  if [ "$curl_status" -ne 0 ]; then
    echo "000"
    return 1
  fi
  echo "$http_code"
  return 0
}

fallback_running() {
  if [ "$FALLBACK_PID" -gt 0 ]; then
    if /bin/ps -p "$FALLBACK_PID" >/dev/null 2>&1; then
      return 0
    fi
  fi
  return 1
}

start_fallback() {
  if [ ! -x "$FALLBACK_SCRIPT" ]; then
    log "fallback_missing script=$FALLBACK_SCRIPT"
    return
  fi
  if fallback_running; then
    log "fallback_running pid=$FALLBACK_PID"
    return
  fi
  log "fallback_starting script=$FALLBACK_SCRIPT"
  nohup "$FALLBACK_SCRIPT" >> "$LOG_FILE" 2>&1 &
  FALLBACK_PID=$!
  write_state
  log "fallback_started pid=$FALLBACK_PID"
}

read_state
http_code="$(check_health || true)"
if [ "$http_code" = "200" ]; then
  FAILURES=0
  write_state
  exit 0
fi

/bin/sleep 2
http_code="$(check_health || true)"
if [ "$http_code" = "200" ]; then
  FAILURES=0
  write_state
  log "health_check_recovered_after_retry http_code=${http_code}"
  exit 0
fi

FAILURES=$((FAILURES + 1))
write_state
log "health_check_failed http_code=${http_code:-unknown} failures=$FAILURES url=$SERVER_URL/"

if [ "$FAILURES" -lt "$FAILURE_THRESHOLD" ]; then
  exit 0
fi

if [ "$http_code" != "200" ]; then
  if /bin/launchctl print "gui/$(/usr/bin/id -u)/$LABEL" >/dev/null 2>&1; then
    if /bin/launchctl print "gui/$(/usr/bin/id -u)/$LABEL" | /usr/bin/grep -q "state = running"; then
      /bin/sleep 5
      http_code="$(check_health || true)"
      if [ "$http_code" = "200" ]; then
        log "health_check_recovered_after_wait http_code=${http_code}"
        exit 0
      fi
    fi

    now_epoch=$(/bin/date +%s)
    if [ $((now_epoch - LAST_RESTART_EPOCH)) -lt "$STARTUP_GRACE_SECONDS" ]; then
      log "restart_skipped startup_grace last_restart=${LAST_RESTART_EPOCH}"
      exit 0
    fi
    if [ $((now_epoch - LAST_RESTART_EPOCH)) -lt "$COOLDOWN_SECONDS" ]; then
      log "restart_skipped cooldown_active last_restart=${LAST_RESTART_EPOCH}"
      exit 0
    fi

    LAST_RESTART_EPOCH="$now_epoch"
    FAILURES=0
    write_state
    if output=$(/bin/launchctl kickstart -k "gui/$(/usr/bin/id -u)/$LABEL" 2>&1); then
      log "kickstart_requested label=$LABEL"
    else
      log "kickstart_failed label=$LABEL error=$output"
    fi
  else
    LAST_RESTART_EPOCH=$(/bin/date +%s)
    FAILURES=0
    write_state
    if output=$(/bin/launchctl bootstrap "gui/$(/usr/bin/id -u)" "$PLIST" 2>&1); then
      log "bootstrap_requested plist=$PLIST"
    else
      log "bootstrap_failed plist=$PLIST error=$output"
    fi
  fi

  for attempt in 1 2 3; do
    /bin/sleep 1
    http_code="$(check_health || true)"
    if [ "$http_code" = "200" ]; then
      log "restart_ok http_code=${http_code} attempt=$attempt"
      exit 0
    fi
  done

  log "restart_failed http_code=${http_code:-unknown} attempts=3"
  start_fallback
fi
