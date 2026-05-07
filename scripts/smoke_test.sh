#!/usr/bin/env bash
# Axion AI — smoke test
#
# Starts uvicorn on an ephemeral port (8765), waits for /api/health to return
# 200, hits key endpoints, validates JSON shape, then kills the server.
# Exit 0 on success, non-zero on any failure.
#
# Usage:
#   bash scripts/smoke_test.sh
#   make smoke

set -euo pipefail

PORT=8765
BASE="http://127.0.0.1:${PORT}"
SERVER_PID=""
FAILED=0

# If AXION_API_KEY is set, pass it as a header on every authenticated request.
CURL_AUTH_ARGS=()
if [ -n "${AXION_API_KEY:-}" ]; then
    CURL_AUTH_ARGS=("-H" "X-API-Key: ${AXION_API_KEY}")
    log "Auth: X-API-Key header will be sent (AXION_API_KEY is set)"
else
    log "Auth: disabled (AXION_API_KEY not set)"
fi

# ---------------------------------------------------------------------------
# Pre-flight: check that uvicorn and fastapi are importable
# ---------------------------------------------------------------------------

if ! python3 -c "import uvicorn, fastapi" 2>/dev/null; then
    echo "[smoke] ERROR: uvicorn / fastapi not found in the active Python environment."
    echo "[smoke] Run:  pip install -r requirements.txt"
    exit 2
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { echo "[smoke] $*"; }
fail() { echo "[FAIL] $*" >&2; FAILED=1; }

# Validate that $1 (JSON string) contains key $2 with optional expected value $3
json_has_key() {
    local json="$1" key="$2"
    if ! echo "$json" | python3 -c "import sys,json; d=json.load(sys.stdin); assert '$key' in d" 2>/dev/null; then
        fail "Response missing key '$key'"
    fi
}

json_array_nonempty() {
    local json="$1" key="$2"
    if ! echo "$json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
arr = d.get('$key') if isinstance(d, dict) else d
assert isinstance(arr, list) and len(arr) > 0, f'expected non-empty list at $key'
" 2>/dev/null; then
        fail "Expected non-empty array at '$key'"
    fi
}

# ---------------------------------------------------------------------------
# Start server
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log "Starting uvicorn on port $PORT …"
cd "$PROJECT_ROOT"
python3 -m uvicorn api.server:app --host 127.0.0.1 --port "$PORT" \
    --log-level warning 2>/dev/null &
SERVER_PID=$!

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Stopping server (PID $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [ "$FAILED" -ne 0 ]; then
        echo ""
        echo "SMOKE TEST FAILED — see [FAIL] lines above."
        exit 1
    else
        echo ""
        echo "SMOKE TEST PASSED."
        exit 0
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Wait for /api/health
# ---------------------------------------------------------------------------

log "Waiting for /api/health (timeout 30s) …"
MAX_WAIT=30
WAITED=0
while true; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/api/health" 2>/dev/null || true)
    if [ "$STATUS" = "200" ]; then
        log "/api/health → 200 OK (${WAITED}s)"
        break
    fi
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        fail "/api/health did not return 200 within ${MAX_WAIT}s (last status: $STATUS)"
        exit 1
    fi
    sleep 1
    WAITED=$((WAITED + 1))
done

# ---------------------------------------------------------------------------
# Endpoint checks
# ---------------------------------------------------------------------------

check() {
    local label="$1" url="$2"
    local body status
    status=$(curl -s -o /tmp/smoke_body.json -w "%{http_code}" \
        "${CURL_AUTH_ARGS[@]}" "$url" 2>/dev/null || true)
    body=$(cat /tmp/smoke_body.json 2>/dev/null || echo "")
    if [ "$status" != "200" ]; then
        fail "$label → HTTP $status (expected 200)"
        return
    fi
    if ! echo "$body" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
        fail "$label → response is not valid JSON"
        return
    fi
    echo "$body"
}

# --- /api/scenarios ---
log "GET /api/scenarios"
body=$(check "/api/scenarios" "$BASE/api/scenarios")
if [ $FAILED -eq 0 ]; then
    if ! echo "$body" | python3 -c "
import sys, json
d = json.load(sys.stdin)
scenarios = d.get('scenarios', d if isinstance(d, list) else [])
assert len(scenarios) > 0, 'expected at least 1 scenario'
" 2>/dev/null; then
        fail "/api/scenarios — no scenarios returned"
    else
        log "  → OK ($(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('scenarios', d if isinstance(d,list) else [])))" 2>/dev/null) scenarios)"
    fi
fi

# --- /api/state ---
log "GET /api/state"
body=$(check "/api/state" "$BASE/api/state")
if [ $FAILED -eq 0 ]; then
    json_has_key "$body" "timestamp"
    log "  → OK"
fi

# --- /api/recommendations ---
log "GET /api/recommendations"
body=$(check "/api/recommendations" "$BASE/api/recommendations")
if [ $FAILED -eq 0 ]; then
    if ! echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); assert isinstance(d,list) or isinstance(d.get('recommendations',[]),list)" 2>/dev/null; then
        fail "/api/recommendations — unexpected shape"
    else
        log "  → OK"
    fi
fi

# --- /api/predictive/forecast ---
log "GET /api/predictive/forecast"
STATUS=$(curl -s -o /tmp/smoke_body.json -w "%{http_code}" \
    "${CURL_AUTH_ARGS[@]}" "$BASE/api/predictive/forecast" 2>/dev/null || true)
body=$(cat /tmp/smoke_body.json 2>/dev/null || echo "")
if [ "$STATUS" = "200" ]; then
    if ! echo "$body" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
        fail "/api/predictive/forecast — invalid JSON"
    else
        log "  → OK"
    fi
elif [ "$STATUS" = "503" ] || [ "$STATUS" = "404" ]; then
    # LSTM model not trained — degraded mode is acceptable
    log "  → $STATUS (LSTM model not loaded — degraded mode, acceptable)"
else
    fail "/api/predictive/forecast → HTTP $STATUS"
fi
