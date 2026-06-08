#!/usr/bin/env bash
#
# Phase 0: prove the real vLLM frontend ferries `kv_transfer_params` to and from our
# engine over the engine-core protocol. This is the prerequisite for P/D behind the
# llm-d routing sidecar, and it needs no NIXL (pure protocol passthrough), so it runs
# anywhere bird one runs.
#
#   request kv_transfer_params:{do_remote_decode:true}
#        ─▶ vllm-rs  ─▶ sampling_params.extra_args["kv_transfer_params"] ─▶ our engine
#   our engine output kv_transfer_params:{...} ─▶ vllm-rs ─▶ HTTP response
#
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
HANDSHAKE_PORT="${HANDSHAKE_PORT:-29551}"
HTTP_HOST="${HTTP_HOST:-127.0.0.1}"
HTTP_PORT="${HTTP_PORT:-8001}"
FRONTEND_BIN="${FRONTEND_BIN:-$HOME/git/vllm-main/rust/target/debug/vllm-rs}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_BIN="$REPO_ROOT/target/debug/mock-engine-nixl"
BASE_URL="http://${HTTP_HOST}:${HTTP_PORT}"
LOG_DIR="$(mktemp -d)"

frontend_pid=""
engine_pid=""
cleanup() {
    [[ -n "$engine_pid" ]] && kill "$engine_pid" 2>/dev/null || true
    [[ -n "$frontend_pid" ]] && kill "$frontend_pid" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

fail() {
    echo "FAIL: $*" >&2
    echo "--- engine log ---" >&2; tail -40 "$LOG_DIR/engine.log" >&2 || true
    echo "--- frontend log ---" >&2; tail -20 "$LOG_DIR/frontend.log" >&2 || true
    exit 1
}

[[ -x "$FRONTEND_BIN" ]] || fail "frontend binary not found at $FRONTEND_BIN"
[[ -x "$ENGINE_BIN" ]] || { echo "building engine..."; (cd "$REPO_ROOT" && cargo build); }

echo "logs: $LOG_DIR"

"$FRONTEND_BIN" serve "$MODEL" \
    --data-parallel-size 1 --data-parallel-size-local 0 \
    --handshake-port "$HANDSHAKE_PORT" --host "$HTTP_HOST" --port "$HTTP_PORT" \
    >"$LOG_DIR/frontend.log" 2>&1 &
frontend_pid=$!

"$ENGINE_BIN" --handshake-address "tcp://127.0.0.1:${HANDSHAKE_PORT}" --log-requests \
    >"$LOG_DIR/engine.log" 2>&1 &
engine_pid=$!

echo "waiting for $BASE_URL/health ..."
for i in $(seq 1 120); do
    kill -0 "$frontend_pid" 2>/dev/null || fail "frontend exited during startup"
    kill -0 "$engine_pid" 2>/dev/null || fail "engine exited during startup"
    curl -fsS "$BASE_URL/health" >/dev/null 2>&1 && break
    sleep 1
    [[ "$i" == "120" ]] && fail "server not healthy within 120s"
done

# Prefill-style request: carry kv_transfer_params with do_remote_decode, max_tokens 1.
echo "--- sending request with kv_transfer_params ---"
RESP=$(curl -fsS "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":1,\"kv_transfer_params\":{\"do_remote_decode\":true}}") \
    || fail "request failed"
echo "$RESP"

# Direction 1 (input): the engine logged the incoming kv_transfer_params.
grep -q "received kv_transfer_params from frontend" "$LOG_DIR/engine.log" \
    || fail "engine never logged the incoming kv_transfer_params (input passthrough broken)"
grep -q "do_remote_decode" "$LOG_DIR/engine.log" \
    || fail "engine did not receive do_remote_decode in extra_args"

# Direction 2 (output): the engine's kv_transfer_params surfaced in the HTTP response.
echo "$RESP" | grep -q '"remote_engine_id":"mock-engine-nixl"' \
    || fail "engine kv_transfer_params did not surface in the response (output passthrough broken)"
echo "$RESP" | grep -q '"phase0_probe":true' \
    || fail "phase0 probe marker missing from response"

echo ""
echo "PASS: kv_transfer_params flows frontend->engine (extra_args) and engine->frontend (output). P/D channel works."
