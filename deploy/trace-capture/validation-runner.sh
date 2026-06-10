#!/bin/bash
# In-pod load driver for the Kueue validation Jobs (see validation-jobs.yaml).
# Runs the phases in $PHASES against the sidecar stack on localhost, marks the
# tap-trace line count at each phase boundary (for slicing the JSONL locally),
# then idles until the trace is fetched so the Job can complete and release
# the GPU. Self-terminates after 2h if nobody fetches.
set -euo pipefail

pip install --quiet httpx

echo "==> waiting for frontend on :8000 (engine startup includes the weight download)"
python - <<'EOF'
import time
import urllib.request

while True:
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2)
        break
    except Exception:
        time.sleep(5)
EOF

loadgen() {
    python /scripts/loadgen.py --url http://127.0.0.1:8000 --model Qwen/Qwen3-8B "$@"
}

mark() {
    wc -l </trace/trace.jsonl | tr -d ' ' >"/trace/marker-$1" 2>/dev/null || echo 0 >"/trace/marker-$1"
}

for phase in $PHASES; do
    case "$phase" in
    sweep)
        # ~1.54 wire tokens per synthetic word: spans ~0.8k-12.3k tokens across
        # the latency model's prompt buckets, at idle and loaded concurrency.
        for words in 512 1000 1500 3000 5500 8000; do
            echo "==> sweep prompt=$words words c1 (45s)"
            loadgen --pattern constant --concurrency 1 --duration 45 \
                --prompt-tokens "$words" --output-tokens 128 \
                --out "/trace/sweep-p$words-c1.json"
            echo "==> sweep prompt=$words words c8 (75s)"
            loadgen --pattern constant --concurrency 8 --duration 75 \
                --prompt-tokens "$words" --output-tokens 128 \
                --out "/trace/sweep-p$words-c8.json"
        done
        ;;
    multiturn)
        echo "==> multiturn 0.4 sessions/s x 5 turns, ~10k-token shared prefix (240s)"
        loadgen --pattern multiturn --rate 0.4 --turns 5 \
            --prefix-tokens 6500 --prompt-tokens 128 --output-tokens 128 \
            --duration 240 --seed 7 \
            --out /trace/multiturn-loadgen.json
        ;;
    *)
        echo "unknown phase: $phase" >&2
        exit 1
        ;;
    esac
    mark "$phase"
done

touch /trace/loadgen-done
echo "==> capture done; waiting for fetch (kubectl exec ... touch /trace/fetched), max 2h"
for _ in $(seq 1440); do
    [ -f /trace/fetched ] && break
    sleep 5
done
echo "==> exiting"
