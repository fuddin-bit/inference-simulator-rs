#!/usr/bin/env bash
# Run headless Claude Code against a rig (capture or replay), in a clean
# scratch workspace, rendering each agent turn as it happens. Usage:
#   demo/run-agent.sh <base_url> <workspace> ["task prompt"]
# The same invocation against the capture rig and the replay rig is the whole
# demo: identical turn-by-turn transcripts, one of them with zero GPUs.
#
# The default task is deliberately multi-step, and its tool outputs are
# deterministic (plain python asserts, no pytest timing lines), so the replay
# reconstructs every prompt byte-for-byte.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BASE_URL=$1
WORKSPACE=$2
TASK=${3:-"Build a small calculator module step by step: (1) write calculator.py with add, sub, mul, and div functions, where div raises ValueError on division by zero; (2) write test_calculator.py that tests all four functions including the div-by-zero error using plain asserts, ending with print('ALL TESTS PASSED'); (3) run the tests with python3 and show the output; (4) if anything fails, fix it and re-run until tests pass."}
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"

rm -rf "$WORKSPACE" && mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

env -u ANTHROPIC_MODEL \
  ANTHROPIC_BASE_URL="$BASE_URL" \
  ANTHROPIC_AUTH_TOKEN=dummy \
  ANTHROPIC_API_KEY=dummy \
  ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL" \
  ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" \
  ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL" \
  CLAUDE_CODE_ATTRIBUTION_HEADER=0 \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
  CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192 \
  claude -p "$TASK" \
  --model "$MODEL" --max-turns 16 \
  --allowedTools "Write,Bash,Read,Edit" --strict-mcp-config \
  --output-format stream-json --verbose \
  | python3 "$SCRIPT_DIR/format-stream.py"
