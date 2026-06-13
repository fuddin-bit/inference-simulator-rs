#!/usr/bin/env python3
"""Render claude -p --output-format stream-json as a readable turn log.

Prints only deterministic content (text, tool names/inputs, tool results),
never ids, timestamps, or durations, so an act-1 transcript and its offline
act-2 replay diff clean when the loop replays byte-identically.
"""
import json
import sys

CYAN = "\033[36m"
DIM = "\033[2m"
RESET = "\033[0m"


def clip(s: str, n: int = 110) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    kind = event.get("type")
    if kind == "assistant":
        for block in event["message"].get("content", []):
            btype = block.get("type")
            if btype == "text" and block["text"].strip():
                print(f"{CYAN}agent:{RESET} {block['text'].strip()}", flush=True)
            elif btype == "tool_use":
                args = json.dumps(block.get("input", {}))
                print(f"  {CYAN}tool:{RESET} {block['name']} {DIM}{clip(args)}{RESET}", flush=True)
    elif kind == "user":
        content = event.get("message", {}).get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    inner = block.get("content")
                    if isinstance(inner, list):
                        text = " ".join(
                            b.get("text", "") for b in inner if isinstance(b, dict)
                        )
                    else:
                        text = str(inner or "")
                    if text.strip():
                        print(f"  {DIM}  -> {clip(text.strip())}{RESET}", flush=True)
    elif kind == "result":
        print(f"{CYAN}done:{RESET} {event.get('num_turns')} turns", flush=True)
