#!/usr/bin/env python3
"""Minimal streaming load generator for tap captures.

Drives an OpenAI-compatible /v1/completions endpoint with synthetic fixed-size
prompts, measuring client-side TTFT and per-token ITL. Four arrival patterns:

  constant   closed loop: --concurrency workers, each fires its next request
             the moment the previous one finishes (the guidellm-style default)
  poisson    open loop: exponential inter-arrivals at --rate req/s, no
             concurrency cap (late responses do not slow down arrivals)
  staircase  closed loop: sweep concurrency levels over the run; --stairs
             "1,2,4,8" or a doubling ramp up to --concurrency by default,
             each level held for duration/len(levels)
  burst      open loop: --burst-size simultaneous requests every
             --burst-interval seconds

The server-side truth comes from the tap; this exists so the same run yields
both views without depending on guidellm's scheduler. Every pattern records
arrival_ms (offset from run start) and the live in-flight count at send time,
which is what an open-loop workload replay needs.

Usage:
  uv run --with httpx loadgen.py --url http://127.0.0.1:8000 \
      --model Qwen/Qwen3-8B --pattern poisson --rate 4 --duration 120 \
      --prompt-tokens 512 --output-tokens 128 --out poisson.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time

import httpx

# ~1 token/word for common BPE vocabs; the tap records the true prompt_tokens
# from the wire, so approximate sizing here is fine.
WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
]


class Gauge:
    """Live in-flight request count. Single-threaded asyncio, no lock needed."""

    def __init__(self) -> None:
        self.value = 0


def make_prompt(rng: random.Random, n_tokens: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_tokens))


async def one_request(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    rng: random.Random,
    run_start: float,
    inflight: Gauge,
) -> dict | None:
    prompt = make_prompt(rng, args.prompt_tokens)
    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.output_tokens,
        "ignore_eos": True,
        "stream": True,
    }
    inflight.value += 1
    concurrency = inflight.value
    start = time.perf_counter()
    stamps: list[float] = []
    try:
        async with client.stream("POST", f"{args.url}/v1/completions", json=body) as r:
            if r.status_code != 200:
                return {"error": f"http {r.status_code}"}
            async for line in r.aiter_lines():
                if not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                stamps.append(time.perf_counter())
    except httpx.HTTPError as e:
        return {"error": str(e)}
    finally:
        inflight.value -= 1
    if not stamps:
        return {"error": "no tokens"}
    first, last = stamps[0], stamps[-1]
    chunks = len(stamps)
    itl_ms = [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])]
    return {
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": chunks,
        "ttft_ms": (first - start) * 1000.0,
        "itl_mean_ms": ((last - first) / (chunks - 1)) * 1000.0 if chunks > 1 else None,
        "itl_ms": itl_ms,
        "concurrency": concurrency,
        "arrival_ms": (start - run_start) * 1000.0,
    }


async def fire(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    rng: random.Random,
    run_start: float,
    inflight: Gauge,
    results: list,
) -> None:
    res = await one_request(client, args, rng, run_start, inflight)
    if res is not None:
        results.append(res)


async def closed_loop_worker(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    deadline: float,
    results: list,
    seed: int,
    run_start: float,
    inflight: Gauge,
    index: int,
    levels: list[int],
    step_s: float,
) -> None:
    """One closed-loop slot. Active while its index is below the current level;
    a single constant level degenerates to today's fixed-concurrency worker."""
    rng = random.Random(seed)
    while (now := time.perf_counter()) < deadline:
        step = min(int((now - run_start) / step_s), len(levels) - 1)
        if index >= levels[step]:
            await asyncio.sleep(0.05)
            continue
        await fire(client, args, rng, run_start, inflight, results)


async def run_closed_loop(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    results: list,
    run_start: float,
    inflight: Gauge,
    levels: list[int],
) -> None:
    deadline = run_start + args.duration
    step_s = args.duration / len(levels)
    await asyncio.gather(
        *(
            closed_loop_worker(
                client, args, deadline, results, args.seed + i, run_start,
                inflight, i, levels, step_s,
            )
            for i in range(max(levels))
        )
    )


async def run_poisson(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    results: list,
    run_start: float,
    inflight: Gauge,
) -> None:
    arrivals = random.Random(args.seed)
    prompts = random.Random(args.seed + 1)
    deadline = run_start + args.duration
    tasks: list[asyncio.Task] = []
    next_at = run_start
    while next_at < deadline:
        await asyncio.sleep(max(0.0, next_at - time.perf_counter()))
        tasks.append(
            asyncio.create_task(fire(client, args, prompts, run_start, inflight, results))
        )
        next_at += arrivals.expovariate(args.rate)
    if tasks:
        await asyncio.gather(*tasks)


async def run_burst(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    results: list,
    run_start: float,
    inflight: Gauge,
) -> None:
    prompts = random.Random(args.seed)
    deadline = run_start + args.duration
    tasks: list[asyncio.Task] = []
    next_at = run_start
    while next_at < deadline:
        await asyncio.sleep(max(0.0, next_at - time.perf_counter()))
        for _ in range(args.burst_size):
            tasks.append(
                asyncio.create_task(fire(client, args, prompts, run_start, inflight, results))
            )
        next_at += args.burst_interval
    if tasks:
        await asyncio.gather(*tasks)


def parse_stairs(args: argparse.Namespace) -> list[int]:
    if args.stairs:
        levels = [int(s) for s in args.stairs.split(",") if s.strip()]
        if not levels or any(lv <= 0 for lv in levels):
            raise SystemExit("--stairs must be positive comma-separated levels")
        return levels
    # Doubling ramp up to --concurrency: 16 -> [1, 2, 4, 8, 16].
    levels = []
    level = 1
    while level < args.concurrency:
        levels.append(level)
        level *= 2
    levels.append(args.concurrency)
    return levels


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument(
        "--pattern",
        choices=["constant", "poisson", "staircase", "burst"],
        default="constant",
    )
    p.add_argument("--concurrency", type=int, default=1,
                   help="closed-loop worker count (constant; staircase ramp ceiling)")
    p.add_argument("--rate", type=float, default=1.0,
                   help="poisson: mean arrival rate in req/s")
    p.add_argument("--stairs",
                   help="staircase: comma-separated concurrency levels, e.g. 1,2,4,8")
    p.add_argument("--burst-size", type=int, default=8,
                   help="burst: simultaneous requests per burst")
    p.add_argument("--burst-interval", type=float, default=10.0,
                   help="burst: seconds between bursts")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=0,
                   help="seeds the arrival schedule and prompt synthesis")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--trace-out",
        help="also append records in the inference-sim trace JSONL schema "
        "(client-side measurements; writes a meta line if the file is new)",
    )
    args = p.parse_args()

    if args.pattern == "poisson" and args.rate <= 0:
        raise SystemExit("--rate must be > 0 for poisson")
    if args.pattern == "burst" and (args.burst_size <= 0 or args.burst_interval <= 0):
        raise SystemExit("--burst-size and --burst-interval must be > 0 for burst")

    results: list[dict] = []
    inflight = Gauge()
    run_start = time.perf_counter()

    # Open-loop patterns must not let the connection pool throttle arrivals.
    if args.pattern in ("constant", "staircase"):
        levels = [args.concurrency] if args.pattern == "constant" else parse_stairs(args)
        limits = httpx.Limits(max_connections=max(levels) + 4)
    else:
        levels = []
        limits = httpx.Limits(max_connections=None)

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0), limits=limits) as client:
        if args.pattern in ("constant", "staircase"):
            await run_closed_loop(client, args, results, run_start, inflight, levels)
        elif args.pattern == "poisson":
            await run_poisson(client, args, results, run_start, inflight)
        else:
            await run_burst(client, args, results, run_start, inflight)

    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": ok, "errors": errs}, f, indent=1)

    if args.trace_out:
        new_file = not os.path.exists(args.trace_out) or os.path.getsize(args.trace_out) == 0
        with open(args.trace_out, "a") as f:
            if new_file:
                meta = {"model": args.model, "source": "loadgen-client", "pattern": args.pattern}
                if args.pattern == "poisson":
                    meta["rate"] = args.rate
                elif args.pattern == "staircase":
                    meta["stairs"] = levels
                elif args.pattern == "burst":
                    meta["burst_size"] = args.burst_size
                    meta["burst_interval"] = args.burst_interval
                f.write(json.dumps({"meta": meta}) + "\n")
            for r in ok:
                rec = {
                    "prompt_tokens": r["prompt_tokens"],
                    "cached_tokens": 0,
                    "output_tokens": r["output_tokens"],
                    "ttft_ms": r["ttft_ms"],
                    "itl_ms": r["itl_ms"],
                    "concurrency": r["concurrency"],
                    # Relative to this invocation's start; appended runs restart at 0.
                    "arrival_ms": r["arrival_ms"],
                }
                f.write(json.dumps(rec) + "\n")
    itls = sorted(r["itl_mean_ms"] for r in ok if r["itl_mean_ms"] is not None)
    ttfts = sorted(r["ttft_ms"] for r in ok)

    def pct(v: list[float], q: float) -> float:
        return v[min(int(q * len(v)), len(v) - 1)] if v else 0.0

    print(
        f"done: {len(ok)} ok, {len(errs)} errors ({len(ok) / args.duration:.2f} req/s) | "
        f"ttft p50/p99 {pct(ttfts, 0.5):.0f}/{pct(ttfts, 0.99):.0f} ms | "
        f"itl-mean p50/p99 {pct(itls, 0.5):.1f}/{pct(itls, 0.99):.1f} ms"
    )


if __name__ == "__main__":
    asyncio.run(main())
