"""Is qwen3_coder's streaming parse cost quadratic in accumulated output?

Simulates an agent writing a progressively larger file via one Write tool
call, feeding the parser 1 token per call (the well-behaved delta layout).
If per-delta cost grows with accumulated length, total work is O(n^2), and
once per-delta cost exceeds the engine's real inter-token latency (~30ms on
our H200 capture), the frontend consumer falls behind at real GPU speed:
deltas merge, and the parser enters its corrupt/drop regimes.
"""

import time

from transformers import AutoTokenizer

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers import ToolParserManager

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

REQUEST = ChatCompletionRequest(
    model=MODEL,
    messages=[{"role": "user", "content": "write the file"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "Write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        }
    ],
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
parser_cls = ToolParserManager.get_tool_parser("qwen3_coder")

CODE_LINE = "    result = compute_value(input_data, options=defaults)  # step\n"

print(f"{'file_size':>10} {'tokens':>7} {'total_s':>8} {'ms/delta p50':>13} {'ms/delta max':>13}")
for lines in (50, 200, 800, 1600):
    content = "".join(f"{CODE_LINE[:-1]} {i}\n" for i in range(lines))
    text = (
        "Writing the file now:\n<tool_call>\n<function=Write>\n"
        "<parameter=file_path>\n/tmp/big.py\n</parameter>\n"
        f"<parameter=content>\n{content}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    parser = parser_cls(tokenizer)
    prev_ids: list[int] = []
    prev = ""
    per_delta = []
    t_start = time.perf_counter()
    for i, tok in enumerate(ids):
        # Incremental decode keeps the harness O(n); only the parser is timed.
        delta = tokenizer.decode([tok])
        cur = prev + delta
        cur_ids = ids[: i + 1]
        t0 = time.perf_counter()
        parser.extract_tool_calls_streaming(
            previous_text=prev,
            current_text=cur,
            delta_text=delta,
            previous_token_ids=prev_ids,
            current_token_ids=cur_ids,
            delta_token_ids=[tok],
            request=REQUEST,
        )
        per_delta.append(time.perf_counter() - t0)
        prev, prev_ids = cur, cur_ids
    total = time.perf_counter() - t_start
    per_delta.sort()
    p50 = per_delta[len(per_delta) // 2] * 1000
    mx = per_delta[-1] * 1000
    print(f"{len(content):>10} {len(ids):>7} {total:>8.2f} {p50:>13.2f} {mx:>13.2f}")
