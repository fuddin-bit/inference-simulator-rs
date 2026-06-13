"""Repro: qwen3_coder streaming tool parser, burst vs paced delivery.

The same response text is fed through extract_tool_calls_streaming twice:
- paced: one call per small delta (as when tokens trickle in at real ITL)
- burst: one call with the entire text as a single delta (as when a replay
  engine emits every token in one scheduler iteration)

A correct streaming parser yields the same tool calls either way.
"""

import json

from transformers import AutoTokenizer

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers import ToolParserManager

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

TEXT = (
    "I'll create the file now:\n"
    "<tool_call>\n<function=Write>\n"
    "<parameter=file_path>\n/tmp/demo/greet.py\n</parameter>\n"
    "<parameter=content>\ndef greet(name):\n    return f'hello, {name}'\n</parameter>\n"
    "</function>\n</tool_call>"
)

REQUEST = ChatCompletionRequest(
    model=MODEL,
    messages=[{"role": "user", "content": "make the file"}],
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


def run(chunk_tokens: int, label: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    parser_cls = ToolParserManager.get_tool_parser("qwen3_coder")
    parser = parser_cls(tokenizer)
    ids = tokenizer.encode(TEXT, add_special_tokens=False)
    tool_names, args_chunks, text_out = [], [], []
    prev_ids: list[int] = []
    for i in range(0, len(ids), chunk_tokens):
        delta_ids = ids[i : i + chunk_tokens]
        cur_ids = ids[: i + len(delta_ids)]
        prev = tokenizer.decode(prev_ids)
        cur = tokenizer.decode(cur_ids)
        result = parser.extract_tool_calls_streaming(
            previous_text=prev,
            current_text=cur,
            delta_text=cur[len(prev):],
            previous_token_ids=prev_ids,
            current_token_ids=cur_ids,
            delta_token_ids=delta_ids,
            request=REQUEST,
        )
        prev_ids = cur_ids
        if result is None:
            continue
        if result.content:
            text_out.append(result.content)
        for tc in result.tool_calls or []:
            if tc.function and tc.function.name:
                tool_names.append(tc.function.name)
            if tc.function and tc.function.arguments:
                args_chunks.append(tc.function.arguments)
    args = "".join(args_chunks)
    print(f"{label}: tool_names={tool_names} args_complete={bool(args)}")
    if args:
        try:
            print(f"  parsed args keys: {sorted(json.loads(args))}")
        except json.JSONDecodeError as e:
            print(f"  args not valid JSON: {e}: {args[:80]}...")


# Chunk-size sweep: a correct streaming parser is invariant to delivery
# granularity. Real-time generation is ~1 token/call; an instant replay
# engine can deliver everything in one call.
for chunk in (1, 2, 4, 8, 16, 32, 10**9):
    label = f"chunk={chunk if chunk < 10**9 else 'ALL'}"
    run(chunk, f"{label:12}")
