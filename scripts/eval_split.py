from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from app.slm_task_client import SLMTaskClient
from app.models import ALLOWED_TOOLS


@dataclass
class EvalRow:
    prompt: str
    expected_split: bool


def load_dataset(path: Path) -> List[EvalRow]:
    rows: List[EvalRow] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rows.append(
            EvalRow(
                prompt=str(item["prompt"]),
                expected_split=bool(item["expected_split"]),
            )
        )
    return rows


async def run_eval(client: SLMTaskClient, rows: List[EvalRow], session_prefix: str) -> Dict[str, Any]:
    total = len(rows)
    correct_split = 0
    expected_true = 0
    true_positive = 0
    false_positive = 0
    json_valid = 0
    tool_valid_rows = 0

    for i, row in enumerate(rows):
        if row.expected_split:
            expected_true += 1
        result = await client.analyze(row.prompt, "{0}-{1}".format(session_prefix, i))
        if result:
            json_valid += 1

        pred_split = bool(result.split)
        if pred_split == row.expected_split:
            correct_split += 1
        if row.expected_split and pred_split:
            true_positive += 1
        if (not row.expected_split) and pred_split:
            false_positive += 1

        tools_ok = all(p.tool in ALLOWED_TOOLS for p in result.prompts)
        if tools_ok:
            tool_valid_rows += 1

    split_accuracy = correct_split / total if total else 0.0
    split_recall = true_positive / expected_true if expected_true else 0.0
    non_split_total = total - expected_true
    false_positive_rate = false_positive / non_split_total if non_split_total else 0.0
    json_valid_rate = json_valid / total if total else 0.0
    tool_valid_rate = tool_valid_rows / total if total else 0.0

    return {
        "total": total,
        "split_accuracy": split_accuracy,
        "split_recall": split_recall,
        "split_false_positive_rate": false_positive_rate,
        "json_valid_rate": json_valid_rate,
        "tool_valid_rate": tool_valid_rate,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen split behavior")
    parser.add_argument("--dataset", required=True, help="Path to JSONL eval dataset")
    parser.add_argument("--api-url", required=True, help="Chat-completions endpoint URL")
    parser.add_argument("--model", default="Qwen3.5-0.8B")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--session-prefix", default="eval")
    args = parser.parse_args()

    rows = load_dataset(Path(args.dataset))
    client = SLMTaskClient(
        api_url=args.api_url,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )
    metrics = await run_eval(client=client, rows=rows, session_prefix=args.session_prefix)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
