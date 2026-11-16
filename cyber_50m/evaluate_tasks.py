"""Generate and score answers on held-out, expert-reviewed defensive QA."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from train_gpt import SmallGPT


STOP = {"about", "after", "also", "and", "are", "for", "from", "into", "that", "the", "their", "then", "this", "with", "your"}


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{3,}", text.casefold()) if w not in STOP}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model = SmallGPT(int(artifact["vocab_size"]), artifact["config"])
    model.load_state_dict(artifact["model"])
    model.to(device).eval()
    block = model.block
    eos = tokenizer.token_to_id("[EOS]")
    rows = [json.loads(line) for line in args.eval_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for example_index, row in enumerate(rows, start=1):
        prompt = f"Question: {row['question']}\nAnswer:"
        prompt_ids = tokenizer.encode(prompt).ids
        reference_ids = tokenizer.encode(" " + row["answer"]).ids
        if len(prompt_ids) >= block:
            prompt_ids = prompt_ids[-(block - 1):]
        target_ids = reference_ids[:max(1, block - len(prompt_ids))]
        combined = prompt_ids + target_ids
        ids = torch.tensor(combined, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(ids[:, :-1]).float()[0]
        start = len(prompt_ids) - 1
        answer_loss = F.cross_entropy(
            logits[start:start + len(target_ids)],
            ids[0, len(prompt_ids):len(prompt_ids) + len(target_ids)],
            reduction="mean",
        ).item()

        generated = list(prompt_ids)
        for _ in range(args.max_new_tokens):
            context = torch.tensor(generated[-block:], dtype=torch.long, device=device).unsqueeze(0)
            next_id = int(model(context)[0, -1].argmax().item())
            if next_id == eos:
                break
            generated.append(next_id)
        completion = tokenizer.decode(generated[len(prompt_ids):], skip_special_tokens=True).strip()
        answer_words = words(completion)
        assertions = [a for a in row.get("assertions", []) if words(a)]
        covered = sum(bool(words(a) & answer_words) for a in assertions)
        coverage = covered / len(assertions) if assertions else 0.0
        results.append({
            "id": row.get("id", ""),
            "theme": row.get("theme", "Unspecified"),
            "question_type": row.get("question_type", "Unspecified"),
            "question": row["question"],
            "reference_answer": row["answer"],
            "generated_answer": completion,
            "answer_token_loss": answer_loss,
            "assertion_coverage_proxy": coverage,
            "assertions_covered": covered,
            "assertions_total": len(assertions),
        })
        if example_index == 1 or example_index % 10 == 0 or example_index == len(rows):
            print(f"task_eval_progress examples={example_index}/{len(rows)}", flush=True)

    with (args.output_dir / "task_generations.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]) if results else [])
        writer.writeheader()
        writer.writerows(results)
    by_theme = defaultdict(list)
    for row in results:
        by_theme[row["theme"]].append(row)
    losses = [r["answer_token_loss"] for r in results]
    summary = {
        "examples": len(results),
        "mean_heldout_answer_token_loss": sum(losses) / max(1, len(losses)),
        "heldout_answer_perplexity": math.exp(min(50, sum(losses) / max(1, len(losses)))),
        "mean_assertion_coverage_proxy": sum(r["assertion_coverage_proxy"] for r in results) / max(1, len(results)),
        "by_theme": {
            name: {
                "examples": len(group),
                "mean_answer_loss": sum(r["answer_token_loss"] for r in group) / len(group),
                "mean_assertion_coverage_proxy": sum(r["assertion_coverage_proxy"] for r in group) / len(group),
            }
            for name, group in sorted(by_theme.items())
        },
        "limitations": "Assertion coverage is a lexical proxy, not correctness. Review task_generations.csv manually; this small held-out set is not deployment validation.",
    }
    (args.output_dir / "task_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
