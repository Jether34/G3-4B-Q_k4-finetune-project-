"""Compare base/adapted answer-token loss and sample generations on protected QAA."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM = "You are a defensive cybersecurity assistant. Your name is Jether. Give concise, accurate, authorized defensive guidance and state uncertainty."


def load_pairs(path: Path):
    rows = []
    for line in path.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        question, answer = str(row.get("question", "")).strip(), str(row.get("answer", "")).strip()
        if question and answer:
            rows.append((question, answer, row))
    if not rows:
        raise RuntimeError(f"No usable evaluation questions in {path}")
    return rows


def load_model(model_id: str, device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    return model, tokenizer


@torch.inference_mode()
def score_pair(model, tokenizer, device, question: str, answer: str, max_length: int):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": answer}],
                                          tokenize=True, add_generation_prompt=False)
    prefix_len = min(len(prefix), len(full) - 1)
    if len(full) > max_length:
        target_len = len(full) - prefix_len
        if target_len >= max_length:
            full = full[-max_length:]
            prefix_len = 0
        else:
            keep_prefix = max_length - target_len
            full = full[-max_length:]
            prefix_len = keep_prefix
    ids = torch.tensor([full], dtype=torch.long, device=device)
    labels = ids.clone()
    labels[:, :prefix_len] = -100
    logits = model(input_ids=ids).logits[:, :-1].float()
    targets = labels[:, 1:]
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                             ignore_index=-100, reduction="sum")
    count = int((targets != -100).sum())
    return float(losses), count


@torch.inference_mode()
def generate(model, tokenizer, device, question: str, max_new_tokens: int):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_tensors="pt").to(device)
    output = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False,
                            repetition_penalty=1.08, pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, tokens.shape[1]:], skip_special_tokens=True).strip()


def evaluate(model_id: str, label: str, pairs, device, sample_count: int, max_length: int):
    model, tokenizer = load_model(model_id, device)
    total_loss = total_tokens = 0
    for index, (question, answer, _) in enumerate(pairs, 1):
        loss, count = score_pair(model, tokenizer, device, question, answer, max_length)
        total_loss += loss
        total_tokens += count
        if index % 10 == 0 or index == len(pairs):
            print(f"holdout_progress model={label} examples={index}/{len(pairs)}", flush=True)
    answer_loss = total_loss / max(1, total_tokens)
    result = {"model": label, "examples": len(pairs), "answer_tokens": total_tokens,
              "answer_loss": answer_loss, "answer_perplexity": math.exp(min(answer_loss, 40))}
    sample_rows = []
    for index, (question, answer, row) in enumerate(pairs[:sample_count]):
        sample_rows.append({"model": label, "id": str(row.get("id", index)), "theme": str(row.get("theme", "")),
                            "question": question, "reference_answer": answer,
                            "generated_answer": generate(model, tokenizer, device, question, 128)})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, sample_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=768)
    args = parser.parse_args()
    pairs = load_pairs(args.eval_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_result, base_rows = evaluate(args.base, "base", pairs, device, args.sample_count, args.max_length)
    tuned_result, tuned_rows = evaluate(str(args.adapted), "adapted", pairs, device, args.sample_count, args.max_length)
    results = [base_result, tuned_result]
    (args.output_dir / "holdout_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (args.output_dir / "holdout_generations.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "id", "theme", "question", "reference_answer", "generated_answer"])
        writer.writeheader()
        writer.writerows(base_rows + tuned_rows)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
