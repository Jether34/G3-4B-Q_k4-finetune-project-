"""Answer-masked SFT for a from-scratch defensive cybersecurity Transformer."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from tokenizers import Tokenizer

from train_gpt import SmallGPT


def read_examples(data_dir: Path) -> list[dict[str, str]]:
    examples = []
    for filename in (
        "cybersecurity_qa_train.jsonl",
        "holycyber_defensive_train.jsonl",
        "fenrir_defensive_sft.jsonl",
        "threat_intel_defensive_sft.jsonl",
        "trendyol_defensive_sft.jsonl",
    ):
        path = data_dir / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                question = str(row.get("question", "")).strip()
                prompt = str(row.get("prompt", "")).strip()
                answer = str(row.get("answer", "")).strip()
                if not prompt and question:
                    prompt = f"Question: {question}\nAnswer:"
                if len(prompt) >= 20 and len(answer) >= 80:
                    examples.append({
                        "id": str(row.get("id", "")),
                        "domain": str(row.get("theme", row.get("domain", "Unspecified"))),
                        "prompt": prompt,
                        "answer": answer,
                    })
    if not examples:
        raise RuntimeError(f"No prompt/answer examples found in {data_dir}")
    unique = {}
    for row in examples:
        unique.setdefault((row["prompt"].casefold(), row["answer"].casefold()), row)
    return list(unique.values())


def make_chunks(examples, tokenizer: Tokenizer, block: int):
    eos = tokenizer.token_to_id("[EOS]")
    chunks: list[tuple[list[int], int]] = []
    for row in examples:
        prefix = tokenizer.encode(row["prompt"]).ids
        answer = tokenizer.encode(" " + row["answer"]).ids + [eos]
        if not prefix or not answer:
            continue
        prefix = prefix[-max(1, block // 2):]
        offset = 0
        while offset < len(answer):
            if offset == 0:
                context = prefix
            else:
                context = (prefix + answer[:offset])[-max(1, block // 2):]
            budget = block + 1 - len(context)
            if budget < 1:
                context = context[-(block - 1):]
                budget = block + 1 - len(context)
            target = answer[offset:offset + budget]
            sequence = context + target
            if len(sequence) >= 2:
                chunks.append((sequence, len(context)))
            offset += len(target)
    if len(chunks) < 8:
        raise RuntimeError("The QA corpus produced too few answer chunks")
    return chunks


def collate(chunks, indices, pad_id: int, device):
    selected = [chunks[i] for i in indices]
    width = max(len(ids) for ids, _ in selected) - 1
    x = torch.full((len(selected), width), pad_id, dtype=torch.long, device=device)
    y = torch.full((len(selected), width), -100, dtype=torch.long, device=device)
    for i, (ids, prefix_len) in enumerate(selected):
        inputs = ids[:-1]
        labels = ids[1:]
        x[i, :len(inputs)] = torch.tensor(inputs, dtype=torch.long, device=device)
        y[i, :len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)
        y[i, :max(0, prefix_len - 1)] = -100
    return x, y


@torch.no_grad()
def evaluate(model, chunks, batch_size: int, pad_id: int, device, max_chunks: int) -> float:
    model.eval()
    # Bounded deterministic sample keeps full validation from dominating Kaggle time.
    chunks = chunks[:max_chunks]
    total_loss, total_tokens = 0.0, 0
    for start in range(0, len(chunks), batch_size):
        x, y = collate(chunks, range(start, min(start + batch_size, len(chunks))), pad_id, device)
        logits = model(x).float()
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                           ignore_index=-100, reduction="sum")
        total_loss += loss.item()
        total_tokens += int((y != -100).sum().item())
    model.train()
    return total_loss / max(1, total_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--patience-evals", type=int, default=8)
    parser.add_argument("--max-eval-chunks", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This SFT pass requires a Kaggle GPU")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    artifact = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = artifact["config"]
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    examples = read_examples(args.data_dir)
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    val_count = max(1, round(len(examples) * 0.1))
    val_examples, train_examples = examples[:val_count], examples[val_count:]
    block = int(config["max_sequence_length"])
    train_chunks = make_chunks(train_examples, tokenizer, block)
    val_chunks = make_chunks(val_examples, tokenizer, block)
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise RuntimeError("Tokenizer has no PAD token")
    model = SmallGPT(int(artifact["vocab_size"]), config).to(device)
    model.load_state_dict(artifact["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "sft_metrics.csv"
    best_val = float("inf")
    no_improvement = 0
    completed_steps = 0
    model.train()
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        log = csv.writer(stream)
        log.writerow(["step", "train_answer_loss", "val_answer_loss", "val_answer_perplexity"])
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            for _ in range(args.grad_accum):
                picks = random.choices(range(len(train_chunks)), k=args.batch_size)
                x, y = collate(train_chunks, picks, pad_id, device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                                       y.reshape(-1), ignore_index=-100)
                    loss = loss / args.grad_accum
                scaler.scale(loss).backward()
                running_loss += loss.item()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            completed_steps = step
            if step == 1 or step % args.eval_interval == 0 or step == args.steps:
                val_loss = evaluate(model, val_chunks, args.batch_size, pad_id, device, args.max_eval_chunks)
                val_ppl = math.exp(val_loss) if val_loss < 50 else float("inf")
                log.writerow([step, running_loss, val_loss, val_ppl])
                stream.flush()
                print(f"sft_step={step} train_answer_loss={running_loss:.4f} val_answer_loss={val_loss:.4f} val_answer_ppl={val_ppl:.2f}")
                if val_loss < best_val:
                    best_val = val_loss
                    no_improvement = 0
                    torch.save({"model": model.state_dict(), "config": config,
                                "vocab_size": artifact["vocab_size"]}, args.out_dir / "best_sft_checkpoint.pt")
                else:
                    no_improvement += 1
                    if args.patience_evals > 0 and no_improvement >= args.patience_evals:
                        print(f"SFT early stopping after {no_improvement} evaluations without improvement")
                        break
    summary = {
        "examples": len(examples), "train_examples": len(train_examples), "validation_examples": len(val_examples),
        "train_answer_chunks": len(train_chunks), "validation_answer_chunks": len(val_chunks),
        "validation_answer_chunks_evaluated": min(len(val_chunks), args.max_eval_chunks),
        "steps": completed_steps, "best_validation_answer_loss": best_val,
        "scope": "defensive cybersecurity task adaptation; evaluate on separate expert-reviewed QA",
    }
    (args.out_dir / "sft_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
