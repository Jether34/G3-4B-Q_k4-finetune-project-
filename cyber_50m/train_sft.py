"""Answer-masked SFT for a from-scratch defensive cybersecurity Transformer."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
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


def make_chunks(examples, tokenizer: Tokenizer, block: int, label: str):
    eos = tokenizer.token_to_id("[EOS]")
    chunks: list[tuple[list[int], int]] = []
    for example_index, row in enumerate(examples, start=1):
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
        if example_index == 1 or example_index % 5000 == 0 or example_index == len(examples):
            print(f"{label}_chunking_progress examples={example_index}/{len(examples)} chunks={len(chunks)}", flush=True)
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


class ChunkSampler:
    """Visit answer chunks in shuffled, non-replacement passes."""
    def __init__(self, size: int, seed: int):
        self.size = size
        self.rng = random.Random(seed)
        self.order = list(range(size))
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.samples_seen = 0

    def next_indices(self, count: int) -> list[int]:
        result = []
        while len(result) < count:
            if self.cursor >= self.size:
                self.rng.shuffle(self.order)
                self.cursor = 0
            take = min(count - len(result), self.size - self.cursor)
            result.extend(self.order[self.cursor:self.cursor + take])
            self.cursor += take
            self.samples_seen += take
        return result

    @property
    def epochs_seen(self) -> float:
        return self.samples_seen / max(1, self.size)


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
    parser.add_argument("--steps", type=int, default=None, help="Optional explicit override; otherwise derive from --epochs")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--patience-evals", type=int, default=8)
    parser.add_argument("--max-eval-chunks", type=int, default=1024)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    if args.epochs <= 0 or args.max_steps < 1 or args.eval_interval < 1 or args.log_interval < 1:
        parser.error("epochs, max-steps, eval-interval, and log-interval must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This SFT pass requires a Kaggle GPU")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    print("stage=load_pretrain_checkpoint_start", flush=True)
    artifact = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = artifact["config"]
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    print("stage=load_sft_examples_start", flush=True)
    examples = read_examples(args.data_dir)
    print(f"stage=load_sft_examples_done examples={len(examples)}", flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    val_count = max(1, round(len(examples) * 0.1))
    val_examples, train_examples = examples[:val_count], examples[val_count:]
    block = int(config["max_sequence_length"])
    train_chunks = make_chunks(train_examples, tokenizer, block, "sft_train")
    val_chunks = make_chunks(val_examples, tokenizer, block, "sft_validation")
    if len(train_chunks) < args.batch_size * args.grad_accum:
        raise RuntimeError("SFT corpus is too small for one effective batch; refusing to silently oversample it")
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
    sampler = ChunkSampler(len(train_chunks), args.seed)
    steps_per_epoch = math.ceil(len(train_chunks) / (args.batch_size * args.grad_accum))
    requested_steps = math.ceil(steps_per_epoch * args.epochs) if args.steps is None else args.steps
    total_steps = min(requested_steps, args.max_steps)
    print(
        f"sft_ready examples={len(examples)} train_examples={len(train_examples)} "
        f"train_chunks={len(train_chunks)} validation_chunks={len(val_chunks)} "
        f"steps_per_epoch={steps_per_epoch} epochs_requested={args.epochs:.2f} "
        f"steps={total_steps} effective_batch={args.batch_size * args.grad_accum}", flush=True,
    )
    started_at = time.monotonic()
    model.train()
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        log = csv.writer(stream)
        log.writerow(["step", "train_answer_loss", "val_answer_loss", "val_answer_perplexity"])
        for step in range(1, total_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            for _ in range(args.grad_accum):
                picks = sampler.next_indices(args.batch_size)
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
            if step % args.log_interval == 0:
                elapsed = time.monotonic() - started_at
                rate = step / max(elapsed, 1e-6)
                remaining = max(0, total_steps - step) / max(rate, 1e-6)
                print(
                    f"sft_progress step={step}/{total_steps} data_epochs={sampler.epochs_seen:.3f} "
                    f"train_answer_loss={running_loss:.4f} steps_per_sec={rate:.3f} eta_min={remaining / 60:.1f}",
                    flush=True,
                )
            if step == 1 or step % args.eval_interval == 0 or step == total_steps:
                val_loss = evaluate(model, val_chunks, args.batch_size, pad_id, device, args.max_eval_chunks)
                val_ppl = math.exp(val_loss) if val_loss < 50 else float("inf")
                log.writerow([step, running_loss, val_loss, val_ppl])
                stream.flush()
                print(f"sft_eval step={step} train_answer_loss={running_loss:.4f} val_answer_loss={val_loss:.4f} val_answer_ppl={val_ppl:.2f}", flush=True)
                if val_loss < best_val:
                    best_val = val_loss
                    no_improvement = 0
                    torch.save({"model": model.state_dict(), "config": config,
                                "vocab_size": artifact["vocab_size"]}, args.out_dir / "best_sft_checkpoint.pt")
                else:
                    no_improvement += 1
                    if args.patience_evals > 0 and no_improvement >= args.patience_evals:
                        print(f"sft_early_stop step={step} no_improvement_evals={no_improvement}", flush=True)
                        break
    summary = {
        "examples": len(examples), "train_examples": len(train_examples), "validation_examples": len(val_examples),
        "train_answer_chunks": len(train_chunks), "validation_answer_chunks": len(val_chunks),
        "validation_answer_chunks_evaluated": min(len(val_chunks), args.max_eval_chunks),
        "steps": completed_steps, "best_validation_answer_loss": best_val,
        "steps_per_epoch": steps_per_epoch, "data_epochs_seen": sampler.epochs_seen,
        "training_chunks_seen": sampler.samples_seen,
        "maximum_steps_configured": args.max_steps,
        "scope": "defensive cybersecurity task adaptation; evaluate on separate expert-reviewed QA",
    }
    (args.out_dir / "sft_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
