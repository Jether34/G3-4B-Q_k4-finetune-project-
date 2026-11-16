"""Full-weight defensive SFT for SmolLM2-135M-Instruct with live telemetry."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


class ChatDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.rows = []
        raw = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        for index, row in enumerate(raw, start=1):
            messages = row["messages"]
            prompt_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
            full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            prefix = min(len(prompt_ids), len(full_ids) - 1)
            if len(full_ids) > max_length:
                # Preserve the full answer where possible; trim oldest prompt tokens first.
                answer_len = max(1, len(full_ids) - prefix)
                keep_prompt = max(1, max_length - answer_len)
                if answer_len >= max_length:
                    full_ids = full_ids[-max_length:]
                    prefix = 0
                else:
                    full_ids = full_ids[-(keep_prompt + answer_len):]
                    prefix = keep_prompt
            if len(full_ids) >= 2:
                labels = list(full_ids)
                labels[:prefix] = [-100] * prefix
                self.rows.append((full_ids, labels, row.get("source", "")))
            if index == 1 or index % 1000 == 0 or index == len(raw):
                print(f"tokenize_progress examples={index}/{len(raw)} retained={len(self.rows)}", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(rows, pad_id: int):
    width = max(len(row[0]) for row in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    tokens = 0
    for i, (x, y, _) in enumerate(rows):
        ids[i, :len(x)] = torch.tensor(x)
        labels[i, :len(y)] = torch.tensor(y)
        mask[i, :len(x)] = 1
        tokens += sum(value != -100 for value in y)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels, "answer_tokens": tokens}


@torch.inference_mode()
def evaluate(model, loader, device, max_batches: int):
    model.eval()
    loss_sum = 0.0
    token_count = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        count = batch.pop("answer_tokens")
        batch = {key: value.to(device) for key, value in batch.items()}
        result = model(**batch)
        loss_sum += float(result.loss) * count
        token_count += count
    model.train()
    loss = loss_sum / max(1, token_count)
    return loss, math.exp(min(loss, 40.0)), token_count


def save_model(model, tokenizer, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-eval-batches", type=int, default=128)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("No Kaggle GPU detected. Select an available GPU accelerator first.")
    if args.batch_size < 1 or args.grad_accum < 1 or args.eval_interval < 1:
        parser.error("batch-size, grad-accum, and eval-interval must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    print(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} model={args.model_id}", flush=True)
    print("model_load_start", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Keep master parameters and optimizer states in FP32. Autocast handles
    # activation compute in FP16 on Kaggle T4 while GradScaler protects updates.
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to(device)
    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"model_load_done parameters={params:,} trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)

    train_set = ChatDataset(args.train, tokenizer, args.max_length)
    val_set = ChatDataset(args.validation, tokenizer, args.max_length)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda rows: collate(rows, tokenizer.pad_token_id), num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda rows: collate(rows, tokenizer.pad_token_id), num_workers=0)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    target_steps = min(args.max_steps, max(1, math.ceil(updates_per_epoch * args.epochs)))
    print(f"data_ready train={len(train_set)} validation={len(val_set)} updates_per_epoch={updates_per_epoch} target_steps={target_steps}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(50, int(target_steps * 0.03)), target_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.csv"
    best_loss = float("inf")
    no_improve = 0
    completed = 0
    started = time.monotonic()
    window_start = started
    window_tokens = window_loss_tokens = 0
    window_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    metrics_stream = metrics_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(metrics_stream)
    writer.writerow(["step", "epoch", "train_loss", "val_loss", "val_perplexity", "tokens_per_sec", "steps_per_sec", "elapsed_sec", "eta_min"])
    try:
        while completed < target_steps:
            epoch_index = completed / max(1, updates_per_epoch)
            for batch_index, batch in enumerate(train_loader):
                if completed >= target_steps:
                    break
                answer_tokens = batch.pop("answer_tokens")
                batch = {key: value.to(device) for key, value in batch.items()}
                group_start = (batch_index // args.grad_accum) * args.grad_accum
                group_size = min(args.grad_accum, len(train_loader) - group_start)
                with torch.autocast("cuda", dtype=torch.float16):
                    raw_loss = model(**batch).loss
                    loss = raw_loss / group_size
                scaler.scale(loss).backward()
                window_loss += float(raw_loss.detach()) * answer_tokens
                window_loss_tokens += answer_tokens
                window_tokens += answer_tokens
                if (batch_index + 1) % args.grad_accum and (batch_index + 1) != len(train_loader):
                    continue
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                completed += 1
                now = time.monotonic()
                elapsed = now - started
                span = max(0.001, now - window_start)
                rate = window_tokens / span
                step_rate = 1.0 / span
                current_epoch = min(args.epochs, completed / max(1, updates_per_epoch))
                remaining = max(0, target_steps - completed)
                eta = (elapsed / max(1, completed) * remaining) / 60
                recent_loss = window_loss / max(1, window_loss_tokens)
                val_loss = val_ppl = float("nan")
                if completed % args.eval_interval == 0 or completed == target_steps:
                    val_loss, val_ppl, val_tokens = evaluate(model, val_loader, device, args.max_eval_batches)
                    print(f"validation step={completed} epoch={current_epoch:.3f} val_loss={val_loss:.4f} val_ppl={val_ppl:.3f} answer_tokens={val_tokens}", flush=True)
                    if val_loss < best_loss:
                        best_loss, no_improve = val_loss, 0
                        save_model(model, tokenizer, args.out_dir / "best_model")
                        (args.out_dir / "best_metrics.json").write_text(json.dumps({"step": completed, "epoch": current_epoch, "val_loss": best_loss, "val_perplexity": val_ppl}, indent=2), encoding="utf-8")
                    else:
                        no_improve += 1
                    if no_improve >= args.patience:
                        print(f"early_stop step={completed} non_improving_evaluations={no_improve}", flush=True)
                writer.writerow([completed, f"{current_epoch:.5f}", f"{recent_loss:.6f}", val_loss, val_ppl, f"{rate:.1f}", f"{step_rate:.3f}", f"{elapsed:.1f}", f"{eta:.1f}"])
                metrics_stream.flush()
                print(f"train_progress step={completed}/{target_steps} epoch={current_epoch:.3f} train_loss={recent_loss:.4f} tokens_per_sec={rate:.1f} steps_per_sec={step_rate:.3f} elapsed_min={elapsed/60:.1f} eta_min={eta:.1f}", flush=True)
                window_start, window_tokens, window_loss, window_loss_tokens = now, 0, 0.0, 0
                if no_improve >= args.patience:
                    break
            if no_improve >= args.patience:
                break
    finally:
        metrics_stream.close()
    if not (args.out_dir / "best_model" / "config.json").exists():
        save_model(model, tokenizer, args.out_dir / "best_model")
    summary = {"model_id": args.model_id, "parameters": params, "steps": completed, "target_steps": target_steps,
               "steps_per_epoch": updates_per_epoch, "data_epochs_seen": min(args.epochs, completed / max(1, updates_per_epoch)),
               "best_validation_loss": best_loss, "metric": "assistant-token cross-entropy/perplexity; MAE is not applicable"}
    (args.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
