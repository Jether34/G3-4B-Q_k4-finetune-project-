"""Train a compact defensive decoder-only Transformer from scratch."""
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
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def read_records(data_dir: Path) -> list[dict[str, str]]:
    records = []
    for filename in (
        "official_defensive_text.jsonl",
        "incident_text.jsonl",
        "cybersecurity_qa_train.jsonl",
        "holycyber_defensive_train.jsonl",
        "fenrir_defensive_sft.jsonl",
        "threat_intel_defensive_sft.jsonl",
        "trendyol_defensive_sft.jsonl",
        "general_language_sample.jsonl",
        "copilot_style_private.jsonl",
    ):
        path = data_dir / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = " ".join(str(item.get("text", "")).split())
                if len(text) >= 80:
                    records.append({"text": text, "source": str(item.get("source", filename))})
    if not records:
        raise RuntimeError(f"No usable JSONL text found in {data_dir}")
    # Keep a little personal style data without letting short conversation turns dominate.
    defensive = [r for r in records if r["source"] != "copilot_style_private.jsonl"]
    style = [r for r in records if r["source"] == "copilot_style_private.jsonl"]
    if style and defensive:
        rng = random.Random(41)
        rng.shuffle(style)
        records = defensive + style[: max(1, min(len(style), len(defensive) // 20))]
    return records


def make_tokenizer(texts: list[str], vocab_size: int, output: Path) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    special = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    def progress_texts():
        for index, text in enumerate(texts, start=1):
            if index == 1 or index % 5000 == 0 or index == len(texts):
                print(f"tokenizer_input_progress records={index}/{len(texts)}", flush=True)
            yield text

    print(f"tokenizer_training_start records={len(texts)} vocab_target={vocab_size}", flush=True)
    tokenizer.train_from_iterator(progress_texts(), trainer=trainer, length=len(texts))
    print(f"tokenizer_training_complete vocab={tokenizer.get_vocab_size()}", flush=True)
    tokenizer.save(str(output))
    return tokenizer


def split_records(records: list[dict[str, str]], seed: int):
    defensive = [r for r in records if r["source"] != "copilot_style_private.jsonl"]
    style = [r for r in records if r["source"] == "copilot_style_private.jsonl"]
    rng = random.Random(seed)
    rng.shuffle(defensive)
    split = max(1, int(len(defensive) * 0.95))
    return defensive[:split] + style, defensive[split:] or defensive[-1:]


def chunk_records(records: list[dict[str, str]], tokenizer: Tokenizer, block: int, label: str):
    eos = tokenizer.token_to_id("[EOS]")
    encoded: list[int] = []
    for index, record in enumerate(records, start=1):
        encoded.extend(tokenizer.encode(record["text"]).ids)
        encoded.append(eos)
        if index == 1 or index % 5000 == 0 or index == len(records):
            print(f"{label}_chunking_progress records={index}/{len(records)} tokens={len(encoded)}", flush=True)
    chunks = [encoded[i:i + block + 1] for i in range(0, len(encoded) - block, block)]
    if len(chunks) < 8:
        raise RuntimeError("Corpus is too small to produce eight training chunks at this context length")
    return chunks


class SmallGPT(nn.Module):
    def __init__(self, vocab: int, config: dict):
        super().__init__()
        width = int(config["hidden_size"])
        self.block = int(config["max_sequence_length"])
        self.token = nn.Embedding(vocab, width)
        self.position = nn.Embedding(self.block, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=int(config["num_attention_heads"]),
            dim_feedforward=int(config["intermediate_size"]),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=int(config["num_layers"]), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)
        if config.get("tie_word_embeddings", True):
            self.head.weight = self.token.weight
        self._initialize(config)

    def _initialize(self, config: dict):
        """Use GPT-scale initialization rather than PyTorch's unit-scale embeddings."""
        layers = int(config["num_layers"])
        std = float(config.get("init_std", 0.02))
        nn.init.normal_(self.token.weight, mean=0.0, std=std)
        nn.init.normal_(self.position.weight, mean=0.0, std=std)
        for module in self.layers.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.normal_(module.in_proj_weight, mean=0.0, std=std)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
        residual_std = std / math.sqrt(2 * layers)
        for layer in self.layers.layers:
            nn.init.normal_(layer.self_attn.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.linear2.weight, mean=0.0, std=residual_std)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, ids):
        _, seq = ids.shape
        pos = torch.arange(seq, device=ids.device)
        x = self.token(ids) + self.position(pos)[None, :, :]
        mask = torch.triu(torch.full((seq, seq), float("-inf"), device=ids.device), diagonal=1)
        x = self.layers(x, mask=mask)
        return self.head(self.norm(x))


class ChunkSampler:
    """Visit corpus chunks in shuffled, non-replacement passes."""
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


def draw_batch(chunks, batch_size: int, block: int, device, sampler: ChunkSampler):
    picks = [chunks[index] for index in sampler.next_indices(batch_size)]
    batch = torch.tensor(picks, dtype=torch.long, device=device)
    return batch[:, :-1], batch[:, 1:]


@torch.no_grad()
def evaluate(model, chunks, batch_size, block, device, max_chunks: int):
    model.eval()
    # Keep validation cost stable as instruction corpora grow. The validation
    # record order is deterministically shuffled, so this prefix is a stable sample.
    chunks = chunks[:max_chunks]
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(chunks), batch_size):
        batch = torch.tensor(chunks[start:start + batch_size], dtype=torch.long, device=device)
        x, y = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train()
    return total_loss / max(1, total_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/derived"))
    parser.add_argument("--config", type=Path, default=Path("cyber_50m/model_config.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("cyber_50m/runs/first_pilot"))
    parser.add_argument("--steps", type=int, default=None, help="Optional explicit override; otherwise derive from --epochs")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--patience-evals", type=int, default=8)
    parser.add_argument("--max-eval-chunks", type=int, default=2048)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.epochs <= 0 or args.max_steps < 1 or args.eval_interval < 1 or args.log_interval < 1:
        parser.error("epochs, max-steps, eval-interval, and log-interval must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This pilot is intended to run on a Kaggle GPU accelerator")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("stage=load_records_start", flush=True)
    records = read_records(args.data_dir)
    print(f"stage=load_records_done records={len(records)}", flush=True)
    train_records, val_records = split_records(records, args.seed)
    print(f"stage=split_done train_records={len(train_records)} validation_records={len(val_records)}", flush=True)
    tokenizer_path = args.out_dir / "tokenizer.json"
    tokenizer = make_tokenizer([r["text"] for r in train_records], int(config["vocab_size"]), tokenizer_path)
    train_chunks = chunk_records(train_records, tokenizer, int(config["max_sequence_length"]), "train")
    val_chunks = chunk_records(val_records, tokenizer, int(config["max_sequence_length"]), "validation")
    actual_vocab_size = tokenizer.get_vocab_size()
    if actual_vocab_size < int(config["vocab_size"]) - 128:
        raise RuntimeError(f"Tokenizer vocabulary is only {actual_vocab_size}; refusing to train a materially undersized model")
    model = SmallGPT(tokenizer.get_vocab_size(), config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    log_path = args.out_dir / "metrics.csv"
    best_val = float("inf")
    evaluations_without_improvement = 0
    completed_steps = 0
    sampler = ChunkSampler(len(train_chunks), args.seed)
    steps_per_epoch = math.ceil(len(train_chunks) / (args.batch_size * args.grad_accum))
    requested_steps = math.ceil(steps_per_epoch * args.epochs) if args.steps is None else args.steps
    total_steps = min(requested_steps, args.max_steps)
    print(
        f"pretrain_ready records={len(train_records)} train_chunks={len(train_chunks)} "
        f"validation_chunks={len(val_chunks)} steps_per_epoch={steps_per_epoch} "
        f"epochs_requested={args.epochs:.2f} steps={total_steps} "
        f"effective_batch={args.batch_size * args.grad_accum} vocab={actual_vocab_size} "
        f"parameters={param_count}", flush=True,
    )
    started_at = time.monotonic()
    model.train()
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        log = csv.writer(stream)
        log.writerow(["step", "train_loss", "val_loss", "val_perplexity"])
        for step in range(1, total_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for _ in range(args.grad_accum):
                x, y = draw_batch(train_chunks, args.batch_size, model.block, device, sampler)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
                    loss = loss / args.grad_accum
                scaler.scale(loss).backward()
                total_loss += loss.item()
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
                    f"pretrain_progress step={step}/{total_steps} data_epochs={sampler.epochs_seen:.3f} "
                    f"train_loss={total_loss:.4f} steps_per_sec={rate:.3f} eta_min={remaining / 60:.1f}",
                    flush=True,
                )
            if step == 1 or step % args.eval_interval == 0 or step == total_steps:
                val_loss = evaluate(model, val_chunks, args.batch_size, model.block, device, args.max_eval_chunks)
                val_ppl = math.exp(val_loss) if val_loss < 50 else float("inf")
                log.writerow([step, total_loss, val_loss, val_ppl])
                stream.flush()
                print(f"pretrain_eval step={step} train_loss={total_loss:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}", flush=True)
                if val_loss < best_val:
                    best_val = val_loss
                    evaluations_without_improvement = 0
                    torch.save({"model": model.state_dict(), "config": config, "vocab_size": tokenizer.get_vocab_size()}, args.out_dir / "best_checkpoint.pt")
                else:
                    evaluations_without_improvement += 1
                    if args.patience_evals > 0 and evaluations_without_improvement >= args.patience_evals:
                        print(f"pretrain_early_stop step={step} no_improvement_evals={evaluations_without_improvement}", flush=True)
                        break

    summary = {
        "parameters": param_count,
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "train_chunks": len(train_chunks),
        "validation_chunks": len(val_chunks),
        "validation_chunks_evaluated": min(len(val_chunks), args.max_eval_chunks),
        "steps": completed_steps,
        "steps_per_epoch": steps_per_epoch,
        "data_epochs_seen": sampler.epochs_seen,
        "training_tokens_seen": sampler.samples_seen * int(config["max_sequence_length"]),
        "maximum_steps_configured": args.max_steps,
        "best_validation_loss": best_val,
        "scope": "defensive cybersecurity pilot; not a capability or safety certification",
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
