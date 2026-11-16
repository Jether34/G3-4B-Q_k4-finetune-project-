"""Save deterministic side-by-side base and adapted answers for manual review."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    "Explain what a high-severity alert means and list the first defensive triage checks.",
    "A user reports a suspicious sign-in from an unfamiliar country. Give a concise, safe response plan.",
    "How should a small organization prioritize an unpatched vulnerability on an internet-facing server?",
    "Describe three practical ways to reduce phishing risk without blaming the recipient.",
    "What information should an incident responder preserve after a suspected account compromise?",
]
SYSTEM = "You are a careful defensive cybersecurity assistant. Give concise, accurate, authorized defensive guidance and state uncertainty."


def run(model_id: str, label: str, tokenizer, device: str):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
    rows = []
    for prompt in PROMPTS:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(tokens, max_new_tokens=128, do_sample=False, repetition_penalty=1.08,
                                   pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(output[0, tokens.shape[1]:], skip_special_tokens=True).strip()
        rows.append({"model": label, "prompt": prompt, "answer": answer})
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapted))
    rows = run(args.base, "base", tokenizer, device) + run(str(args.adapted), "adapted", tokenizer, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "prompt", "answer"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} generations for blinded/manual review to {args.output}", flush=True)


if __name__ == "__main__":
    main()
