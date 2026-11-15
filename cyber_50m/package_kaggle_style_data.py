"""Prepare a private Kaggle input dataset from the reviewed local style file."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True, help="Kaggle account username")
    parser.add_argument("--slug", default="defensive-cyber-50m-style-v1")
    args = parser.parse_args()
    style_source = Path("data/derived/copilot_style_private.jsonl")
    text_source = Path("data/derived/combined_defensive_text.jsonl")
    if not text_source.exists():
        text_source = Path("data/derived/official_defensive_text.jsonl")
    if not text_source.exists():
        raise SystemExit("Missing official defensive corpus; run prepare_official_text.py first")
    out = Path("data/kaggle-upload")
    out.mkdir(parents=True, exist_ok=True)
    stale_combined = out / "combined_defensive_text.jsonl"
    if stale_combined.exists():
        stale_combined.unlink()
    shutil.copy2(text_source, out / "official_defensive_text.jsonl")
    incident_source = Path("data/derived/incident_text.jsonl")
    if incident_source.exists():
        shutil.copy2(incident_source, out / "incident_text.jsonl")
    qa_source = Path("data/derived/cybersecurity_qa_train.jsonl")
    if qa_source.exists():
        shutil.copy2(qa_source, out / "cybersecurity_qa_train.jsonl")
    holycyber_source = Path("data/derived/holycyber_defensive_train.jsonl")
    if holycyber_source.exists():
        shutil.copy2(holycyber_source, out / "holycyber_defensive_train.jsonl")
    qa_eval_source = Path("data/derived/cybersecurity_qa_eval.jsonl")
    if qa_eval_source.exists():
        shutil.copy2(qa_eval_source, out / "cybersecurity_qa_eval.jsonl")
    if style_source.exists():
        shutil.copy2(style_source, out / style_source.name)
    shutil.copy2(Path("cyber_50m/train_gpt.py"), out / "train_gpt.py")
    shutil.copy2(Path("cyber_50m/evaluate_tasks.py"), out / "evaluate_tasks.py")
    shutil.copy2(Path("cyber_50m/train_sft.py"), out / "train_sft.py")
    shutil.copy2(Path("cyber_50m/fetch_additional_sources.py"), out / "fetch_additional_sources.py")
    shutil.copy2(Path("cyber_50m/model_config.json"), out / "model_config.json")
    (out / "README.md").write_text(
        "Private defensive cybersecurity training input. Sources: CISA KEV and MITRE CWE; "
        "NIST SP 800-61, 800-53/53A, and 800-218; selected OWASP Cheat Sheets (CC BY-SA 4.0); "
        "Cybersecurity QAA expert-reviewed subset (MIT; "
        "https://huggingface.co/datasets/Rowden/CybersecurityQAA); and selected HolyCyber "
        "defensive examples (per-row source/license recorded; "
        "https://huggingface.co/datasets/Reizxn/HolyCyber). Only train-split rows from selected "
        "defensive domains and approved recorded licenses are included; this third-party set is "
        "not expert-reviewed row by row. Retain OWASP attribution and license. The separately "
        "named QAA evaluation split is never used for training. Additional sources are fetched "
        "inside the private Kaggle notebook: Fenrir and Trendyol (Apache-2.0), selected Cosmopedia "
        "language-only samples (Apache-2.0), and defensive-only CTI categories from the Threat "
        "Intelligence Dataset (CC BY-NC-SA 4.0; non-commercial/share-alike; upstream attribution "
        "required). Do not redistribute that dataset or a derived model commercially without "
        "separate rights review. The optional Copilot file contains redacted Human-only tone "
        "samples and must remain private.\n",
        encoding="utf-8",
    )
    metadata = {
        "title": "Private defensive cyber 50M pilot data",
        "id": f"{args.owner}/{args.slug}",
        "licenses": [{"name": "other"}],
    }
    (out / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared private Kaggle inputs at {out.resolve()}")
    print("Kaggle CLI creates datasets privately by default; do not pass --public.")


if __name__ == "__main__":
    main()
