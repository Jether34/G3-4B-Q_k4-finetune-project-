"""Build a private, minimal Kaggle input for the 135M defensive adaptation."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def copy_if_exists(source: Path, dest: Path) -> None:
    if source.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", default="defensive-cyber-135m-data")
    parser.add_argument("--output", type=Path, default=Path("data/kaggle-upload-135m-next"))
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing bundle directory: {out.resolve()}")
    (out / "data").mkdir(parents=True)
    (out / "cyber_135m").mkdir(parents=True)

    for filename in ("cybersecurity_qa_train.jsonl", "cybersecurity_qa_eval.jsonl", "holycyber_defensive_train.jsonl", "official_defensive_text.jsonl"):
        copy_if_exists(Path("data/derived") / filename, out / "data" / filename)
    for filename in ("prepare_sft.py", "train_sft.py", "evaluate_samples.py", "evaluate_holdout.py", "guardrail_policy.py", "guardrails.md", "style_profile.md", "model_card.md", "sources_and_licenses.md", "requirements.txt"):
        copy_if_exists(Path("cyber_135m") / filename, out / "cyber_135m" / filename)

    readme = (
        "Private input bundle for a defensive full-weight fine-tune of "
        "HuggingFaceTB/SmolLM2-135M-Instruct (Apache-2.0). Includes the reviewed QAA train/eval "
        "split and reference-only HolyCyber/official defensive text; Fenrir v2.1 is fetched from "
        "its Apache-2.0 dataset-card source by the Kaggle notebook. Raw Copilot messages and style rows "
        "rows are not included; only an abstract style preference is bundled. The Threat Intelligence "
        "Dataset (CC BY-NC-SA 4.0), Trendyol exact duplicates, and unlicensed Kaggle aggregates are excluded. "
        "Check upstream cards and retain notices before redistribution.\n"
    )
    (out / "README.md").write_text(readme, encoding="utf-8")
    metadata = {
        "title": "Private defensive cyber 135M adaptation data",
        "id": f"{args.owner}/{args.slug}",
        "licenses": [{"name": "other"}],
    }
    (out / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared private 135M Kaggle input: {out.resolve()}")
    print("Copilot message rows copied: 0")


if __name__ == "__main__":
    main()
