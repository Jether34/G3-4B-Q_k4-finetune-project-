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
    text_source = Path("data/derived/official_defensive_text.jsonl")
    if not text_source.exists():
        raise SystemExit("Missing official defensive corpus; run prepare_official_text.py first")
    out = Path("data/kaggle-upload")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(text_source, out / text_source.name)
    if style_source.exists():
        shutil.copy2(style_source, out / style_source.name)
    shutil.copy2(Path("cyber_50m/train_gpt.py"), out / "train_gpt.py")
    shutil.copy2(Path("cyber_50m/model_config.json"), out / "model_config.json")
    (out / "README.md").write_text(
        "Private inputs and training code for the defensive cyber 50M Kaggle pilot. "
        "Contains public defensive reference text and, if present, user-authorized Human-only "
        "tone samples with common email/IP/credential patterns redacted. Keep private.\n",
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
