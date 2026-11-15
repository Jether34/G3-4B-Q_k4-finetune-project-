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
    source = Path("data/derived/copilot_style_private.jsonl")
    if not source.exists():
        raise SystemExit("Missing private style file; prepare local data with --include-copilot-style first")
    out = Path("data/kaggle-upload")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, out / source.name)
    (out / "README.md").write_text(
        "Private user-authorized tone samples for a defensive cybersecurity model. "
        "Contains Human messages only, with common email/IP/credential patterns redacted. "
        "Do not change this dataset to public visibility. Review contents before upload.\n",
        encoding="utf-8",
    )
    metadata = {
        "title": "Private defensive cyber 50M tone samples",
        "id": f"{args.owner}/{args.slug}",
        "licenses": [{"name": "other"}],
    }
    (out / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared {len(source.read_text(encoding='utf-8').splitlines())} style records at {out.resolve()}")
    print("Upload with Kaggle's --private option; do not publish this dataset.")


if __name__ == "__main__":
    main()
