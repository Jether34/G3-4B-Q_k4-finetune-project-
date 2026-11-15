"""Filter and serialize defensive, source-attributed HolyCyber training examples."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path("data/raw/third_party/holycyber")
OUT = Path("data/derived/holycyber_defensive_train.jsonl")
MIN_CORE_SHARE = 0.30
CODE = re.compile(r"```.*?```", re.S)
SPACE = re.compile(r"\s+")


def allowed_license(row: dict) -> bool:
    license_name = str(row.get("license", "")).casefold()
    return (
        "free for defensive use" in license_name
        or "public domain" in license_name
        or "cc by-sa 4.0" in license_name
    )


def clean(text: str) -> str:
    text = CODE.sub(" ", text)
    text = text.split("## Reference:", 1)[0]
    return SPACE.sub(" ", text).strip()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows = []
    skipped = Counter()
    by_domain = Counter()
    by_license = Counter()
    files = sorted(ROOT.glob("*_train.jsonl"))
    for path in files:
        for line in path.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped["invalid_json"] += 1
                continue
            if row.get("split") != "train":
                skipped["not_train_split"] += 1
                continue
            if not allowed_license(row):
                skipped["license_not_approved"] += 1
                continue
            if float(row.get("core_share", 0.0)) < MIN_CORE_SHARE:
                skipped["low_core_share"] += 1
                continue
            messages = row.get("messages", [])
            user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            assistant = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
            system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
            question, answer = clean(str(user)), clean(str(assistant))
            if len(question) < 30 or len(answer) < 160:
                skipped["short_or_empty"] += 1
                continue
            safety = clean(str(system))
            prompt = f"System: {safety}\nUser: {question}\nAssistant:"
            text = f"{prompt} {answer}"
            digest = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
            if digest in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(digest)
            license_name = str(row["license"])
            rows.append({
                "id": row.get("id", ""),
                "source": f"HolyCyber: {row.get('source', 'unknown')}",
                "license": license_name,
                "domain": row.get("domain", ""),
                "task_type": row.get("task_type", ""),
                "group_key": row.get("group_key", ""),
                "core_share": row.get("core_share", 0.0),
                "prompt": prompt,
                "answer": answer,
                "text": text,
            })
            by_domain[str(row.get("domain", "unknown"))] += 1
            by_license[license_name] += 1
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "records": len(rows),
        "source_files": len(files),
        "minimum_core_share": MIN_CORE_SHARE,
        "by_domain": dict(by_domain),
        "by_license": dict(by_license),
        "skipped": dict(skipped),
        "output": str(OUT.resolve()),
        "dataset_card_limit": "Third-party generated set, not reviewed row by row; auxiliary training only; evaluate on held-out expert-reviewed tasks.",
    }
    Path("data/derived/holycyber_filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
