"""Merge baseline and expanded defensive references with exact duplicate removal."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path("data/derived")
    inputs = [
        root / "official_defensive_text.jsonl",
        root / "expanded_defensive_text.jsonl",
    ]
    output = root / "combined_defensive_text.jsonl"
    seen: set[str] = set()
    rows = []
    for source_path in inputs:
        if not source_path.exists():
            continue
        with source_path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = " ".join(str(row.get("text", "")).split())
                digest = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
                if len(text) >= 80 and digest not in seen:
                    seen.add(digest)
                    row["text"] = text
                    rows.append(row)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"records": len(rows), "output": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
