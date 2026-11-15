"""Prepare a clean train/test split from the MIT-licensed Cybersecurity QAA data."""
from __future__ import annotations

import json
import random
from pathlib import Path


def main() -> None:
    source = Path("data/raw/official/cybersecurity_qaa.json")
    training_out = Path("data/derived/cybersecurity_qa_train.jsonl")
    evaluation_out = Path("data/derived/cybersecurity_qa_eval.jsonl")
    training_out.parent.mkdir(parents=True, exist_ok=True)
    raw = json.loads(source.read_text(encoding="utf-8"))
    eligible = []
    for index, row in enumerate(raw):
        item = row.get("vars", {})
        if item.get("reviewed_by_expert") != "TRUE":
            continue
        theme = str(item.get("theme", ""))
        if theme.casefold() == "attacks_and_defences":
            continue
        if int(item.get("expert_score") or 0) < 1:
            continue
        question = " ".join(str(item.get("question", "")).split())
        answer = " ".join(str(item.get("answer", "")).split())
        if len(question) < 20 or len(answer) < 80:
            continue
        eligible.append({
            "id": f"qaa-{index:04d}",
            "source": "Cybersecurity QAA (MIT; expert-reviewed subset)",
            "theme": theme or "Unspecified",
            "question_type": str(item.get("question_type", "Unspecified")),
            "expert_score": int(item.get("expert_score") or 0),
            "question": question,
            "answer": answer,
            "assertions": [
                " ".join(str(assertion.get("value", "")).split())
                for assertion in row.get("assert", [])
                if isinstance(assertion, dict) and assertion.get("value")
            ],
            "text": f"Question: {question}\nAnswer: {answer}",
        })
    rng = random.Random(5021)
    rng.shuffle(eligible)
    test_count = max(1, round(len(eligible) * 0.2))
    heldout, train = eligible[:test_count], eligible[test_count:]
    training_out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train), encoding="utf-8")
    evaluation_out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in heldout), encoding="utf-8")
    print(json.dumps({
        "source_rows": len(raw),
        "expert_reviewed_usable": len(eligible),
        "training_rows": len(train),
        "heldout_rows": len(heldout),
        "training_file": str(training_out),
        "heldout_file": str(evaluation_out),
        "license": "MIT; preserve upstream notice",
    }, indent=2))


if __name__ == "__main__":
    main()
