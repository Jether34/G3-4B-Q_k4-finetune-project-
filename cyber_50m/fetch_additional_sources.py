"""Fetch and normalize additional defensive cybersecurity and language data on Kaggle.

Third-party records are processed as data only. Threat-intelligence records are
restricted to defensive analysis/response categories; the NC-SA source remains
non-commercial and share-alike for any downstream distribution.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path


DATA = Path("/kaggle/working/data")
SPACE = re.compile(r"\s+")
SAFE_CTI_CATEGORIES = {
    "api-security", "cloud-saas-security", "container-security",
    "defensive-countermeasures", "digital-risk-management", "email-threats",
    "ics-advisories", "ics-ot-security", "incident-case-analysis",
    "incident-response-forensics", "security-monitoring-detection",
    "threat-hunting", "threat-intelligence", "threat-intelligence-feeds",
    "threat-intelligence-operations", "threat-modeling", "vulnerabilities-cves",
    "web-application-security",
}


def clean(value) -> str:
    return SPACE.sub(" ", str(value or "")).strip()


def dump(name: str, rows: list[dict]) -> int:
    path = DATA / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def chat_row(row: dict, source: str, index: int) -> dict | None:
    user = clean(row.get("user", row.get("instruction", "")))
    extra = clean(row.get("input", ""))
    answer = clean(row.get("assistant", row.get("output", "")))
    system = clean(row.get("system", ""))
    if extra:
        user = f"{user}\nContext: {extra}"
    if len(user) < 20 or len(answer) < 80:
        return None
    prefix = (system + "\n") if system else "You are a defensive cybersecurity assistant.\n"
    prompt = f"{prefix}User: {user}\nAssistant:"
    key = hashlib.sha256((user.casefold() + "\n" + answer.casefold()).encode()).hexdigest()
    return {
        "id": str(row.get("id", row.get("row_id", f"{source}-{index}"))),
        "source": source,
        "license": row.get("license", "See source dataset card"),
        "group_key": key,
        "prompt": prompt,
        "question": user,
        "answer": answer,
        "text": prompt + " " + answer,
    }


def load_hf_json(url: str) -> list[dict]:
    from datasets import load_dataset
    data = load_dataset("json", data_files=url, split="train")
    return [dict(row) for row in data]


def main() -> None:
    from datasets import load_dataset

    counters = Counter()
    dedupe: set[str] = set()
    heldout_path = DATA / "cybersecurity_qa_eval.jsonl"
    if heldout_path.exists():
        for line in heldout_path.open(encoding="utf-8"):
            try:
                heldout = json.loads(line)
            except json.JSONDecodeError:
                continue
            question, answer = clean(heldout.get("question")), clean(heldout.get("answer"))
            if question and answer:
                dedupe.add(hashlib.sha256((question.casefold() + "\n" + answer.casefold()).encode()).hexdigest())
        counters["heldout_pairs_protected"] = len(dedupe)
    fenrir = []
    fenrir_url = (
        "https://huggingface.co/datasets/AlicanKiraz0/"
        "Cybersecurity-Dataset-Fenrir-v2.1/resolve/main/"
        "220426-CyberSec-Dataset_escaped.jsonl"
    )
    for i, raw in enumerate(load_hf_json(fenrir_url)):
        row = chat_row(raw, "Fenrir v2.1 (Apache-2.0)", i)
        if row and row["group_key"] not in dedupe:
            dedupe.add(row["group_key"])
            row["license"] = "Apache-2.0"
            fenrir.append(row)
        else:
            counters["fenrir_short_or_duplicate"] += 1
    counters["fenrir_rows"] = dump("fenrir_defensive_sft.jsonl", fenrir)

    # The source repository publishes an entity-disjoint train/eval split.
    # Only train is used. Whitelist defensive analyst categories and exclude
    # malware, exploit/zero-day, red-team, ransomware, actor-targeting and
    # other potentially operationally harmful categories.
    threat_rows = []
    threat_url = (
        "https://raw.githubusercontent.com/reloading01/"
        "threat-intelligence-dataset/main/data/train.jsonl"
    )
    request = urllib.request.Request(threat_url, headers={"User-Agent": "cyber-50m-research/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        for i, line in enumerate(response):
            try:
                raw = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                counters["threat_invalid_json"] += 1
                continue
            metadata = raw.get("metadata", {}) or {}
            category = clean(metadata.get("category", "")).casefold()
            if category not in SAFE_CTI_CATEGORIES:
                counters["threat_excluded_category"] += 1
                continue
            grounding = clean(metadata.get("grounding", "")).casefold()
            if "openphish" in grounding:
                counters["threat_excluded_openphish"] += 1
                continue
            row = chat_row(raw, "Threat Intelligence Dataset (CC BY-NC-SA 4.0)", i)
            if row and row["group_key"] not in dedupe:
                dedupe.add(row["group_key"])
                row["license"] = "CC BY-NC-SA 4.0; non-commercial, share-alike, attribute upstream"
                row["domain"] = category
                threat_rows.append(row)
            else:
                counters["threat_short_or_duplicate"] += 1
    counters["threat_rows"] = dump("threat_intel_defensive_sft.jsonl", threat_rows)

    trendyol_rows = []
    trendyol = load_dataset("Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset", split="train")
    for i, raw in enumerate(trendyol):
        row = chat_row(dict(raw), "Trendyol Cybersecurity Defense (Apache-2.0)", i)
        if row and row["group_key"] not in dedupe:
            dedupe.add(row["group_key"])
            row["license"] = "Apache-2.0"
            trendyol_rows.append(row)
        else:
            counters["trendyol_short_or_duplicate"] += 1
    counters["trendyol_rows"] = dump("trendyol_defensive_sft.jsonl", trendyol_rows)

    # Cosmopedia was optional generic text. Its remote OpenStax shard triggered
    # repeated Hugging Face file-descriptor retries in the active run. It adds
    # little value beside the selected docs and Q&A, so do not fetch it at run time.
    print("Skipping optional Cosmopedia download; cyber corpus and official references are sufficient for this pilot.", flush=True)
    summary = {
        "counts": dict(counters),
        "cti_categories_included": sorted(SAFE_CTI_CATEGORIES),
        "cti_train_only": True,
        "cosmopedia": "skipped to avoid unneeded runtime downloads and retry loops",
        "note": "Threat Intelligence Dataset is CC BY-NC-SA 4.0; private/non-commercial research only unless rights are separately cleared.",
    }
    (DATA / "additional_sources_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
