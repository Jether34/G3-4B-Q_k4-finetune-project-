"""Audit local cybersecurity archives and optionally prepare private style samples.

No source data is copied into Git. Run from any directory; pass --workspace-root
to the folder containing archive (1).zip ... archive (7).zip and the Copilot CSV.
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import re
import zipfile
from pathlib import Path
from typing import TextIO


EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SECRET = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*['\"]?[^\s,'\"]+"
)
WHITESPACE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    value = SECRET.sub("[REDACTED_SECRET]", value)
    value = EMAIL.sub("[REDACTED_EMAIL]", value)
    value = IPV4.sub("[REDACTED_IP]", value)
    return WHITESPACE.sub(" ", value).strip()


def csv_stream(stream: TextIO, source: str, report: dict) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(stream)
    fields = reader.fieldnames or []
    rows: list[dict[str, str]] = []
    blanks: collections.Counter[str] = collections.Counter()
    seen: set[str] = set()
    duplicates = 0
    malformed = 0
    for row in reader:
        if None in row:
            malformed += 1
            continue
        normalized = {str(k).strip(): (v or "").strip() for k, v in row.items()}
        for key, value in normalized.items():
            if not value:
                blanks[key] += 1
        fingerprint = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        if fingerprint in seen:
            duplicates += 1
        seen.add(fingerprint)
        rows.append(normalized)
    target_counts: dict[str, dict[str, int]] = {}
    for field in fields:
        normalized_name = field.casefold().replace("_", " ")
        if any(term in normalized_name for term in ("label", "target", "class", "anomaly", "escalation")):
            target_counts[field] = dict(collections.Counter(row.get(field, "") for row in rows))
        elif normalized_name in {"attack type", "attack type name", "severity"}:
            target_counts[field] = dict(collections.Counter(row.get(field, "") for row in rows))
    report[source] = {
        "rows": len(rows),
        "columns": fields,
        "malformed_rows": malformed,
        "exact_duplicate_rows": duplicates,
        "blank_cells_by_column": dict(blanks),
        "candidate_target_value_counts": target_counts,
    }
    return fields, rows


def read_archive(workspace: Path, report: dict, text_rows: list[dict[str, str]]) -> None:
    for archive in sorted(workspace.glob("archive (*.zip")):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if not member.lower().endswith(".csv"):
                    continue
                key = f"{archive.name}/{member}"
                # Keep the 2.5M-row flow file streaming; it is tabular and not LM text.
                if "cicids2017_cleaned.csv" in member.lower():
                    with io.TextIOWrapper(zf.open(member), encoding="utf-8-sig", errors="replace", newline="") as f:
                        reader = csv.DictReader(f)
                        blanks: collections.Counter[str] = collections.Counter()
                        labels: collections.Counter[str] = collections.Counter()
                        fields = reader.fieldnames or []
                        label_fields = [name for name in fields
                                        if any(t in name.casefold().replace("_", " ")
                                               for t in ("label", "target", "class", "anomaly"))
                                        or name.casefold().replace("_", " ") == "attack type"]
                        rows = malformed = 0
                        for row in reader:
                            rows += 1
                            if None in row:
                                malformed += 1
                                continue
                            for name, value in row.items():
                                if not (value or "").strip():
                                    blanks[name] += 1
                            for name in label_fields:
                                labels[(row.get(name) or "").strip()] += 1
                        report[key] = {
                            "rows": rows,
                            "columns": reader.fieldnames or [],
                            "malformed_rows": malformed,
                            "blank_cells_by_column": dict(blanks),
                            "candidate_target_value_counts": dict(labels),
                            "exact_duplicate_rows": "not computed for large file",
                        }
                    continue
                with io.TextIOWrapper(zf.open(member), encoding="utf-8-sig", errors="replace", newline="") as f:
                    fields, rows = csv_stream(f, key, report)
                if member.lower() == "ai_agent_incidents_clean.csv":
                    for row in rows:
                        title = clean_text(row.get("title", ""))
                        summary = clean_text(row.get("summary", ""))
                        if title and summary:
                            text_rows.append({
                                "source": key,
                                "id": row.get("incident_id", ""),
                                "text": f"Incident: {title}\nSummary: {summary}",
                            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived"))
    parser.add_argument("--include-copilot-style", action="store_true",
                        help="write redacted Human messages to an ignored local file")
    args = parser.parse_args()
    workspace = args.workspace_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"workspace_root": str(workspace), "sources": {}}
    incident_text: list[dict[str, str]] = []
    read_archive(workspace, report["sources"], incident_text)
    (out / "incident_text.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in incident_text),
        encoding="utf-8",
    )

    copilot = workspace / "copilot-activity-history.csv"
    if copilot.exists():
        human_count = ai_count = blank = 0
        tone_rows: list[dict[str, str]] = []
        with copilot.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                author = (row.get("Author") or "").strip().casefold()
                message = clean_text(row.get("Message") or "")
                if not message:
                    blank += 1
                if author == "human":
                    human_count += 1
                    if args.include_copilot_style and message:
                        tone_rows.append({"text": message})
                elif author == "ai":
                    ai_count += 1
        report["copilot"] = {
            "human_rows": human_count,
            "ai_rows": ai_count,
            "blank_messages": blank,
            "style_export_enabled": args.include_copilot_style,
            "note": "Human-only messages are style samples, not instruction-response pairs.",
        }
        if args.include_copilot_style:
            (out / "copilot_style_private.jsonl").write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in tone_rows),
                encoding="utf-8",
            )

    report["text_corpus_records"] = len(incident_text)
    (out / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "workspace_root": str(workspace),
        "audited_sources": len(report["sources"]),
        "incident_text_records": len(incident_text),
        "copilot": report.get("copilot", "not found"),
        "report": str(out / "audit_report.json"),
        "outputs_are_gitignored": True,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
