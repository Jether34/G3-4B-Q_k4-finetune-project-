"""Convert curated defensive references into attributed JSONL training records."""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from pypdf import PdfReader


SPACE = re.compile(r"\s+")
CODE = re.compile(r"```.*?```", re.S)
LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
HTML = re.compile(r"<[^>]+>")
ALLOWLIST = re.compile(
    r"(Access_Control|Authentication|Authorization|Cryptographic_Storage|"
    r"Error_Handling|File_Upload|Input_Validation|Logging|Password_Storage|"
    r"Secure_Headers|Session_Management|TLS|Transport_Layer_Security|"
    r"Vulnerability_Disclosure|Secrets_Management|Denial_of_Service)", re.I
)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def clean(text: str) -> str:
    text = CODE.sub(" ", text)
    text = LINK.sub(r"\1", text)
    text = HTML.sub(" ", text)
    return SPACE.sub(" ", text).strip()


def main() -> None:
    raw = Path("data/raw/official")
    output = Path("data/derived/expanded_defensive_text.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []

    for filename, source in [
        ("nist_800_218_ssdf.pdf", "NIST SP 800-218 SSDF"),
        ("cisa_incident_playbooks.pdf", "CISA Incident and Vulnerability Response Playbooks"),
        ("cisa_incident_plan_basics.pdf", "CISA Incident Response Plan Basics"),
    ]:
        pdf_path = raw / filename
        if not pdf_path.exists():
            continue
        pdf = PdfReader(str(pdf_path))
        for page_no, page in enumerate(pdf.pages, start=1):
            text = clean(page.extract_text() or "")
            if len(text) >= 200:
                records.append({"source": source, "id": f"page-{page_no}", "text": text})

    controls = ET.parse(raw / "nist_800_53_controls.xml").getroot()
    for node in controls.iter():
        if local_name(node.tag).lower() not in {"control", "securitycontrol"}:
            continue
        text = clean(" ".join(node.itertext()))
        identifier = node.attrib.get("id") or node.attrib.get("number") or ""
        if len(text) >= 120:
            records.append({"source": "NIST SP 800-53 Rev. 5 Controls", "id": identifier, "text": text})

    assessment_path = raw / "nist_800_53a_assessment.txt"
    if assessment_path.exists():
        text = clean(assessment_path.read_text(encoding="utf-8", errors="replace"))
        for offset in range(0, len(text), 4000):
            part = text[offset:offset + 4000].strip()
            if len(part) >= 200:
                records.append({"source": "NIST SP 800-53A Rev. 5", "id": f"segment-{offset // 4000 + 1}", "text": part})

    archive_path = raw / "owasp_cheatsheets_master.zip"
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if "/cheatsheets/" not in name or not name.lower().endswith(".md") or not ALLOWLIST.search(Path(name).stem):
                continue
            text = clean(archive.read(name).decode("utf-8", errors="replace"))
            if len(text) >= 250:
                records.append({
                    "source": "OWASP Cheat Sheet Series (CC BY-SA 4.0)",
                    "id": Path(name).stem,
                    "text": text,
                })

    seen: set[str] = set()
    unique = []
    for row in records:
        text = clean(row["text"])
        digest = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
        if text and digest not in seen:
            seen.add(digest)
            row["text"] = text
            unique.append(row)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in unique), encoding="utf-8")
    by_source: dict[str, int] = {}
    for row in unique:
        by_source[row["source"]] = by_source.get(row["source"], 0) + 1
    print(json.dumps({"records": len(unique), "by_source": by_source, "output": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
