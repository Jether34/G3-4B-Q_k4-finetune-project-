"""Convert the downloaded KEV and CWE references into defensive-only JSONL."""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from pypdf import PdfReader


SPACE = re.compile(r"\s+")


def text_content(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return SPACE.sub(" ", " ".join(element.itertext())).strip()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child(parent: ET.Element, name: str) -> ET.Element | None:
    return next((node for node in parent if local_name(node.tag) == name), None)


def main() -> None:
    raw = Path("data/raw/official")
    output = Path("data/derived/official_defensive_text.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []

    kev_path = raw / "cisa_kev.json"
    kev = json.loads(kev_path.read_text(encoding="utf-8"))
    for item in kev.get("vulnerabilities", []):
        title = item.get("vulnerabilityName", "").strip()
        description = item.get("shortDescription", "").strip()
        action = item.get("requiredAction", "").strip()
        if description or action:
            records.append({
                "source": "CISA KEV",
                "id": item.get("cveID", ""),
                "text": f"Vulnerability: {title}\nDescription: {description}\nDefensive action: {action}".strip(),
            })

    cwe_path = raw / "cwe_latest.xml.zip"
    with zipfile.ZipFile(cwe_path) as archive:
        xml_name = next(name for name in archive.namelist() if name.lower().endswith(".xml"))
        root = ET.fromstring(archive.read(xml_name))
    for weakness in (node for node in root.iter() if local_name(node.tag) == "Weakness"):
        name = weakness.attrib.get("Name", "").strip()
        cwe_id = weakness.attrib.get("ID", "").strip()
        description = text_content(child(weakness, "Description"))
        mitigation_nodes = child(weakness, "Potential_Mitigations")
        mitigations = []
        if mitigation_nodes is not None:
            for mitigation in mitigation_nodes:
                if local_name(mitigation.tag) == "Mitigation":
                    value = text_content(child(mitigation, "Description"))
                    if value:
                        mitigations.append(value)
        safe_mitigations = " ".join(dict.fromkeys(mitigations))
        if name and (description or safe_mitigations):
            records.append({
                "source": "MITRE CWE",
                "id": f"CWE-{cwe_id}" if cwe_id else "",
                "text": f"Weakness: {name}\nDescription: {description}\nPrevention and mitigation: {safe_mitigations}".strip(),
            })

    pdf = PdfReader(str(raw / "nist_sp_800_61r3.pdf"))
    for page_number, page in enumerate(pdf.pages, start=1):
        page_text = SPACE.sub(" ", page.extract_text() or "").strip()
        if len(page_text) >= 200:
            records.append({
                "source": "NIST SP 800-61 Rev. 3",
                "id": f"page-{page_number}",
                "text": page_text,
            })

    seen: set[str] = set()
    unique = []
    for record in records:
        text = SPACE.sub(" ", record["text"]).strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            record["text"] = text
            unique.append(record)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in unique), encoding="utf-8")
    print(json.dumps({"records": len(unique), "output": str(output.resolve()), "gitignored": True}, indent=2))


if __name__ == "__main__":
    main()
