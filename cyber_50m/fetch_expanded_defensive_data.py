"""Fetch curated official and openly licensed defensive cybersecurity references."""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SOURCES = {
    "nist_800_53_controls.xml": (
        "https://csrc.nist.gov/CSRC/media/Projects/risk-management/800-53%20Downloads/800-53r5/SP_800-53_v5_1_XML.xml",
        "NIST SP 800-53 Rev. 5 controls; official derivative data format",
    ),
    "nist_800_53a_assessment.txt": (
        "https://csrc.nist.gov/files/pubs/sp/800/53/a/r5/final/docs/sp800-53ar5-assessment-procedures.txt",
        "NIST SP 800-53A Rev. 5 assessment procedures",
    ),
    "nist_800_218_ssdf.pdf": (
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218.pdf",
        "NIST SP 800-218 Secure Software Development Framework",
    ),
    "cisa_incident_playbooks.pdf": (
        "https://www.cisa.gov/sites/default/files/2024-08/Federal_Government_Cybersecurity_Incident_and_Vulnerability_Response_Playbooks_508C.pdf",
        "CISA incident and vulnerability response playbooks",
    ),
    "cisa_incident_plan_basics.pdf": (
        "https://www.cisa.gov/sites/default/files/2024-02/Incident-Response-Plan-Basics_508c_1Feb2024.pdf",
        "CISA incident response plan basics",
    ),
    "owasp_cheatsheets_master.zip": (
        "https://codeload.github.com/OWASP/CheatSheetSeries/zip/refs/heads/master",
        "OWASP Cheat Sheet Series, CC BY-SA 4.0",
    ),
}


def main() -> None:
    out = Path("data/raw/official").resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for filename, (url, description) in SOURCES.items():
        target = out / filename
        request = urllib.request.Request(url, headers={"User-Agent": "defensive-cyber-50m-dataset-fetch/1.0"})
        digest = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        except (urllib.error.URLError, TimeoutError) as error:
            manifest.append({"file": filename, "url": url, "description": description, "error": str(error)})
            print(f"skipped {filename}: {error}")
            continue
        manifest.append({
            "file": filename,
            "url": url,
            "description": description,
            "bytes": size,
            "sha256": digest.hexdigest(),
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "git_tracked": False,
        })
        print(f"downloaded {filename}: {size:,} bytes")
    (out / "expanded_download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
