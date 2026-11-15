"""Fetch small, official defensive references into ignored local storage."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SOURCES = {
    "cisa_kev.json": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
    "cwe_latest.xml.zip": "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip",
    "nist_sp_800_61r3.pdf": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r3.pdf",
}


def main() -> None:
    out = Path("data/raw/official").resolve()
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for filename, url in SOURCES.items():
        destination = out / filename
        request = urllib.request.Request(url, headers={"User-Agent": "defensive-cyber-50m-dataset-fetch/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as target:
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        entries.append({
            "file": filename,
            "url": url,
            "bytes": size,
            "sha256": digest.hexdigest(),
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "git_tracked": False,
        })
        print(f"downloaded {filename}: {size:,} bytes; sha256={digest.hexdigest()}")
    (out / "download_manifest.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
