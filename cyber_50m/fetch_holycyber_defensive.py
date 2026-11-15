"""Download train partitions for selected defensive HolyCyber domains."""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = "https://huggingface.co/datasets/Reizxn/HolyCyber/resolve/main/layers"
DOMAINS = (
    "vulnerability_management", "detection_engineering", "defense_engineering",
    "incident_response", "network_security", "cloud_security", "identity_access",
    "governance_risk", "secure_coding", "cryptography", "ai_ml_security",
)


def main() -> None:
    root = Path("data/raw/third_party/holycyber")
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for domain in DOMAINS:
        filename = f"{domain}_train.jsonl"
        destination = root / filename
        url = f"{REPO}/{domain}/train.jsonl"
        request = urllib.request.Request(url, headers={"User-Agent": "defensive-cyber-50m-dataset-fetch/1.0"})
        digest = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        except (urllib.error.URLError, TimeoutError) as error:
            manifest.append({"file": filename, "url": url, "error": str(error)})
            print(f"skipped {domain}: {error}")
            continue
        manifest.append({"file": filename, "url": url, "bytes": size, "sha256": digest.hexdigest(),
                         "retrieved_utc": datetime.now(timezone.utc).isoformat(), "split": "train only",
                         "git_tracked": False})
        print(f"downloaded {domain}: {size:,} bytes")
    (root / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
