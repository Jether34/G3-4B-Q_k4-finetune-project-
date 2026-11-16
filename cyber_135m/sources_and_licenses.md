# Source register for the 135M adaptation

We are building a bounded, source-audited defensive assistant dataset, not mirroring every dataset online. Each source must have an identifiable license, a defensive use, a stable split, and a quality check. Kaggle's downloadable tables are often intended for intrusion-detection classifiers; they are not automatically language-model training data.

| Source | Use in this run | Rights / constraints | Link |
|---|---|---|---|
| SmolLM2-135M-Instruct | Pretrained English and instruction starting point; all model weights are fine-tuned | Apache-2.0 model card; retain upstream attribution and configuration | https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct |
| Fenrir v2.1 | Main defensive instruction dataset; exact deduplication and validation split | Dataset card states Apache-2.0; check source card at retrieval time | https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1 |
| CybersecurityQAA | Expert-reviewed training pairs; separate evaluation split protected | MIT; retain notice; only human-reviewed rows | https://huggingface.co/datasets/Rowden/CybersecurityQAA |
| HolyCyber selected subset | Auxiliary train examples from the existing per-row license-filtered bundle | Preserve each row's source and license; not reviewed row by row | https://huggingface.co/datasets/Reizxn/HolyCyber |
| NIST CSF 2.0 reference | Authoritative machine-readable controls for future retrieval and additional curated examples | Preserve NIST source and version | https://csrc.nist.gov/Projects/cybersecurity-framework/Filters |
| CISA KEV | Current remediation prioritization reference; use as retrieval data rather than memorize a stale snapshot | Preserve update date and source; refresh at use time | https://www.cisa.gov/known-exploited-vulnerabilities-catalog |
| MITRE CWE | Weakness descriptions and mitigations | Use mitigation-oriented fields with attribution | https://cwe.mitre.org/data/downloads.html |
| OWASP Cheat Sheet Series | Application security reference material | CC BY-SA 4.0; comply with attribution/share-alike terms if sharing adaptations | https://github.com/OWASP/CheatSheetSeries |
| Cosmopedia (optional) | Not needed for this run because English behavior comes from the pretrained model | Apache-2.0 dataset card; synthetic source; do not let it displace cybersecurity examples | https://huggingface.co/datasets/HuggingFaceTB/cosmopedia |

## Excluded from this training bundle

- The Threat Intelligence Dataset labeled CC BY-NC-SA 4.0: excluded to avoid silently constraining use of the resulting model.
- The large `oi-uae/cyber-security` aggregate: its card says research-only/non-commercial, notes upstream rights vary, and includes offensive material.
- Trendyol in the previous audit: it contributed no unique rows after deduplication against Fenrir.
- Raw Copilot CSV and message rows: never include these in the Kaggle bundle; use only `style_profile.md` derived locally.
- CICIDS2017, UNSW-NB15, and other packet/flow tables: keep for separate supervised detection experiments with explicit labels; do not concatenate as language text.
- Unlabeled or unlicensed Kaggle dataset dumps and generated QA with unclear provenance.
