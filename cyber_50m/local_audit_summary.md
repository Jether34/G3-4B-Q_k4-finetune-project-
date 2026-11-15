# Local dataset audit (2026-11-16)

The audit ran against the seven ZIP archives and Copilot export in the parent workspace. Raw files remain outside the repository. Machine-readable details and cleaned outputs are generated under the gitignored `data/derived/` directory.

| Source | Rows/files | Audit result | Recommended use |
| --- | ---: | --- | --- |
| AI agent incidents, clean CSV | 395 rows | No malformed rows, blank cells, or exact duplicate rows detected. Taxonomy mixes descriptive names with shorthand/multi-label tags. | Small, topic-specific incident summaries only; normalize taxonomy before classification. |
| Global cybersecurity threats | 3,000 rows | No malformed rows, blanks, or exact duplicates detected. Archive has no license or provenance note. | Hold out until provenance, licensing, and label-generation method are established. |
| CICIDS2017 flow data | 2,520,751 rows | No malformed rows detected; exact duplicates not computed. Class counts are highly imbalanced: Normal 2,095,057; Port Scanning 90,694; Web Attacks 2,143; Brute Force 9,150; DDoS 128,014; Bots 1,948; DoS 193,745. | Separate defensive flow classifier. Use class-aware metrics and a time/day-grouped split; not LLM text. |
| Escalation-within-15-minutes table | 25,000 rows | No malformed rows, blanks, or exact duplicates detected. Label: 22,509 negative, 2,491 positive. | Separate defensive alert-risk classifier; account for 9.96% positive rate. |
| GraphML bundle | 3 files | Not parsed by the CSV audit; node/edge schemas and labels need review. | Keep out of language-model training until graph semantics and license are documented. |
| 6G network-slicing security | 1,000 rows | No malformed rows, blanks, or exact duplicates detected. `anomaly_label`: 852 normal, 148 anomaly. `recent_anomaly_year` is populated only for anomalies, so it leaks the label. | Separate defensive classifier; drop `recent_anomaly_year` from features. |
| Network security feature table | 5,000 rows | No malformed rows, blanks, or exact duplicates detected; no target/label column is present. | Do not use for supervised training until trusted labels and provenance are supplied. |
| Copilot activity export | 3,065 rows | 1,559 Human, 1,506 AI, 38 blank messages; 1,522 nonblank Human messages exported as private style samples after common email/IP/credential redaction. | Private tone examples only. Not instruction-response data and not committed. Review for remaining personal information before training. |

The local text preparation yields 395 incident-summary records plus CISA KEV, CWE prevention, and extracted NIST SP 800-61 Rev. 3 defensive reference text. Combined with the small style export, this is only a few megabytes of text, far below a useful from-scratch corpus for a 50M model. Do not interpret the row count of the flow tables as language-model training volume.

The `dataset.csv` and network tables appear task-specific; the current audit validates CSV structure and missingness, not whether their labels are accurate or their samples are independent. External data needs source-level documentation and grouped/time-aware validation before model use.
