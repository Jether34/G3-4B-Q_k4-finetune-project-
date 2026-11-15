# Defensive Cyber 50M Model

This is a separate from-scratch, decoder-only Transformer experiment. It does not change the existing Gemma 3 4B QLoRA workflow. The intended scope is defensive support: security-alert triage, incident-response guidance, vulnerability prioritization, secure configuration, and security awareness.

## What 50M parameters can do

Treat this as a constrained, task-specific model, not a general cybersecurity expert. Prefer a few measurable tasks and use retrieval from current authoritative sources for facts that change. Network-flow and telemetry tables belong in separate classifiers; they are not natural-language pretraining text.

No single dataset or model can cover “90% of cybersecurity tasks.” Coverage depends on a defined task list and evaluation set. This project starts with the five defensive areas above and will report per-task performance rather than a single coverage claim.

## Local source data

The source archives and `copilot-activity-history.csv` remain in the parent workspace. Run the local audit and preparation script from the repository root:

```powershell
python cyber_50m/prepare_local_data.py --workspace-root .. --output-dir data/derived
```

The script writes only ignored local artifacts. It produces a quality report, a small text corpus from incident titles/summaries, and an optional tone-only file from the Human rows in your Copilot export. The tone export is opt-in and redacts common email, IP, and credential patterns. Review it locally before training. It is not committed. Since Human messages alone are not prompt/answer pairs, they can inform style but do not teach the model how to answer cybersecurity questions.

Fetch and convert the official defensive references with:

```powershell
python -m pip install -r cyber_50m/requirements-data.txt
python cyber_50m/fetch_reference_data.py
python cyber_50m/prepare_official_text.py
```

Tabular files are audited separately and are not converted to prose. Use only labeled data for defensive classification; the current `network_security_dataset.csv` has no target column in its schema and needs source documentation or labels before supervised use.

## Additional data sources

See [`data_sources.json`](data_sources.json). The initial recommendations are official, defensive references: CISA Known Exploited Vulnerabilities and cybersecurity advisories, NIST incident-response guidance, NVD CVE records, and MITRE CWE prevention information. For network detection benchmarks, the existing CICIDS2017 data can be supplemented with the official UNSW ToN_IoT or CSE-CIC-IDS2018 releases. Respect each source's terms, citations, and size; do not commit downloaded corpora.

For language-model training, prioritize clean, licensed text and carefully reviewed defensive question/answer examples. Keep a time-based and source-grouped evaluation split. Don't mix generated training examples into evaluation. Don't train on exploit payloads, weaponization instructions, secrets, personal records, or unauthorized data.

## Training pilot

`train_gpt.py` creates a byte-level BPE tokenizer and trains a causal Transformer from scratch using the 50M configuration. The first Kaggle notebook run is capped at 100 steps to confirm the training path. That run is not enough to claim useful expertise. The data should be measured in tokens, not rows. A broad claim like MAE 0.001 is not an LLM quality criterion; use held-out perplexity for language modeling and task-specific precision/recall, calibration, and human review for triage.
