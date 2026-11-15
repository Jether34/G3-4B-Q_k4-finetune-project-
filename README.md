# Defensive Cybersecurity 50M Language Model

A from-scratch decoder-only Transformer experiment for five defensive workflows: security-alert triage, incident-response guidance, vulnerability prioritization, secure configuration, and security awareness. This is a small research model, not a general cybersecurity expert; current references should be retrieved at use time.

## Project files

- `cyber_50m/` contains the model configuration, source catalog, and scripts to audit, clean, download, and prepare defensive text.
- `kaggle/cyber_50m_pretrain.ipynb` trains a short first-run model on Kaggle's free GPU.
- `cyber_50m/local_audit_summary.md` records what the supplied files contain and their current limitations.

The old Gemma 3 4B demo, control-center UI, and abliteration experiments have been removed from the current project tree.

## Data safety

Raw archives, downloaded reference data, processed corpora, Copilot conversation samples, Kaggle upload bundles, checkpoints, and credentials are local or private runtime inputs and are excluded from Git. The preprocessing scripts keep network-flow tables separate from language-model text. Only defensive reference material and reviewed, sanitized text belong in the LM corpus.

## Local preparation

Run from the repository root:

```powershell
python -m pip install -r cyber_50m/requirements-data.txt
python cyber_50m/prepare_local_data.py --workspace-root .. --include-copilot-style
python cyber_50m/fetch_reference_data.py
python cyber_50m/prepare_official_text.py
```

Review `data/derived/` locally before uploading anything. The Copilot export contains private conversations; the generated style file is redacted for common email/IP/credential patterns, but needs human review.

## Kaggle first run

See [`kaggle/README.md`](kaggle/README.md) for the private data packaging steps and GPU notebook. The current corpus is only a pilot corpus and is far too small to claim useful broad cybersecurity capability. The first run is capped at 100 steps to check the pipeline; improve data coverage and evaluate before treating its checkpoint as a usable assistant.

No claim of 90% cybersecurity task coverage is made. The network-flow datasets are for separate defensive classifiers and aren't fed to the language model as prose.
