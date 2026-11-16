# Defensive Cyber 50M Model

## Local chat interface

Install the chat dependencies and start the local browser app from the repository root:

```powershell
python -m pip install -r cyber_50m/requirements-chat.txt
python -m streamlit run cyber_50m/chat_app.py
```

Streamlit opens the chat in your browser (usually `http://localhost:8501`). The app can load the version 7 50M checkpoint or a Hugging Face `save_pretrained` folder from the 135M adaptation. Set the path in the sidebar; the 135M folder defaults to `data/kaggle-output/135m/best_model`. The sidebar also lets you edit and save `cyber_135m/guardrails.md`; the same file is added to the next SFT prompts. Prompt guardrails are not a guarantee, and answers must be verified.

This is a separate from-scratch, decoder-only Transformer experiment. It does not change the existing Gemma 3 4B QLoRA workflow. The intended scope is defensive support: security-alert triage, incident-response guidance, vulnerability prioritization, secure configuration, and security awareness.

## What 50M parameters can do

Treat this as a constrained, task-specific model, not a general cybersecurity expert. Prefer a few measurable tasks and use retrieval from current authoritative sources for facts that change. Network-flow and telemetry tables belong in separate classifiers; they are not natural-language pretraining text.

No single dataset or model can cover ?90% of cybersecurity tasks.? Coverage depends on a defined task list and evaluation set. This project starts with the five defensive areas above and will report per-task performance rather than a single coverage claim.

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

See [`data_sources.json`](data_sources.json). The current corpus combines CISA KEV, MITRE CWE mitigations, NIST SP 800-61, NIST SP 800-53/53A control and assessment data, NIST SSDF, selected OWASP defensive cheat sheets, and the expert-reviewed subset of the MIT-licensed Cybersecurity QAA dataset. Data is downloaded into ignored local storage and packaged as a private Kaggle dataset. The OWASP material is CC BY-SA 4.0; preserve attribution and do not redistribute derived artifacts without reviewing share-alike requirements. CISA playbook PDF downloads returned HTTP 403 from this runtime and were not included.

The QAA source contains 1,563 records, but only 445 are marked expert-reviewed. The preparation script filters that subset, excludes the attack-focused theme, reserves 20% for held-out evaluation, and trains only on the remaining items. The other rows are excluded. The held-out questions and answers are placed in the Kaggle input bundle but are not read by the training loader.

The expanded Kaggle run fetches Fenrir v2.1 (Apache-2.0), Trendyol?s defensive instruction set (Apache-2.0), and the supplied threat-intelligence dataset (CC BY-NC-SA 4.0) at runtime. The CTI importer uses only its train split, applies a defensive category allowlist, excludes OpenPhish-grounded records, and protects exact held-out QAA pairs from ingestion. The CTI license restricts use to non-commercial terms and share-alike; do not redistribute the derived model without a rights review. A small capped sample from Cosmopedia?s auto_math_text, khanacademy, and openstax subsets adds general-language exposure only. See [`data_sources.json`](data_sources.json) for details.

The existing CICIDS2017, escalation, and 6G telemetry tables remain separate defensive classifier datasets; they are not language-model text. For language-model pretraining, the current selected corpus is still small compared with what a 50M model typically needs. This is an iterative, task-scoped experiment rather than a claim of broad expertise.

For language-model training, prioritize clean, licensed text and carefully reviewed defensive question/answer examples. Keep a time-based and source-grouped evaluation split. Don't mix generated training examples into evaluation. Don't train on exploit payloads, weaponization instructions, secrets, personal records, or unauthorized data.

## Training and evaluation

`train_gpt.py` creates a byte-level BPE tokenizer and trains a causal Transformer from scratch using the 50M configuration. It uses GPT-scale initialization and shuffled, non-replacement corpus passes. The staged Kaggle notebook derives steps from the prepared training chunks, targets up to two passes per phase, and caps each at 30,000 updates with early stopping. It logs progress every 50 updates, validates every 500 updates on a fixed 512-chunk sample, and evaluates generated answers on a separate held-out expert-reviewed QAA split before and after SFT. These settings give a measurable corpus exposure budget; they do not guarantee broad capability or a compute-optimal 50M model. `evaluate_tasks.py` reports answer-token loss and a lexical assertion-coverage proxy; the proxy is not correctness, so review generated examples manually. For a text model, MAE is not an appropriate target. Report validation cross-entropy/perplexity and per-task quality measures instead.

The first 100-step baseline had 48,196,608 parameters, train loss 16.91, validation loss 17.41, and 23 validation chunks. Its very high loss and tiny corpus mean it was not useful. Do not ship a checkpoint unless held-out generations are coherent, grounded, safe, and reviewed by a cybersecurity practitioner.
