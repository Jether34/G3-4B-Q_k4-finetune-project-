# Next retraining plan: fluent base, defensive adaptation

## What the last run tells us

The current chat is a 49,436,160-parameter Transformer initialized from random weights. Its 30,000 pretraining updates exposed it to about 123 million token positions (roughly 2.5 positions per parameter); SFT then saw about 0.46 passes over its answer chunks. The decreasing validation loss only shows that it fit the sampled corpus. The screenshot's incoherent answer confirms that it did not learn usable language generation. Making the same run 150M parameters would make this worse unless training tokens and compute rose substantially.

There is also a specific tone-data gap: the local Copilot export produced 1,522 redacted Human-only messages (median 8 words). The SFT loader never reads `copilot_style_private.jsonl`, so these messages did not tune answer style. Even as raw-text pretraining, human messages alone do not teach the model how to answer. We will express the preference as a concise, direct, conversational system instruction, while keeping the private CSV off Kaggle. The app's editable guardrails are in `guardrails.md`.

## Model choice

Start from Hugging Face's `HuggingFaceTB/SmolLM2-135M-Instruct` (about 135M parameters, Apache-2.0), then fine-tune all model weights with backpropagation. The base model was trained on 2 trillion tokens and already has English generation and instruction behavior; adapt it to the curated defensive corpus instead of trying to relearn English from a few hours of free GPU time. It is close to 150M, but is a much stronger choice than a randomly initialized exact-150M model for this compute budget. Preserve the upstream model card and weight provenance.

Kaggle free GPU access and session quotas change. Before a long run, inspect the account's current quota and the notebook's assigned GPU; do not infer available hours from old runs. Save resumable checkpoints and metrics to `/kaggle/working` and the final output to the notebook artifact.

## Data selection

Do not try to download every dataset found online. Size does not guarantee quality, provenance, safe scope, or permission to train. Use this bounded, auditable set:

| Source | Role | License / handling |
|---|---|---|
| SmolLM2-135M-Instruct | English fluency and instruction following are inherited from the base model; no separate giant English corpus is needed for the first adaptation | Apache-2.0 model card; retain attribution and model provenance |
| Existing expert-reviewed CybersecurityQAA subset | Small held-out task benchmark; its 79-item holdout stays out of training | MIT; train only the reviewed training split |
| Fenrir v2.1 defensive instruction set | Main supervised defensive task data after schema, duplicate, length, contamination, and quality checks | Dataset card states Apache-2.0; verify current source card and preserve notice |
| Official NIST CSF 2.0, SP 800-61 Rev. 3, SSDF, selected 800-53 controls | Governance, incident response, recovery, and secure development knowledge; convert source references into grounded tasks | Keep source URLs and attribution; NIST publishes machine-readable CSF reference data |
| CISA KEV and selected CISA advisories | Remediation and incident-response context; prefer mitigation and response fields | Store source/date and retrieval URL; refresh facts at use time |
| MITRE CWE mitigation fields | Weakness descriptions, mitigations, and safe coding guidance | Use mitigation-oriented fields; retain source attribution |
| Selected OWASP Cheat Sheets | Application security, authentication, logging, and secure coding | CC BY-SA 4.0; attribution/share-alike terms apply to redistributed adapted material |
| Existing Copilot CSV | Derive only an abstract local style profile; do not upload raw conversations | Private data; no raw rows in training bundle or Git |

The existing threat-intelligence set is CC BY-NC-SA 4.0 and remains excluded from a redistributable model bundle unless its noncommercial/share-alike obligations are accepted. The large community `oi-uae/cyber-security` aggregate is research-only, includes offensive material, and has mixed upstream rights, so it is excluded. Random Kaggle tables (CICIDS, UNSW-NB15, etc.) are telemetry/classification data, not prose or chat; keep them for separate supervised classifiers.

## Training phases and live reporting

1. Audit sources, remove exact/near duplicates, verify licenses and splits, and protect all benchmark questions from training.
2. Run a short smoke session first: load the exact base and data, train a few steps, save and reload a checkpoint, and generate fixed prompts. Stop if generations become less coherent.
3. Fine-tune the 135M instruct model on the defensive SFT mixture with a low learning rate, answer-only loss, mixed precision, gradient accumulation, validation-based early stopping, and resumable checkpoints. Keep the expert-reviewed holdout fixed.
4. Log every step to stdout and CSV: global step, epoch fraction, running train cross-entropy, tokens/second, steps/second, elapsed time, ETA, and last validation cross-entropy/perplexity. Show an update at least every second when training is active; validate every fixed number of steps. MAE is not a language-model quality metric.
5. Compare base vs tuned model on the same held-out defensive tasks, with generation samples reviewed manually. Report loss/perplexity and rubric scores for correctness, groundedness, clarity, and safe behavior. Do not claim broad coverage from loss alone.

## Guardrails and style

`guardrails.md` is editable in the chat sidebar and is inserted as an inference system instruction. The same policy must be attached to SFT examples and evaluated with defensive and refusal cases. Prompt guardrails are not a guarantee; they do not replace model evaluation or access controls.

The Copilot export has many short, direct user messages, not instruction-response examples. Use an abstract preference such as “direct, concise, plainspoken; avoid filler; explain uncertainty,” not copied private dialogue. This can guide responses but does not warrant a claim that the model learned a measured personal style from SFT.
