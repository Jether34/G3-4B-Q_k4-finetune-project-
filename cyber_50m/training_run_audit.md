# Kaggle Run 3: Failure and Runtime Audit

Audit date: 2026-11-16 UTC

## Finding

The available run log does **not** show a training-loop exception. It shows source preparation ending at about 110 seconds, then no optimizer-step lines. Kaggle initially reported `RUNNING`; a later status query returned `CANCEL_ACKNOWLEDGED`, with no failure message and no new output artifacts. The files listed by Kaggle are from the previous completed run. The evidence cannot establish whether tokenizer construction, training, or a service-side cancellation ended the run; the status alone is not proof of training progress.

## Root cause of the frozen-looking log

Commit `f759d5d` introduced the expanded run as one notebook cell that performs remote data downloads, tokenizer training, 16,000 pretraining updates, a generation evaluation, 4,000 SFT updates, and a second generation evaluation. Both training phases run as child Python processes without `-u`; their step `print()` calls were also buffered. Kaggle therefore had no reliable per-step notebook output while the cell was waiting for the full pipeline. This is the main cause of the unchanged visible log.

The same commit added an optional Cosmopedia stream. Its OpenStax shard emitted `[Errno 9] Bad file descriptor` and Hugging Face retry messages at roughly 110 seconds. However, the following source audit completed and recorded 500 OpenStax samples, so those messages are not evidence of a four-hour retry loop. The importer then proceeded into the long training cell.

The long run was made worse by an arbitrary 20,000-update budget (16,000 pretraining plus 4,000 SFT) and validation every 200 updates. Each validation could scan up to 2,048 pretraining chunks or 1,024 SFT chunks. The previous 8,000-update run completed and showed validation loss still improving, but that is not proof that 8,000 or 20,000 steps produce useful instruction following; its held-out generations were weak.

## Dataset and training audit

- The round-3 source audit reported 99,837 retained Fenrir rows, 4,691 defensive CTI rows, and zero unique Trendyol rows after deduplication. It excluded non-allowlisted CTI categories and OpenPhish-grounded examples. The importer separately capped Cosmopedia at 500 examples per subset.
- Local inputs include 2,737 official defensive reference records, 395 incident summaries, 318 expert-reviewed QAA training rows, 3,553 selected HolyCyber rows, and 1,522 private style-only samples. Exact QAA question/answer overlap checks found no match with the local official, incident, QAA-train, HolyCyber, or Copilot files. This check does not rule out paraphrase leakage or overlap in remote sources.
- Flow/telemetry tables are not language-model training text. They need separate supervised detection tasks and metrics.
- With effective batch 8 and context 512, 8,000 updates expose about 32.8 million input-token positions; 16,000 expose 65.5 million. These amounts are far below a common compute-optimal token budget for 50M parameters. Repeating a small corpus more times can overfit; step count by itself is not an acceptance measure.
- The existing expert-reviewed holdout is small. The assertion-coverage score is lexical and cannot establish answer correctness. A cybersecurity practitioner must review generations before any claim that the model is useful.

## Changes for the next run

1. Removed the optional runtime Cosmopedia download, the observed source of OpenStax descriptor retries.
2. Split the Kaggle notebook into separate source-preparation, pretraining, pretraining-evaluation, SFT, and SFT-evaluation cells.
3. Pass `-u` to child Python processes and emit flushed progress every 50 optimizer steps, including throughput, estimated remaining time, and corpus passes seen.
4. Derive update counts from actual training-chunk counts and effective batch size, target up to two non-replacement passes per phase, cap each phase at 30,000 updates, and retain validation-plateau early stopping.
5. Reduce validation frequency to every 500 updates and use a fixed 512-chunk validation sample.

These changes should make the next run observable and finite. They do not promise broad cybersecurity coverage or a production-ready model. The new run's step budget, data passes, losses, wall time, and held-out generations must be reported from its actual artifacts.

## Follow-up: version 7

Version 7 completed source preparation in about 90 seconds. At 90.6 seconds the notebook entered pretraining; the log had not reached the first optimizer update after more than six minutes. I initially missed that `train_gpt.py` did not print while reading the full records, fitting the tokenizer, and encoding records into chunks. `-u` cannot make silent code report progress. This is an implementation and observability gap in my follow-up, not evidence of a Kaggle platform fault. The next revision emits stage markers, tokenizer-input counts, and chunking counts before the first optimizer update. Version 7's actual failure point is unknown until Kaggle returns a terminal status or artifacts.
