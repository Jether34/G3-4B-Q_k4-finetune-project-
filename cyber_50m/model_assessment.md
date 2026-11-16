# 50M Defensive Cyber Model Assessment

Assessment date: 2026-11-16. Kaggle artifacts are private and local data/model files are gitignored.

## Findings

The initial 100-step run did not produce a useful model. It had 48,196,608 parameters, training loss 16.91, validation loss 17.41, and only 23 validation chunks. The original PyTorch default embedding initialization also produced badly scaled logits for a tied embedding/head.

The first corrected 3,000-step retrain used deterministic full validation, GPT-scale initialization, 3,469 official text records, 395 incident records, and 355 expert-reviewed QA training records. The best validation loss was 2.910 at step 1,920 (perplexity 18.36). At step 3,000, training loss had fallen to 0.596 while validation loss rose to 3.118, so the run overfit after its best checkpoint.

On a separate 89-question expert-reviewed QA holdout, answer-token loss was 5.368 (perplexity 214.5). The lexical assertion coverage proxy was 0.528. Generated samples were repetitive and incomplete. The model therefore learned the training text but did not achieve usable cybersecurity instruction following.

## Round 2

Round 2 adds 3,553 auxiliary instruction examples from selected defensive layers of HolyCyber. The importer uses only train-split rows, requires a recorded defensive-use/public-domain/CC BY-SA license, requires `core_share >= 0.30`, removes reference appendices and fenced code, and excludes attack-knowledge, pentest, malware-analysis, and threat-intelligence layers. The dataset is third-party generated and is not reviewed row by row, so it is auxiliary; the separate expert-reviewed QAA holdout remains the main task evaluation.

Round 2 used an 8,000-step cap with validation every 200 steps and early stopping after eight non-improving evaluations. Its 79-question held-out generation quality remained weak.

## Round 3: expanded sources and SFT

Kaggle version 6 used the expanded data importer and a single long notebook cell. Its source audit retained 99,837 Fenrir rows and 4,691 defensive CTI rows; Trendyol contributed zero unique examples because its 53,201 normalized pairs duplicated Fenrir. The CTI filter excluded 6,306 non-allowlisted rows and 45 OpenPhish-grounded rows. The optional Cosmopedia download was removed in version 7 after OpenStax descriptor retries.

Version 7 completed with 49,436,160 parameters. Pretraining ran 30,000 updates over 153,087 chunks, saw 1.568 corpus passes, and reached best validation loss 1.7893 (perplexity 5.99). SFT stopped at 12,500 updates after six validation checks without improvement; its best answer-validation loss was 2.2899 at step 9,500. SFT saw about 0.457 passes through its answer chunks.

On the same 79-question holdout, answer-token loss moved from 3.8288 before SFT to 3.6890 after SFT; perplexity moved from 46.01 to 40.00; lexical assertion coverage moved from 0.772 to 0.873. These metrics show a modest improvement, but the lexical proxy is not correctness. Spot-checked outputs remained repetitive and generic. This does not establish useful cybersecurity task performance.

Version 8 was submitted before version 7 stopped, so both sessions ran concurrently. Version 7 completed; version 8 later returned `CANCEL_ACKNOWLEDGED` without output artifacts. This duplicate run was unnecessary GPU use.

See [`training_run_audit.md`](training_run_audit.md) for the log diagnosis and detailed run counts. The complete local metrics and generated-answer CSV are under the ignored `data/kaggle-output/version-7/` directory. The CTI license is CC BY-NC-SA 4.0; keep model artifacts private pending a source-by-source rights review.

## Deployment decision

Version 7 then remained silent inside pretraining setup for over six minutes. Inspection found another missing progress layer: reading the full instruction corpus, fitting the BPE tokenizer, and chunking records ran without messages before the first optimizer step. This is a defect in the submitted training code's observability, not a demonstrated Kaggle error. The next revision logs each setup stage and reports record/token counts during tokenizer input and chunking. No version 7 training steps or quality results can be claimed from the available log.

Do not deploy or describe the current checkpoint as a cybersecurity expert. A language-model MAE target is not applicable; report cross-entropy/perplexity, task-specific metrics, human review, and safety evaluation instead. The 50M parameter budget and current corpus cannot support a claim of broad coverage across 90% of cybersecurity tasks.
