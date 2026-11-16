# 50M Defensive Cyber Model Assessment

Assessment date: 2026-11-16. Kaggle artifacts are private and local data/model files are gitignored.

## Findings

The initial 100-step run did not produce a useful model. It had 48,196,608 parameters, training loss 16.91, validation loss 17.41, and only 23 validation chunks. The original PyTorch default embedding initialization also produced badly scaled logits for a tied embedding/head.

The first corrected 3,000-step retrain used deterministic full validation, GPT-scale initialization, 3,469 official text records, 395 incident records, and 355 expert-reviewed QA training records. The best validation loss was 2.910 at step 1,920 (perplexity 18.36). At step 3,000, training loss had fallen to 0.596 while validation loss rose to 3.118, so the run overfit after its best checkpoint.

On a separate 89-question expert-reviewed QA holdout, answer-token loss was 5.368 (perplexity 214.5). The lexical assertion coverage proxy was 0.528. Generated samples were repetitive and incomplete. The model therefore learned the training text but did not achieve usable cybersecurity instruction following.

## Round 2

Round 2 adds 3,553 auxiliary instruction examples from selected defensive layers of HolyCyber. The importer uses only train-split rows, requires a recorded defensive-use/public-domain/CC BY-SA license, requires `core_share >= 0.30`, removes reference appendices and fenced code, and excludes attack-knowledge, pentest, malware-analysis, and threat-intelligence layers. The dataset is third-party generated and is not reviewed row by row, so it is auxiliary; the separate expert-reviewed QAA holdout remains the main task evaluation.

Round 2 uses an 8,000-step cap, full validation every 200 steps, and early stopping after eight evaluations without improvement. Record results and any deviations below after Kaggle completes.

## Round 3: expanded sources and SFT

Kaggle kernel version 6 was submitted with a 16,000-step pretraining cap followed by up to 4,000 answer-masked SFT steps. The source audit in the notebook retained 99,837 Fenrir v2.1 rows and 4,691 defensive CTI rows after category and duplicate filtering. It sampled 500 rows each from Cosmopedia `auto_math_text`, `khanacademy`, and `openstax`. The Trendyol dataset added zero unique rows because all 53,201 normalized prompt-answer pairs duplicated Fenrir examples. The CTI loader excluded 6,306 non-allowlisted categories, 45 OpenPhish-grounded rows, and protected all 79 held-out QAA pairs from exact inclusion.

The available Round 3 log ended after the source audit at about 110 seconds and contained no optimizer-step lines. Kaggle later returned `CANCEL_ACKNOWLEDGED`; no Round 3 result artifacts were published and no failure message was supplied. The log includes OpenStax file-descriptor retries, but the source summary recorded the requested 500 OpenStax samples, so that error alone does not explain the long runtime. The notebook's single long cell and buffered child-process output made progress invisible. The CTI data is CC BY-NC-SA 4.0, which limits use to non-commercial terms and share-alike; keep any checkpoint private unless rights are reviewed.

The follow-up revision removes the optional Cosmopedia download, separates stages into distinct notebook cells, starts child Python in unbuffered mode, prints progress every 50 optimizer updates, derives update counts from actual corpus chunks, targets up to two shuffled passes per phase, caps each phase at 30,000 updates, and validates every 500 updates on at most 512 chunks. These are observability and scheduling corrections, not evidence of improved model quality. Results remain pending.

## Deployment decision

Version 7 then remained silent inside pretraining setup for over six minutes. Inspection found another missing progress layer: reading the full instruction corpus, fitting the BPE tokenizer, and chunking records ran without messages before the first optimizer step. This is a defect in the submitted training code's observability, not a demonstrated Kaggle error. The next revision logs each setup stage and reports record/token counts during tokenizer input and chunking. No version 7 training steps or quality results can be claimed from the available log.

Do not deploy or describe the current checkpoint as a cybersecurity expert. A language-model MAE target is not applicable; report cross-entropy/perplexity, task-specific metrics, human review, and safety evaluation instead. The 50M parameter budget and current corpus cannot support a claim of broad coverage across 90% of cybersecurity tasks.
