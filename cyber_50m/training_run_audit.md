# Kaggle Training Run Audit and Results

Audit date: 2026-11-16 UTC

## Run status

Kaggle kernel version 7 completed. Version 8 returned `CANCEL_ACKNOWLEDGED` and published no output artifacts. I submitted version 8 while version 7 was still running; Kaggle allowed both sessions to run concurrently. That duplicated GPU use. Version 7 was already farther along and is the run assessed below.

## What caused the frozen-looking logs

The earlier version 6 notebook put source downloads, pretraining, SFT, and both evaluations in one cell. Its child Python processes buffered output, so the visible log stopped after the source audit even while the cell continued.

My first follow-up, version 7, separated notebook stages and used unbuffered Python, but I missed silent setup inside `train_gpt.py`: it read the corpus, trained the tokenizer, and encoded records into chunks without printing before the first optimizer update. Version 7 eventually entered training and completed, but the missing setup logs made it look stuck. This was an observability defect in my code, not evidence of a Kaggle training-loop exception.

The OpenStax `[Errno 9] Bad file descriptor` retries came from the optional Cosmopedia fetch in version 6. That importer still recorded 500 OpenStax samples, so the retries do not explain the hours of silence. Version 7 skipped Cosmopedia entirely.

## Completed version 7 results

- Exact model size: 49,436,160 parameters; 16,000-token vocabulary.
- Pretraining: 30,000 updates; 153,087 training chunks; 1.568 passes through the chunks; 122.88 million input-token positions. Best validation loss: 1.7893 (perplexity 5.99).
- SFT: 12,500 updates; stopped after six validation checks without improvement. Best answer-validation loss: 2.2899 at step 9,500. This corresponds to about 0.457 passes through the SFT answer chunks.
- Held-out QAA evaluation (same 79 examples before and after SFT): answer-token loss 3.8288 to 3.6890; perplexity 46.01 to 40.00; lexical assertion-coverage proxy 0.772 to 0.873.
- Qualitative spot checks still showed repetitive, generic answers. The lexical proxy is not a measure of correctness, and no blinded cybersecurity expert review was performed.
- Total Kaggle session time in the log: about 9,347 seconds (2 hours 36 minutes), including data preparation, pretraining, SFT, and both evaluations.

## Interpretation

The falling pretraining validation loss is clear evidence that the model learned patterns in its training corpus. SFT produced a modest improvement on the small held-out QA set. The generated-answer samples remain too repetitive and generic to establish reliable cybersecurity assistance. Treat the result as a learning pipeline milestone, not a usable cybersecurity expert model.

The 30,000-step pretraining cap was reached, but the training set was only traversed 1.568 times. SFT stopped before one pass through its answer chunks because validation plateaued. More repeated steps alone would not establish capability; the next improvements should focus on expert-reviewed instruction data, broader independent evaluation, semantic answer scoring, and output quality.

## Local artifacts

Metrics, summaries, generated answers, and Kaggle logs are stored under the ignored local directory `data/kaggle-output/version-7/`. The checkpoint remains in Kaggle output unless separately downloaded. Raw and private training data are excluded from Git.
