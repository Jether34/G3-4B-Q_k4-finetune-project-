# A From-Scratch 49.4M-Parameter Transformer for Defensive Cybersecurity: Data Curation, Training, and Preliminary Evaluation

**Jether Garque**

*Preliminary technical report - 16 November 2026 - expanded Kaggle run completed*

## Abstract

This report documents the design, data curation, training, and preliminary evaluation of a small decoder-only Transformer intended for defensive cybersecurity assistance. The latest completed Kaggle run used 30,000 pretraining updates and 12,500 answer-masked supervised fine-tuning (SFT) updates on a 49,436,160-parameter model. Pretraining validation loss reached 1.789 (perplexity 5.99). On the 79-question expert-reviewed holdout, answer-token loss improved from 3.829 before SFT to 3.689 after SFT; lexical assertion-coverage proxy increased from 0.772 to 0.873. Despite these measurable improvements, inspected generations remained repetitive and generic. The results support that the model learned corpus patterns, but do not establish reliable cybersecurity assistance. Mean absolute error is not an appropriate objective for this generative task. This remains a research prototype without a tested runtime guardrail.

## 1 Introduction

Language models can help practitioners summarize security guidance, triage alerts, prioritize vulnerabilities, and structure incident response. Models at this scale are often costly to train from scratch, so this project studies whether a compact model can be built and inspected using a free Kaggle GPU. The work targets ownership and accessibility of the model artifacts and training code, while keeping the initial application scope defensive.

The model is a 49.4-million-parameter causal Transformer. We call it a “50M” model as a size class, not as an exact parameter count. It is trained from random initialization; it does not inherit general language ability from a larger pretrained model. This distinction matters because a small parameter budget and a relatively constrained training corpus limit what can be learned. A model’s parameter count alone does not establish its knowledge, competence, or safety.

The project has evolved through several training runs. The completed expanded run added Fenrir and defensive threat-intelligence examples, trained through 30,000 pretraining updates, and then applied answer-masked SFT with validation-based early stopping. It improved held-out token metrics modestly, but its free-form generations remained repetitive, so we do not claim effective task performance.

This report makes four contributions. First, it specifies a reproducible from-scratch model configuration and the changes made to its initialization and evaluation code. Second, it describes source-level data cleaning, split construction, and licensing constraints. Third, it reports the completed pilot results using both language-model loss and a small held-out task evaluation. Fourth, it distinguishes training-data filtering from runtime safety controls and states what evidence is still required before practical use.

## 2 Background and Scope

The Transformer uses self-attention to process token sequences and remains a common basis for language models [Vaswani et al., 2017]. Here, a decoder-only network is optimized with a next-token prediction objective. Fine-tuning a model on prompt/answer examples can teach response formats and task behavior, but it cannot compensate automatically for limited pretraining, poor examples, or inadequate evaluation.

Training compute and token exposure affect model quality alongside parameter count. Compute-optimal scaling work emphasizes allocating sufficient data as model size grows [Hoffmann et al., 2022]. This project’s objective is constrained by Kaggle’s free GPU session rather than a compute-optimal training budget. Its results should therefore be interpreted as a small-scale engineering study, not as an estimate of the best achievable performance for a 50M model.

The target tasks are defensive: interpreting security advisories, vulnerability prioritization, secure configuration, security monitoring and alert triage, incident response, threat-intelligence analysis, security awareness, and governance. Network-flow and telemetry archives are not treated as language-model prose; those tasks require separately labeled data and task-specific classifiers. The project does not establish coverage of “90% of cybersecurity tasks.” No operational task taxonomy or representative benchmark has been completed to support such a percentage.

The project initially considered a mean absolute error (MAE) target of 0.001. MAE is not a suitable primary measure for autoregressive language generation. This work instead reports token cross-entropy and perplexity, held-out answer-token loss, a lexical assertion-coverage proxy, and qualitative generations. The proxy does not measure factual correctness, completeness, or safe behavior.

## 3 Model Architecture

The model is a decoder-only Transformer implemented in PyTorch. Its attention mask prevents each position from attending to future tokens. The implementation uses pre-normalized Transformer encoder blocks with a causal mask, GELU feed-forward activations, learned positional embeddings, a final layer normalization, and a linear language-model head tied to the token embedding matrix.

The configured model has 13 layers, hidden width 512, eight attention heads, feed-forward width 2,048, maximum sequence length 512, and a 16,000-token byte-level BPE vocabulary. The exact configured parameter count is 49,436,160, including embeddings and positional embeddings. The code uses GPT-scale initialization for token, position, attention, and feed-forward weights; residual output projections receive depth-scaled initialization. This replaced an earlier default initialization that produced poorly scaled logits with tied embeddings.

The tokenizer is trained on the training corpus for each from-scratch run. The small context window limits long document reasoning and truncates long prompts or answers in task evaluation. No retrieval system, tool-use loop, or external knowledge refresh mechanism is included in the model itself.

## 4 Data Sources and Curation

### 4.1 Existing defensive corpus

The base corpus combines public defensive references and structured instruction examples. Sources include CISA Known Exploited Vulnerabilities metadata, MITRE CWE descriptions and mitigations, NIST incident-response and security-control publications, the NIST Secure Software Development Framework, selected OWASP Cheat Sheets, incident summaries, and a filtered subset of the Cybersecurity Question-Answer-Assertion (QAA) dataset. Data was downloaded or transformed locally and packaged into a private Kaggle input dataset; raw datasets, private conversations, checkpoints, and generated corpora are excluded from Git.

The QAA source contains 1,563 records, but the upstream card reports that only a subset has expert review. The project keeps rows marked expert-reviewed, removes the attack-focused theme and low-quality entries, then reserves a separate holdout. The latest split contains 397 eligible records: 318 training examples and 79 held-out examples. The earlier 3,000-step run used a previous split with 89 held-out questions, so metrics across those two splits are not strictly comparable.

The auxiliary HolyCyber import retains 3,553 records from selected defensive domains. It uses train-split rows, applies per-row license and domain checks, requires a minimum core-content share, removes reference appendices and fenced code, and excludes attack-knowledge, penetration-testing, malware-analysis, and threat-intelligence layers. The source is generated and is not reviewed row by row; it is treated as auxiliary data rather than ground truth.

### 4.2 Sources added for the expanded run

The current run fetches and cleans additional sources at Kaggle runtime. The audit reported 99,837 retained Fenrir v2.1 system/user/assistant examples after short-record and duplicate filtering. The source card describes the set as defensive instruction-tuning data and lists an Apache-2.0 license [Kiraz, 2026a].

The supplied threat-intelligence repository contributes 4,691 examples after filtering. The importer uses the training split only, allows defensive analysis and response categories, excludes non-allowlisted categories, removes records grounded in OpenPhish, and protects exact question/answer matches with the QAA holdout. The audit excluded 6,306 records by category and 45 OpenPhish-grounded records. The upstream dataset is licensed CC BY-NC-SA 4.0, with non-commercial, attribution, and share-alike terms; the model and its distribution therefore require a separate rights review before any release or commercial use [Reloading, 2026].

Cosmopedia was considered during an earlier run, but the OpenStax shard produced file-descriptor retries. The completed version 7 run skipped the optional Cosmopedia download; it contributed no training examples. This removed an unnecessary network dependency from the cyber-focused corpus.

Trendyol’s defensive instruction set was also fetched and normalized. Its 53,201 rows contributed zero unique records in the current source audit because the normalized user/assistant pairs exactly duplicated Fenrir examples. Deduplication prevented a duplicated source from inflating the training set. The upstream card reports Apache-2.0 [Trendyol Security Team, 2025].

### 4.3 Data limitations and licensing

The corpus mixes authoritative standards and government materials with generated instruction examples. A source’s presence does not imply that every derived answer is correct, current, or representative. The current data loader performs schema checks, text cleanup, license and category filtering where metadata is available, and exact duplicate removal. It does not include comprehensive expert review of all retained examples, semantic deduplication, or an adversarial prompt audit.

Some included sources carry attribution and share-alike terms, and the CTI dataset explicitly carries a non-commercial restriction. The project keeps the data bundle and Kaggle notebook private. This report does not make a legal conclusion about the terms governing derivative model weights. Model publication should wait for a source-by-source licensing review.

## 5 Training Procedure

### 5.1 From-scratch pretraining

For each run, the tokenizer is learned from the training portion of the text corpus. Documents and instruction records are serialized as text, tokenized, concatenated, and split into chunks of at most 512 tokens plus the next-token target. The model minimizes next-token cross-entropy using AdamW, gradient accumulation, mixed precision on the Kaggle GPU, and gradient clipping. Validation is run at configured intervals; the best checkpoint by validation loss is saved. Early stopping is enabled in the expanded notebook.

The initial pilot was capped at 100 steps. After initialization and validation code were corrected, a 3,000-step run used full validation every 200 steps and exhibited a widening train/validation gap. A subsequent 8,000-step run improved text validation but remained weak on held-out questions. The completed expanded run derived its budget from prepared chunks, capped pretraining at 30,000 updates, and used a 512-chunk validation sample every 500 updates. It processed 122.88 million input-token positions, equivalent to 1.568 passes through 153,087 training chunks. The 30,000 cap prevented the planned two full passes.

### 5.2 Answer-masked supervised fine-tuning

The loss is applied to answer tokens while prompt tokens and padding are masked. The completed run allowed up to two passes through SFT chunks, capped at 30,000 updates, used AdamW at a learning rate of 5e-5, and validated every 500 updates on a fixed sample. Validation plateau triggered early stopping at step 12,500; the best answer-validation loss, 2.2899, occurred at step 9,500. The run processed about 0.457 passes through its SFT answer chunks.

### 5.3 Reproducibility and privacy

The project’s scripts, notebook, configuration, and source manifest are maintained in the associated GitHub repository. The private Kaggle dataset contains the train/evaluation files and a redacted, Human-only Copilot conversation sample used only as a limited style signal. The original conversation export and the redacted file are not committed to the repository. Training artifacts remain in private Kaggle output until reviewed.

## 6 Evaluation Protocol

The principal task evaluation uses the 79-question expert-reviewed QAA holdout. These examples are excluded from the QAA training split and from exact matches in the newly fetched instruction sources. For each question, the evaluator computes the mean cross-entropy of reference answer tokens conditioned on the question and generates a greedy completion. The answer-token perplexity is the exponential of mean answer-token loss.

The evaluator also reports lexical assertion coverage. It measures whether the generated answer shares at least one non-stopword token with each reference assertion. It is a rough lexical overlap proxy: a model can receive credit for a shared term while being factually wrong, or fail to receive credit for a correct paraphrase. It must not be described as accuracy.

Qualitative review checks whether answers are coherent, relevant, sufficiently complete, and defensively framed. It is not a blinded expert study, and no statistically powered comparison has been performed. The expanded run evaluates the same held-out QAA set before and after SFT to permit a within-run comparison, and the completed run shows a modest loss improvement; its generations remained weak.

## 7 Results

### 7.1 Completed training runs

| Run | Training result | Held-out task result | Interpretation |
|---|---|---|---|
| Initial baseline | 48,196,608 parameters; 100 steps; train loss 16.91; validation loss 17.41; 23 validation chunks | Not a meaningful task evaluation | Undertrained baseline with poorly scaled tied-embedding initialization. |
| Corrected run 1 | Best validation loss 2.910 at step 1,920 (PPL 18.36); at step 3,000 train loss 0.596 and validation loss 3.118 | 89 questions; answer loss 5.368; PPL 214.5; assertion coverage proxy 0.528 | Training loss continued to fall after validation worsened, indicating overfitting; generations were repetitive and incomplete. |
| Expanded run 2 | 49,436,160 parameters; 16k vocabulary; 11,155 training chunks; 564 validation chunks; 8,000 steps; best validation loss 1.6998 (PPL 5.47) | 79 questions; answer loss 5.147; PPL 171.94; assertion coverage proxy 0.620 | Text validation improved, but task metrics remained weak and generations were incoherent or incomplete. |
| Expanded run 3 (Kaggle v7) | 49,436,160 parameters; 30,000 pretraining steps (best validation loss 1.789; PPL 5.99); 12,500 SFT steps (best validation answer loss 2.290 at step 9,500; early-stopped after six evaluations without improvement) | 79 questions; answer loss 3.829 to 3.689; PPL 46.01 to 40.00; assertion-coverage proxy 0.772 to 0.873 | Clear learning signal and modest held-out metric gains; generations remained repetitive and generic, so task competence is not established. |

Runs 1, 2, and 3 use different corpus compositions and validation samples, so their validation losses are not a controlled comparison. Within run 3, pre- and post-SFT task metrics use the same held-out questions. Answer-token loss fell by about 3.6%; perplexity fell by about 13%; the lexical assertion proxy rose by 10.1 percentage points. The proxy can reward superficial word overlap and cannot establish factual correctness.

### 7.2 Qualitative assessment

The run-3 generations improved on the lexical proxy but remained repetitive and often generic on spot-checked held-out prompts. For example, answers frequently repeated a question?s framing instead of offering specific, complete guidance. This is qualitative evidence only; no blinded expert review or statistically powered comparison was performed.

## 8 Discussion

The completed runs point to both data and configuration constraints. The original initialization defect and tiny 100-step baseline were configuration and training-schedule problems. Correcting initialization and increasing training improved mixed-corpus loss. Yet the later checkpoint still performed poorly on held-out task questions and free-form generation. This suggests that lower text loss alone is not the missing ingredient. The limited expert-reviewed QA count, generated auxiliary data, short context window, small parameter budget, and mismatch between plain text pretraining and answer generation all constrain task behavior.

Answer-masked SFT modestly improved held-out answer-token metrics, but did not produce clearly useful generations in the inspected examples. The SFT validation loss reached its minimum of 2.2899 at step 9,500; after six evaluations without improvement, training stopped at step 12,500, having processed about 0.457 passes through its answer chunks. Early stopping limited further overfitting, but the model did not complete one pass over the SFT chunk set. Larger expert-reviewed evaluation and semantic scoring are needed before capability claims.

The system has no tested runtime guardrail. Data filters and refusal examples reduce some unwanted training exposure, but they cannot guarantee that a model will refuse harmful requests. Prior work has shown that refusal behavior in language models can be brittle under changes to model internals [Arditi et al., 2024]. For this prototype, a practical safety claim would require a separate, policy-defined test set, manual review of outputs, and a runtime control layer tested independently from model training.

The project does not support a claim that it covers 90% of cybersecurity tasks. Such a claim would first require a defined task universe, a representative test set for each task, and a coverage rule. It also does not support MAE = 0.001 as a meaningful language-model success criterion. A better next milestone is measurable defensive performance on a bounded suite, including vulnerability triage, incident response, secure configuration, and alert analysis, together with a refusal and hallucination assessment.

## 9 Limitations

This study has major limitations. First, the model is small compared with current instruction-following systems and was trained from scratch on limited compute. Second, data quality is heterogeneous; some instruction examples are generated or not reviewed row by row. Third, the task evaluation contains only 79 questions. Fourth, the lexical assertion proxy is not semantic scoring, and the qualitative review is not blinded or independently replicated. Fifth, cybersecurity knowledge changes rapidly; the model has no retrieval or update system. Sixth, behavior under adversarial prompts and harmful-completion rates were not comprehensively measured. Finally, SFT stopped before one pass through its answer chunks, and no blinded practitioner study was performed.

## 10 Responsible Use and Safety

This model should be treated as an experimental research artifact. It is not suitable for autonomous security operations, incident-response decisions, exploit development, or unsupervised advice to third parties. Defensive training-data filtering does not constitute a robust safety guarantee. Any generated recommendation should be independently verified against current authoritative sources by a qualified human.

The threat-intelligence training data has CC BY-NC-SA 4.0 terms, and other sources also have attribution or share-alike conditions. The checkpoint should remain private pending licensing review. User conversation data used for style adaptation must remain private and must not appear in public artifacts. No raw conversation content is reproduced in this report.

## 11 Conclusion

This project demonstrates a working pipeline for creating a 49.4M-parameter decoder-only Transformer from scratch, curating defensive cybersecurity data, training on a Kaggle GPU, and evaluating held-out expert-reviewed questions. The completed expanded run learned the training corpus and showed modest improvements in held-out answer-token metrics after SFT. However, generated answers remained repetitive and generic, and semantic correctness and safety were not comprehensively evaluated. The evidence supports a research prototype, not a capable or safe cybersecurity expert. Future work should expand expert-reviewed task evaluations, test refusal behavior, add retrieval from current sources, and resolve dataset-license and model-distribution questions before releasing weights.

## Acknowledgements

The author acknowledges the maintainers of the public datasets and standards described in this report. Any errors in the project’s data handling, implementation, or interpretation remain the author’s responsibility.

# Appendix A: Model and Training Configuration

| Setting | Value |
|---|---:|
| Architecture | Decoder-only causal Transformer |
| Parameters | 49,436,160 |
| Layers | 13 |
| Hidden size | 512 |
| Attention heads | 8 |
| Feed-forward size | 2,048 |
| Maximum context | 512 tokens |
| Vocabulary | 16,000 byte-level BPE tokens |
| Embeddings | Token embeddings tied to language-model head |
| Pretraining objective | Next-token cross-entropy |
| Expanded pretraining | 30,000 steps; 1.568 passes through 153,087 training chunks; best validation loss 1.789 |
| SFT | 12,500 steps; early stopping after six validation non-improvements; best answer-validation loss 2.290 at step 9,500 |
| SFT learning rate | 5×10⁻⁵ |
| Accelerator | Kaggle Tesla T4 |

# Appendix B: Data Filtering Summary

| Source | Retained rows in the expanded run | Filtering or role |
|---|---:|---|
| Fenrir v2.1 | 99,837 | Defensive prompt/answer rows; short and duplicate rows removed. |
| Threat Intelligence Dataset | 4,691 | Train split; defensive category allowlist; OpenPhish-grounded examples excluded. |
| Cosmopedia | 0 | Optional download skipped in the completed run after earlier OpenStax retries. |

| Trendyol cybersecurity | 0 unique | 53,201 normalized rows matched Fenrir exactly and were deduplicated. |
| HolyCyber | 3,553 | Selected defensive layers; auxiliary generated examples. |
| Expert-reviewed QAA | 318 train / 79 holdout | Holdout kept outside training and exact-match filtering. |

The counts are the current run’s import audit, not counts of independently authored or expert-validated facts. Row totals do not equal unique knowledge units or coverage of operational cybersecurity tasks.

# Appendix C: Evaluation Metric Definitions

**Token cross-entropy** is the mean negative log-likelihood of target tokens. **Perplexity** is the exponential of mean cross-entropy and represents uncertainty under the modeled token distribution; it is not a measure of answer correctness. **Held-out answer-token loss** evaluates reference answer tokens conditioned on each question. **Assertion coverage proxy** counts reference assertions with at least one content-word overlap in the generated answer. The proxy is lexical only and must not be interpreted as semantic correctness or accuracy.

# Appendix D: Run Chronology

1. **Initial baseline:** 100 steps; 48,196,608 parameters; train/validation losses 16.91/17.41; 23 validation chunks.
2. **Corrected run 1:** 3,000-step cap; best validation loss 2.910 at step 1,920; later overfit; 89-question task holdout.
3. **Expanded run 2:** added selected HolyCyber defensive data; 8,000 steps; best validation loss 1.6998; 79-question task holdout; generation quality remained poor.
4. **Expanded run 3 (Kaggle v7):** Fenrir and defensive-only CTI; no Cosmopedia rows; 30,000 pretraining steps and 12,500 answer-masked SFT steps. Best pretraining validation loss was 1.789; best SFT validation loss was 2.290. On the same 79-question holdout, answer loss improved from 3.829 to 3.689 after SFT, while generated answers remained repetitive. Version 7 completed. Duplicate version 8 later returned `CANCEL_ACKNOWLEDGED` and published no output artifacts.

# Appendix E: Reproducibility and Artifact Availability

Training scripts, data-source manifest, model configuration, and Kaggle notebook are in the project repository. Source datasets, derived JSONL files, private Copilot style examples, Kaggle credentials, and model checkpoints are excluded from Git. The private Kaggle notebook fetches several public datasets at runtime and saves logs, metrics, generations, and checkpoints to private Kaggle output. The expanded run is not reproducible solely from the repository unless the external dataset versions and licenses remain accessible and unchanged; source cards and import filters are documented in the project manifest.

# References

Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., and Nanda, N. (2024). Refusal in Language Models Is Mediated by a Single Direction. *Advances in Neural Information Processing Systems 38*. https://arxiv.org/abs/2406.11717

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., et al. (2022). Training Compute-Optimal Large Language Models. *Advances in Neural Information Processing Systems 35*. https://arxiv.org/abs/2203.15556

Kiraz, A. (2026a). Cybersecurity Dataset Fenrir v2.1. https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1

National Institute of Standards and Technology. (2025). *Incident Response Recommendations and Considerations for Cybersecurity Risk Management: A CSF 2.0 Community Profile* (SP 800-61 Rev. 3). https://csrc.nist.gov/pubs/sp/800/61/r3/final

National Institute of Standards and Technology. (2022). *Secure Software Development Framework (SSDF) Version 1.1: Recommendations for Mitigating the Risk of Software Vulnerabilities* (SP 800-218). https://csrc.nist.gov/pubs/sp/800/218/final

OWASP Foundation. (n.d.). OWASP Cheat Sheet Series. CC BY-SA 4.0. https://owasp.org/projects/cheat-sheet-series

Reloading. (2026). Threat Intelligence Dataset. CC BY-NC-SA 4.0. https://github.com/reloading01/threat-intelligence-dataset

Rowden, J. (2023). Cybersecurity Question-Answer-Assertion (QAA) Dataset. MIT License. https://huggingface.co/datasets/Rowden/CybersecurityQAA

Trendyol Security Team. (2025). Trendyol Cybersecurity Defense Instruction-Tuning Dataset v2.0. Apache-2.0. https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems 30*. https://arxiv.org/abs/1706.03762
