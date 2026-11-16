# Defensive cybersecurity adaptation based on SmolLM2-135M-Instruct

This project performs full-weight supervised fine-tuning of `HuggingFaceTB/SmolLM2-135M-Instruct` with defensive cybersecurity examples. The adapted model is about 135M parameters; the underlying pretrained model is Apache-2.0. Keep upstream attribution, configuration, and model provenance with every export.

The starting model supplies English fluency and instruction behavior from large-scale pretraining. The adaptation data is restricted to the existing expert-reviewed CybersecurityQAA training split, selected license-screened HolyCyber records, and the defensive Fenrir v2.1 dataset card's Apache-2.0 release. The 79-item reviewed QAA evaluation set is protected from training. Threat-intelligence data marked CC BY-NC-SA 4.0, raw Copilot messages, and unlicensed community aggregates are excluded.

The training interface logs per-step training cross-entropy, examples-seen epoch fraction, tokens per second, steps per second, elapsed time, ETA, and held-out answer loss/perplexity. MAE is not applicable to autoregressive language modeling. Human review remains necessary: a lower validation loss is not proof of factual correctness, safe behavior, or broad task coverage.

This is a small research adaptation, not a reliable incident commander or security expert. Verify answers against authoritative, current sources before acting.
