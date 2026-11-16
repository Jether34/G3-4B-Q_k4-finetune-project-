# Defensive Cyber 135M adaptation

The chat output from the 50M from-scratch pilot is incoherent. It is not a prompt-tuning problem: the model learned from random initialization with only about 123M token positions. The SFT loader also did not use the Copilot style rows as answer targets.

This path starts from `HuggingFaceTB/SmolLM2-135M-Instruct`, a pretrained English instruction model, and fine-tunes all weights on a bounded defensive cybersecurity dataset. This yields an owned local fine-tuned checkpoint while retaining the pretrained English capability. It is 135M rather than exactly 150M because changing architecture size would discard the pretrained weights and return us to training from scratch.

## Prepare and submit to Kaggle

The package excludes raw Copilot conversations and message-level style samples. It includes only an abstract style preference, reviewed QAA train/eval data, source-attributed reference text, and code. Fenrir v2.1 is fetched at runtime. The CC BY-NC-SA threat-intelligence set and unlicensed Kaggle aggregates are excluded.

From the repository root, package a fresh private input bundle:

```powershell
.\.venv\Scripts\python.exe cyber_135m\package_kaggle_data.py --owner YOUR_KAGGLE_USERNAME
```

Create the private Kaggle dataset once using `kaggle datasets create` with `--dir-mode zip`; for subsequent revisions, use `kaggle datasets version`. Then push the notebook in `kaggle/cyber_135m/` with `kaggle kernels push -p kaggle/cyber_135m`. The notebook trains at most one data pass/12,000 optimizer steps, evaluates every 500 steps, and stops after a validation plateau. Kaggle's available free GPU and quota vary; check its current quota before starting.

The CSV reports train cross-entropy, validation answer cross-entropy/perplexity, data epoch fraction, tokens per second, steps per second, elapsed time, and ETA. MAE is not meaningful for next-token prediction. The held-out expert-reviewed QAA set is kept separate and scored against both the base and adapted models. Manual review of the included generation comparison is still necessary.

After downloading the Kaggle output folder into `data/kaggle-output/135m/`, choose **135M fine-tuned** in the local chat app and set the model folder to `...\best_model`. Edit guardrails in the app sidebar; changes save to `cyber_135m/guardrails.md` and can be included in the next training bundle.
