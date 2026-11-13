# Kaggle preparation: Gemma 3 4B IT

## What is here

- `Gemma3-4B-IT-Kaggle-Vision-QLoRA.ipynb`: prepared training and inspection notebook. It loads the instruction-tuned checkpoint, reports model configuration, and follows the official Unsloth 4-bit LoRA/QLoRA vision workflow.
- `Kaggle-Gemma3-4B-Vision.ipynb` and `.py`: downloaded official Unsloth reference files, kept unchanged.

The customized notebook retains the reference notebook's LaTeX OCR example dataset. That demonstrates the pipeline; replace the dataset and prompt formatting with your own task data before training for your intended use.

## Kaggle setup

1. Upload/import `Gemma3-4B-IT-Kaggle-Vision-QLoRA.ipynb` into Kaggle.
2. In Notebook settings, enable a GPU accelerator and Internet.
3. On Hugging Face, review and accept the Gemma terms for [`unsloth/gemma-3-4b-it`](https://huggingface.co/unsloth/gemma-3-4b-it). Create a read token and add it in Kaggle as a secret named `HF_TOKEN`.
4. Run the notebook from the top. Its installation cell installs PyTorch/CUDA wheels, Unsloth, Transformers, TRL, and supporting packages into the Kaggle runtime. Kaggle's Linux GPU environment is the target; these CUDA packages are intentionally not installed on this Windows laptop.
5. The notebook saves the LoRA adapter to `gemma_3_lora`. Save/download that output from the Kaggle session before it ends.

## What “reverse engineering” can inspect

The notebook can inspect the released checkpoint's configuration, tokenizer/processor, prompt template, quantization/loading setup, and observed outputs on a fixed evaluation set. It cannot reconstruct the model's original training corpus or undocumented training process from weights. Keep a held-out evaluation set separate from the fine-tuning data so baseline and adapted behavior can be compared.

## Checkpoint relationship

The local Ollama package is `gemma3:4b`, reported as Gemma 3 3.9B, Q4_K_M, with a vision projector. Its manifest identifies an `it` (instruction-tuned) quantized package. Kaggle fine-tunes the corresponding Hugging Face checkpoint, not the Ollama GGUF blobs. The resulting LoRA adapter is a separate artifact; converting it back to an Ollama-compatible model is a later export step.

## Dependency source

The prepared notebook uses the package installation cell from the official Unsloth Kaggle notebook. Package versions in that cell are pinned where the reference pins them; the Kaggle runtime and GPU driver provide the platform-specific CUDA environment.
