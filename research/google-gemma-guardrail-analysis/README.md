# Gemma 3 Guardrail and Residual-Writer Analysis

## Scope

This folder records what can be established from the public Gemma 3 model card, the local `google/gemma-3-4b-it` checkpoint, the installed Transformers implementation, and published refusal-direction research. It does not claim access to Google's private training pipeline or internal safety code.

The target is the local BF16 Hugging Face checkpoint `google/gemma-3-4b-it`. The existing Ollama `gemma3:4b` GGUF is a separate quantized artifact and is not edited by the weight workflow.

## What Google publishes

The Gemma 3 model card says the family is multimodal and instruction-tuned variants are available. It describes data cleaning, including sensitive-data and CSAM filtering, and reports separate safety evaluations for child safety, content safety, and representational harms. It also says evaluation was conducted without safety filters. These are statements about training-data handling and evaluation; the card does not disclose a complete inference-time refusal implementation, exact training examples, or a switchable guardrail module.

Therefore, the local work can study learned refusal behavior in the released weights. It cannot reconstruct Google's private training code or assert that every refusal comes from one mechanism.

## Local architecture facts

The cached checkpoint configuration and safetensors index show a 34-layer text decoder with residual width 2,560 and feed-forward width 10,240. The checkpoint is BF16 and uses tied input/output token embeddings.

In the installed Gemma 3 implementation, each text layer has two residual branches:

1. Self-attention output projection (`self_attn.o_proj`), followed by `post_attention_layernorm`, then residual addition.
2. MLP down projection (`mlp.down_proj`), followed by `post_feedforward_layernorm`, then residual addition.

The RMSNorm implementation multiplies by `1 + weight`. That diagonal scale must be accounted for when targeting an output direction before the normalization. The multimodal projector also writes image features into the language-model width through `multi_modal_projector.mm_input_projection_weight`.

The proposed edit touches only these residual writers: the tied token embedding table, the two residual-branch output matrices in each text layer, and the final image-to-text projection. Query/key/value, gate/up projections, attention internals, vision encoder, normalization weights, and all other checkpoint tensors are left byte-for-byte unchanged.

## Research basis and limits

Arditi et al. report a refusal-related residual-stream direction in 13 open chat models, including Gemma 2B and 7B variants. Their weight-orthogonalization method projects residual-writing matrices away from a selected direction. Gemma 3 4B was not one of the tested models. The authors also describe direction selection as heuristic and evaluate both refusal behavior and benign-task KL to choose a direction.

This repository's local prompt set is a small pilot, not the paper's full benchmark or selection procedure. A positive held-out activation gap only shows that the prompt classes separate along the measured vector; it does not prove complete refusal removal or unchanged general capability. The generated checkpoint must be evaluated separately before it is treated as a replacement model.

The math targets one rank-one residual direction. That is a narrow edit by construction, but it cannot guarantee that the direction is exclusive to refusal. If benign capabilities rely on the same direction, those behaviors can change too. The original checkpoint remains the source of truth and is never overwritten.

## Local artifacts

`abliteration/abliterate_weights.py` extracts one direction in memory, applies the weight-space projection to a separate output folder under `artifacts/`, and writes a report. The direction is not used as a runtime hook. The multi-gigabyte output checkpoint is intentionally excluded from Git; scripts, prompts, analysis, and metadata are suitable for the repository.

## Sources

- [Google's Gemma 3 4B IT model card](https://huggingface.co/google/gemma-3-4b-it)
- [Google's Gemma PyTorch implementation](https://github.com/google/gemma_pytorch)
- [Transformers Gemma 3 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modular_gemma3.py)
- [Arditi et al., “Refusal in Language Models Is Mediated by a Single Direction,” NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf)
- [Authors' research code](https://github.com/andyrdt/refusal_direction)
