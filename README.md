# Gemma 3 4B Q4_K_M Fine-Tuning Project

A local control center and experimentation workspace for Gemma 3 4B. The project currently provides an Ollama-backed chat dashboard, a model-structure explorer, and a Kaggle QLoRA starter notebook. A production MCP server, trained project-specific adapter, and instrumented activation viewer are future work.

## Project status

| Area | Status | What it does |
| --- | --- | --- |
| Local inference dashboard | Implemented | Text/image prompts, streamed replies, model selection, generation controls, and Ollama timing/token statistics. |
| Layer explorer | Implemented | 2D overview and rotatable 3D selected-block view with configured dimensions and sampled neurons. |
| Fine-tuning | Starter workflow | Kaggle notebook demonstrates QLoRA with a LaTeX OCR sample task; replace its demo data before training for another task. |
| Adapter deployment | Not implemented | A trained adapter still needs evaluation and conversion/merging for the chosen Ollama runtime. |
| MCP tools | Planned | No MCP server is included yet. Tools should be narrow, separately configured, and confirmed visibly to the user. |
| Live activation capture | Not implemented | Ollama's generation API does not return per-neuron activations or attention tensors. |

## Repository contents

- `start_control_center.py` starts local Ollama (if needed) and the dashboard server.
- `visualizer/` contains the browser UI, its local Python API bridge, and a vendored Three.js module.
- `kaggle/Gemma3-4B-IT-Kaggle-Vision-QLoRA.ipynb` is the prepared training/inspection notebook.
- `kaggle/Kaggle-Gemma3-4B-Vision.ipynb` and `.py` are the upstream Unsloth reference files kept for comparison.
- `kaggle/KAGGLE-START-HERE.md` contains the Kaggle setup notes.

The Ollama manifest and model blobs stay local and are excluded from Git because the language and vision files are multi-gigabyte artifacts. Install the model locally with `ollama pull gemma3:4b` instead of expecting the repository checkout to contain its weights. [MODEL_SHA256SUMS.txt](MODEL_SHA256SUMS.txt) records the SHA-256 checksums of the exact local model and vision-projector blobs used when this project snapshot was prepared, along with their Ollama manifest identifiers. The `gemma3:4b` tag can move; the recorded hashes describe the local snapshot, not a guarantee about every future pull of that tag.

## Requirements

- Windows, macOS, or Linux with Python 3.10 or newer.
- Ollama installed and available on `PATH` (or installed in Ollama's standard Windows location).
- `gemma3:4b` pulled locally for the default dashboard experience.
- A supported GPU environment and accepted Gemma access terms for the optional Kaggle fine-tuning workflow.

## Run the dashboard

```bash
ollama pull gemma3:4b
python start_control_center.py
```

Open <http://127.0.0.1:8765>. The launcher starts Ollama and the dashboard when they are not already responding. To start only the UI/API bridge, run `python visualizer/server.py`.

The server binds to localhost by default. Configuration:

- `OLLAMA_HOST`: Ollama API base URL; defaults to `http://127.0.0.1:11434`.
- `VISUALIZER_PORT`: dashboard port; defaults to `8765`.

The dashboard's API bridge exposes `/api/models`, `/api/model`, and the streaming `/api/generate` endpoint to its local browser UI.

## Desktop and responsive layout

On desktop widths, the dashboard uses the available viewport height and width so the input panel, layer explorer, neuron controls, response panel, and trace information remain visible together. The center panel reduces its graph and directory density on shorter desktop screens. At tablet and mobile widths, panels stack into a vertical layout so controls remain readable; the page can scroll on those smaller screens.

The neuron directory is paginated. It shows 20 IDs at a time rather than trying to place thousands of labels on screen simultaneously.

## Layer and neuron visualization

The 2D overview is a readable schematic of six representative blocks around the selected block. The 3D toggle expands the currently selected block and component. For the default text MLP view, stages represent:

1. 2,560-wide residual input.
2. Pre-feed-forward RMSNorm.
3. Separate 10,240-wide gate and up projections.
4. The elementwise GELU-tanh(gate) × up product.
5. The 10,240-to-2,560 down projection.
6. Post-feed-forward RMSNorm and residual addition.

The layer selector addresses the configured block number. The diagram renders 16 sampled indices at each stage and labels the full model widths; it does not draw all 10,240 units at once. The current component selector also supports the vision MLP and text query/key/value channel subviews.

The animated connections are a visual guide, not model weights or prompt-dependent activity. Ollama does not expose neuron firing, activation tensors, attention maps, or per-weight values. A true live internals view requires a separate instrumented Transformers/PyTorch inference backend and should report the tensors and performance costs it actually captures.

### Gemma 3 4B reference dimensions

| Component | Blocks/layers | Hidden width | Feed-forward width | Other dimensions |
| --- | ---: | ---: | ---: | --- |
| Text decoder | 34 | 2,560 | 10,240 | 8 query heads, 4 key/value heads, head width 256 |
| Vision encoder | 27 | 1,152 | 4,304 | Vision configuration is supplied from the released Gemma 3 4B config because Ollama's model info is text-oriented. |

The text block count and widths are read from the installed Ollama model metadata where available. Vision dimensions use the released configuration. See Google's [Gemma 3 4B configuration](https://github.com/google/gemma_pytorch/blob/main/gemma/config.py), the [Gemma 3 decoder implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modular_gemma3.py), and the [gated MLP implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py).

## Fine-tuning workflow

The checked-in GGUF is an inference artifact, not the training checkpoint. The prepared Kaggle notebook loads the instruction-tuned Hugging Face checkpoint and applies 4-bit LoRA/QLoRA using Unsloth.

1. Read `kaggle/KAGGLE-START-HERE.md` and import the prepared notebook into Kaggle.
2. Enable a GPU and internet in the Kaggle notebook settings.
3. Accept the Gemma terms for the selected Hugging Face checkpoint and store `HF_TOKEN` as a Kaggle secret.
4. Replace the notebook's LaTeX OCR demo dataset and prompt formatting with licensed task-specific examples.
5. Keep a held-out evaluation set separate; compare the adapter with the untuned base before deployment.
6. Save the adapter output, then plan and document conversion/merging for the target inference runtime.

The local Ollama GGUF blobs are not a suitable default fine-tuning source. Keep adapters, datasets, credentials, and generated checkpoints out of Git unless their licensing, size, and destination are explicitly appropriate.

## MCP roadmap

An MCP server has not been added yet. A planned implementation should:

- expose a small allowlist of task-specific tools;
- keep tool credentials and configuration outside model prompts;
- avoid unrestricted shell or filesystem tools;
- validate inputs and apply bounded timeouts;
- show tool calls and results to the user;
- require confirmation before consequential or external actions.

Model fine-tuning and MCP tool access are separate capabilities: changing model weights does not itself grant tools, permissions, or live access to model internals.

## Data, privacy, and limits

- Chat generation is sent to the locally configured Ollama server by default; the dashboard server binds to loopback.
- The server accepts image payloads for Ollama and rejects request bodies above 20 MB.
- The checked-in model package does not reveal its original training corpus or complete training process; those cannot be reconstructed from the GGUF weights.
- The layer animation is not evidence of a neuron's actual response to a prompt.
- The UI currently uses Ollama generation statistics and approximate word counts where exact tokenizer statistics are unavailable.

## Model terms and source licensing

Gemma 3 model weights and derivatives are subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Review those terms before training, modifying, or distributing weights or adapters. The source-code license for this project has not been specified in this repository yet; do not infer that the Gemma model terms license the dashboard code.

## Development roadmap

1. Replace demo training data with licensed task data and establish held-out evaluation.
2. Compare base and adapted checkpoints and document adapter export/serving steps.
3. Implement a separately configured, narrowly scoped MCP server.
4. Add real activation capture only through an instrumented backend, keeping measured tensors separate from the schematic display.
