"""Extract one refusal-related direction and orthogonalize Gemma 3 residual writers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import mmap
import shutil
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig


MODEL_ID = "google/gemma-3-4b-it"
ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = Path(__file__).resolve().parent / "contrastive_prompts.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "gemma-3-4b-it-abliterated"
EMBEDDING_KEY = "language_model.model.embed_tokens.weight"
VISION_WRITER_KEY = "multi_modal_projector.mm_input_projection_weight"


def get_text_layers(model):
    candidates = (
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    )
    for candidate in candidates:
        layers = getattr(candidate, "layers", None)
        if layers is not None:
            return layers
    raise RuntimeError("Could not find Gemma 3 text decoder layers.")


def load_quantized_model(snapshot: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for activation capture.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading the source checkpoint for direction capture ({dtype}, NF4).", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot,
        quantization_config=quantization,
        dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def capture_prompt_states(tokenizer, model, prompt: str) -> torch.Tensor:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt")
    inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
    output = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    if output.hidden_states is None:
        raise RuntimeError("The checkpoint returned no hidden states.")
    last = int(inputs["attention_mask"][0].sum().item()) - 1
    states = torch.stack([state[0, last].float().cpu() for state in output.hidden_states[1:]])
    if not torch.isfinite(states).all().item():
        raise RuntimeError("Non-finite activations found; refusing to edit weights.")
    return states


def extract_direction(snapshot: Path, prompts_path: Path):
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    harmful = prompts.get("harmful", [])
    harmless = prompts.get("harmless", [])
    if len(harmful) < 8 or len(harmless) < 8:
        raise RuntimeError("At least eight prompts per class are required for the pilot split.")

    tokenizer, model = load_quantized_model(snapshot)
    layers = get_text_layers(model)
    captured: dict[str, list[torch.Tensor]] = {"harmful": [], "harmless": []}
    for label, examples in (("harmful", harmful), ("harmless", harmless)):
        for index, prompt in enumerate(examples, start=1):
            captured[label].append(capture_prompt_states(tokenizer, model, prompt))
            print(f"Captured {label} activation {index}/{len(examples)}", flush=True)

    harmful_states = torch.stack(captured["harmful"])
    harmless_states = torch.stack(captured["harmless"])
    split = min(len(harmful) - 4, len(harmless) - 4)
    if split < 4:
        raise RuntimeError("The prompt set must leave at least four held-out examples per class.")
    train_diff = harmful_states[:split].mean(0) - harmless_states[:split].mean(0)
    validation_diff = harmful_states[split:].mean(0) - harmless_states[split:].mean(0)

    # The published method selects directions using refusal-score and benign-KL
    # validation. This compact pilot selects the strongest train-set layer and
    # checks held-out activation alignment; it does not claim equivalent validation.
    selectable_layers = max(1, int(len(layers) * 0.8))
    scores = torch.linalg.vector_norm(train_diff[:selectable_layers], dim=-1)
    layer_index = int(torch.argmax(scores).item())
    raw_direction = train_diff[layer_index]
    norm = torch.linalg.vector_norm(raw_direction)
    if not torch.isfinite(norm).item() or norm.item() < 1e-8:
        raise RuntimeError("The selected residual direction is invalid; no weights were edited.")
    direction = raw_direction / norm

    validation_vector = validation_diff[layer_index]
    validation_norm = torch.linalg.vector_norm(validation_vector)
    validation_cosine = float(torch.dot(direction, validation_vector / validation_norm).item()) if validation_norm > 1e-8 else 0.0
    harmful_projection = (harmful_states[split:, layer_index] @ direction).mean()
    harmless_projection = (harmless_states[split:, layer_index] @ direction).mean()
    validation_gap = float((harmful_projection - harmless_projection).item())
    if validation_cosine <= 0 or validation_gap <= 0:
        raise RuntimeError(
            "Held-out prompts did not align with the candidate direction; refusing to edit weights."
        )

    direction_hash = hashlib.sha256(direction.contiguous().numpy().tobytes()).hexdigest()
    report = {
        "model_id": MODEL_ID,
        "method": "difference-in-means, one residual direction; compact pilot selection",
        "direction_layer": layer_index,
        "direction_sha256": direction_hash,
        "direction_norm": float(norm.item()),
        "hidden_size": int(direction.numel()),
        "text_layers": len(layers),
        "train_prompts_per_class": split,
        "validation_prompts_per_class": len(harmful) - split,
        "validation_projection_gap": validation_gap,
        "validation_direction_cosine": validation_cosine,
        "compute_dtype": str(next(model.parameters()).dtype),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitation": "Activation separation is not a behavioral refusal or capability evaluation.",
    }

    del model, tokenizer, harmful_states, harmless_states, captured
    gc.collect()
    torch.cuda.empty_cache()
    return direction.contiguous(), report


def read_header(path: Path):
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"Invalid safetensors header: {path}")
        header_length = struct.unpack("<Q", prefix)[0]
        header_bytes = handle.read(header_length)
    header = json.loads(header_bytes)
    return 8 + header_length, header


def read_small_tensor(path: Path, key: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).float().contiguous()


def copy_assets(source: Path, staging: Path) -> None:
    staging.mkdir(parents=True)
    for item in source.iterdir():
        if item.is_file() and not (item.name.startswith("model-") and item.suffix == ".safetensors"):
            shutil.copy2(item, staging / item.name)


def patch_row_vectors(mapping, offset: int, shape: list[int], direction: torch.Tensor) -> None:
    rows, width = shape
    if width != direction.numel():
        raise RuntimeError(f"Row writer width {width} does not match direction width {direction.numel()}.")
    tensor = torch.frombuffer(mapping, dtype=torch.bfloat16, count=rows * width, offset=offset).view(rows, width)
    for first in range(0, rows, 512):
        last = min(first + 512, rows)
        block = tensor[first:last].float()
        coefficient = block @ direction
        block -= coefficient[:, None] * direction[None, :]
        tensor[first:last].copy_(block.to(torch.bfloat16))
    del tensor


def patch_linear_writer(mapping, offset: int, shape: list[int], direction: torch.Tensor) -> None:
    outputs, inputs = shape
    if outputs != direction.numel():
        raise RuntimeError(f"Linear writer output width {outputs} does not match direction width {direction.numel()}.")
    tensor = torch.frombuffer(mapping, dtype=torch.bfloat16, count=outputs * inputs, offset=offset).view(outputs, inputs)
    for first in range(0, inputs, 1024):
        last = min(first + 1024, inputs)
        block = tensor[:, first:last].float()
        coefficient = direction @ block
        block -= direction[:, None] * coefficient[None, :]
        tensor[:, first:last].copy_(block.to(torch.bfloat16))
    del tensor


def build_target_directions(snapshot: Path, direction: torch.Tensor, weight_map: dict[str, str]):
    targets: dict[str, tuple[str, torch.Tensor | None]] = {
        EMBEDDING_KEY: ("rows", direction),
        VISION_WRITER_KEY: ("rows", direction),
    }
    n_layers = 34
    for index in range(n_layers):
        prefix = f"language_model.model.layers.{index}"
        post_attention = f"{prefix}.post_attention_layernorm.weight"
        post_feedforward = f"{prefix}.post_feedforward_layernorm.weight"
        attention_writer = f"{prefix}.self_attn.o_proj.weight"
        mlp_writer = f"{prefix}.mlp.down_proj.weight"
        for required in (post_attention, post_feedforward, attention_writer, mlp_writer):
            if required not in weight_map:
                raise RuntimeError(f"Expected Gemma 3 residual tensor is missing: {required}")
        attn_path = snapshot / weight_map[post_attention]
        mlp_path = snapshot / weight_map[post_feedforward]
        attn_gamma = 1.0 + read_small_tensor(attn_path, post_attention)
        mlp_gamma = 1.0 + read_small_tensor(mlp_path, post_feedforward)
        attn_direction = direction * attn_gamma
        mlp_direction = direction * mlp_gamma
        attn_direction /= torch.linalg.vector_norm(attn_direction).clamp_min(1e-12)
        mlp_direction /= torch.linalg.vector_norm(mlp_direction).clamp_min(1e-12)
        targets[attention_writer] = ("linear", attn_direction.contiguous())
        targets[mlp_writer] = ("linear", mlp_direction.contiguous())

    missing = sorted(set(targets) - set(weight_map))
    if missing:
        raise RuntimeError("Residual writer tensors are missing from checkpoint: " + ", ".join(missing))
    return targets


def edit_checkpoint(snapshot: Path, output_dir: Path, direction: torch.Tensor, direction_report: dict) -> Path:
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(f"Sharded safetensors index not found: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    if int(text_config.get("num_hidden_layers", 0)) != 34 or int(text_config.get("hidden_size", 0)) != direction.numel():
        raise RuntimeError("Checkpoint architecture does not match this Gemma 3 4B editor.")
    tie_word_embeddings = text_config.get("tie_word_embeddings", config.get("tie_word_embeddings", True))
    if tie_word_embeddings is not True:
        raise RuntimeError("Expected tied Gemma token embeddings; refusing an unreviewed architecture variant.")

    targets = build_target_directions(snapshot, direction, weight_map)
    staging = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or staging.exists():
        raise RuntimeError(f"Output or partial output already exists: {output_dir} / {staging}")
    if shutil.disk_usage(output_dir.parent).free < sum(
        (snapshot / shard).stat().st_size for shard in set(weight_map.values())
    ) * 1.1:
        raise RuntimeError("Not enough free disk space for a separate edited checkpoint and working margin.")

    copy_assets(snapshot, staging)
    shards = sorted(set(weight_map.values()))
    changed = []
    for shard_name in shards:
        source_path = snapshot / shard_name
        output_path = staging / shard_name
        print(f"Copying checkpoint shard {shard_name} ({source_path.stat().st_size / 1e9:.2f} GB).", flush=True)
        shutil.copy2(source_path, output_path)
        data_start, header = read_header(output_path)
        keys_here = [key for key in targets if weight_map[key] == shard_name]
        with output_path.open("r+b") as file_handle:
            mapped = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_WRITE)
            try:
                for key in keys_here:
                    entry = header[key]
                    if entry.get("dtype") != "BF16":
                        raise RuntimeError(f"Expected BF16 tensor for {key}; found {entry.get('dtype')}.")
                    begin, end = entry["data_offsets"]
                    kind, writer_direction = targets[key]
                    if end - begin != math.prod(entry["shape"]) * 2:
                        raise RuntimeError(f"Unexpected byte size for {key}.")
                    absolute_offset = data_start + begin
                    if kind == "rows":
                        patch_row_vectors(mapped, absolute_offset, entry["shape"], writer_direction)
                    else:
                        patch_linear_writer(mapped, absolute_offset, entry["shape"], writer_direction)
                    changed.append(key)
                    print(f"Orthogonalized {key}", flush=True)
                mapped.flush()
            finally:
                mapped.close()

    if set(changed) != set(targets):
        raise RuntimeError(f"Edited {len(changed)} of {len(targets)} planned residual writers.")

    source_revision = (Path.home() / ".cache" / "huggingface" / "hub" / "models--google--gemma-3-4b-it" / "refs" / "main")
    revision = source_revision.read_text(encoding="utf-8").strip() if source_revision.is_file() else "unknown"
    report = {
        **direction_report,
        "source_revision": revision,
        "weight_method": "W_out <- W_out - q(q^T W_out), where q accounts for the following Gemma RMSNorm scale",
        "embedding_and_vision_method": "row vectors <- row vectors - (row dot r) r",
        "changed_tensor_count": len(changed),
        "changed_tensors": changed,
        "unchanged_tensor_policy": "All tensors outside this exact residual-writer allowlist were copied byte-for-byte.",
        "output_checkpoint": str(output_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "safety_note": "This is an experimental weight edit, not a claim of exact or behavior-preserving guardrail removal.",
    }
    (staging / "abliteration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    staging.replace(output_dir)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=PROMPTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        snapshot = Path(snapshot_download(MODEL_ID, local_files_only=True))
    except Exception as error:
        print(
            f"The approved {MODEL_ID} checkpoint is not fully cached. Download it with the Hugging Face CLI first: {error}",
            file=sys.stderr,
        )
        return 1

    direction, direction_report = extract_direction(snapshot, args.prompts)
    output = edit_checkpoint(snapshot, args.output.resolve(), direction, direction_report)
    print(f"Wrote a separate edited checkpoint to {output}", flush=True)
    print(f"Report: {output / 'abliteration_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("Interrupted; source checkpoint remains unchanged.")
