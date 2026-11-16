"""Local browser chat UI for the trained 50M defensive cybersecurity GPT."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
import torch
from tokenizers import Tokenizer

from train_gpt import SmallGPT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "data/kaggle-output/version-7/cyber_50m_sft/best_sft_checkpoint.pt"
DEFAULT_TOKENIZER = ROOT / "data/kaggle-output/version-7/cyber_50m_pretrain/tokenizer.json"
DEFAULT_HF_CHECKPOINT = ROOT / "data/kaggle-output/135m/best_model"
GUARDRAILS_PATH = ROOT / "cyber_135m/guardrails.md"


@st.cache_resource(show_spinner="Loading the model…")
def load_model(checkpoint_path: str, tokenizer_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = artifact["config"]
    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = SmallGPT(int(artifact["vocab_size"]), config)
    model.load_state_dict(artifact["model"], strict=True)
    model.to(device).eval()
    return model, tokenizer, device


@st.cache_resource(show_spinner="Loading the fine-tuned 135M model…")
def load_hf_model(checkpoint_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=dtype)
    model.to(device).eval()
    return model, tokenizer, device


def sample_next(logits: torch.Tensor, temperature: float, top_p: float, recent: list[int], penalty: float):
    logits = logits.float()
    if penalty > 1 and recent:
        for token_id in set(recent):
            logits[token_id] = logits[token_id] / penalty if logits[token_id] > 0 else logits[token_id] * penalty
    logits /= max(temperature, 0.05)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probabilities = torch.softmax(sorted_logits, dim=-1)
    remove = torch.cumsum(probabilities, dim=-1) - probabilities > top_p
    sorted_logits[remove] = -float("inf")
    filtered = torch.full_like(logits, -float("inf"))
    filtered.scatter_(0, sorted_indices, sorted_logits)
    return int(torch.multinomial(torch.softmax(filtered, dim=-1), 1).item())


@torch.inference_mode()
def generate(model, tokenizer, device, history: list[dict[str, str]], max_new_tokens: int,
             temperature: float, top_p: float, repetition_penalty: float, guardrails: str) -> str:
    # Match the chat-style prompt used by the instruction data and reserve room
    # for generated tokens inside the model's context window.
    transcript = f"System: {guardrails.strip()}\n"
    transcript += "\n".join(f"User: {m['content']}\nAssistant: {m.get('answer', '')}" for m in history[:-1])
    transcript += f"\nUser: {history[-1]['content']}\nAssistant:"
    ids = tokenizer.encode(transcript).ids
    block = int(model.block)
    ids = ids[-max(1, block - max_new_tokens):]
    eos_id = tokenizer.token_to_id("[EOS]")
    generated: list[int] = []
    for _ in range(max_new_tokens):
        context = torch.tensor([ids + generated], dtype=torch.long, device=device)
        next_logits = model(context[:, -block:])[0, -1]
        token_id = sample_next(next_logits, temperature, top_p, generated[-128:], repetition_penalty)
        if token_id == eos_id:
            break
        generated.append(token_id)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.inference_mode()
def generate_hf(model, tokenizer, device, history: list[dict[str, str]], max_new_tokens: int,
                temperature: float, top_p: float, repetition_penalty: float, guardrails: str) -> str:
    messages = [{"role": "system", "content": guardrails.strip()}]
    for item in history:
        messages.append({"role": "user", "content": item["content"]})
        if item.get("answer"):
            messages.append({"role": "assistant", "content": item["answer"]})
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                            temperature=max(temperature, 0.05), top_p=top_p,
                            repetition_penalty=repetition_penalty,
                            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


st.set_page_config(page_title="Defensive Cyber Chat", page_icon="🛡️", layout="centered")
st.title("Defensive Cyber Chat")
st.caption("Local chat. The 50M pilot is weak; choose the 135M fine-tuned model after its Kaggle checkpoint is available.")

with st.sidebar:
    model_type = st.radio("Model", ["50M pilot", "135M fine-tuned"], index=0)
    st.subheader("Model files")
    if model_type == "50M pilot":
        checkpoint = st.text_input("50M SFT checkpoint", value=os.getenv("CYBER50M_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
        tokenizer_file = st.text_input("50M tokenizer", value=os.getenv("CYBER50M_TOKENIZER", str(DEFAULT_TOKENIZER)))
    else:
        checkpoint = st.text_input("135M model folder", value=os.getenv("CYBER135M_CHECKPOINT", str(DEFAULT_HF_CHECKPOINT)))
        tokenizer_file = ""
    max_tokens = st.slider("Maximum reply tokens", 16, 256, 128, 16)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8, 0.05)
    top_p = st.slider("Top-p", 0.5, 1.0, 0.9, 0.05)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 1.5, 1.1, 0.05)
    default_guardrails = GUARDRAILS_PATH.read_text(encoding="utf-8") if GUARDRAILS_PATH.exists() else "Be helpful, accurate, and defensive."
    guardrails = st.text_area("Your guardrails", value=default_guardrails, height=210, key="guardrails_editor")
    if st.button("Save guardrails"):
        GUARDRAILS_PATH.write_text(guardrails.strip() + "\n", encoding="utf-8")
        st.success("Saved to cyber_50m/guardrails.md")
    st.caption("Model answers can be inaccurate. Verify important security decisions against trusted sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_text = st.chat_input("Ask a defensive cybersecurity question…")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        try:
            if model_type == "50M pilot":
                if not Path(checkpoint).is_file() or not Path(tokenizer_file).is_file():
                    raise FileNotFoundError("Checkpoint or tokenizer file not found. Set their paths in the sidebar.")
                model, tokenizer, device = load_model(checkpoint, tokenizer_file)
            else:
                if not (Path(checkpoint) / "config.json").is_file():
                    raise FileNotFoundError("135M model folder is missing config.json. Download the Kaggle best_model folder first.")
                model, tokenizer, device = load_hf_model(checkpoint)
            history = []
            pending_user = None
            for item in st.session_state.messages:
                if item["role"] == "user":
                    if pending_user is not None:
                        history.append({"content": pending_user, "answer": ""})
                    pending_user = item["content"]
                elif pending_user is not None:
                    history.append({"content": pending_user, "answer": item["content"]})
                    pending_user = None
            if pending_user is not None:
                history.append({"content": pending_user, "answer": ""})
            if model_type == "50M pilot":
                answer = generate(model, tokenizer, device, history, max_tokens, temperature, top_p,
                                  repetition_penalty, guardrails)
            else:
                answer = generate_hf(model, tokenizer, device, history, max_tokens, temperature, top_p,
                                     repetition_penalty, guardrails)
            if not answer:
                answer = "The model returned an empty reply. Try a shorter or more specific prompt."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as exc:
            st.error(f"Could not load or run the model: {exc}")
            st.info("The Kaggle checkpoint must be fully downloaded and intact. A .kaggle-partial marker indicates an interrupted download; resume/download the complete artifact before chatting.")
