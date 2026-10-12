from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_causal_lm(
    model_name: str,
    dtype: torch.dtype | None = None,
    device_map: str | dict | None = "auto",
) -> PreTrainedModel:
    dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )


def attach_lora(model: PreTrainedModel, rank: int = 64, alpha: int = 128) -> PreTrainedModel:
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, cfg)


def load_with_optional_adapter(base_path: str, adapter_path: str | None = None) -> PreTrainedModel:
    model = load_causal_lm(base_path)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    return model
