"""HF generation utilities shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class GenConfig:
    max_new_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 50
    do_sample: bool = True


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    cfg: GenConfig | None = None,
) -> str:
    cfg = cfg or GenConfig()
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature if cfg.do_sample else None,
            top_p=cfg.top_p if cfg.do_sample else None,
            top_k=cfg.top_k if cfg.do_sample else None,
            do_sample=cfg.do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def count_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))
