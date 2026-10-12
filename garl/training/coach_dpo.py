"""Guidance-aware DPO for Coach (Eq. DPO)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from garl.chat import apply_chat_template, coach_messages
from garl.training.load import load_causal_lm, load_tokenizer
from garl.training.logprob import sequence_logprob


class CoachDPODataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int = 4096) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def _encode_pair(self, row: dict, response: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        messages = coach_messages(row["question"], row["initial_solution"])
        prompt = apply_chat_template(self.tokenizer, messages, add_generation_prompt=True)
        full = prompt + response
        enc = self.tokenizer(full, truncation=True, max_length=self.max_length, return_tensors="pt")
        prompt_len = len(
            self.tokenizer(prompt, truncation=True, max_length=self.max_length)["input_ids"]
        )
        labels = enc["input_ids"].clone()
        labels[:, :prompt_len] = -100
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), labels.squeeze(0)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        w_ids, w_mask, w_labels = self._encode_pair(row, row["chosen"])
        l_ids, l_mask, l_labels = self._encode_pair(row, row["rejected"])
        return {
            "w_input_ids": w_ids,
            "w_attention_mask": w_mask,
            "w_labels": w_labels,
            "l_input_ids": l_ids,
            "l_attention_mask": l_mask,
            "l_labels": l_labels,
        }


def _pad_batch(items: list[dict], key: str, pad_id: int) -> torch.Tensor:
    tensors = [it[key] for it in items]
    max_len = max(t.size(0) for t in tensors)
    out = torch.full((len(tensors), max_len), pad_id, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, : t.size(0)] = t
    return out


def collate_dpo(batch: list[dict], pad_id: int) -> dict:
    return {
        "w_input_ids": _pad_batch(batch, "w_input_ids", pad_id),
        "w_attention_mask": _pad_batch(batch, "w_attention_mask", 0),
        "w_labels": _pad_batch(batch, "w_labels", -100),
        "l_input_ids": _pad_batch(batch, "l_input_ids", pad_id),
        "l_attention_mask": _pad_batch(batch, "l_attention_mask", 0),
        "l_labels": _pad_batch(batch, "l_labels", -100),
    }


@dataclass
class CoachDPOConfig:
    model_name: str
    ref_model_name: str | None
    output_dir: str
    beta: float = 0.1
    lr: float = 1e-5
    weight_decay: float = 1e-2
    max_steps: int = 300
    batch_size: int = 1
    grad_accum: int = 8
    max_length: int = 4096


def dpo_loss(
    policy: torch.nn.Module,
    ref: torch.nn.Module,
    batch: dict,
    beta: float,
) -> torch.Tensor:
    pi_w = sequence_logprob(policy, batch["w_input_ids"], batch["w_attention_mask"], batch["w_labels"])
    pi_l = sequence_logprob(policy, batch["l_input_ids"], batch["l_attention_mask"], batch["l_labels"])
    with torch.no_grad():
        ref_w = sequence_logprob(ref, batch["w_input_ids"], batch["w_attention_mask"], batch["w_labels"])
        ref_l = sequence_logprob(ref, batch["l_input_ids"], batch["l_attention_mask"], batch["l_labels"])
    logits = beta * ((pi_w - pi_l) - (ref_w - ref_l))
    return -F.logsigmoid(logits).mean()


def train_coach_dpo(data_path: str | Path, cfg: CoachDPOConfig) -> Path:
    data_path = Path(data_path)
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = load_tokenizer(cfg.model_name)
    policy = load_causal_lm(cfg.model_name)
    ref_name = cfg.ref_model_name or cfg.model_name
    ref = load_causal_lm(ref_name)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    policy.train()
    ds = CoachDPODataset(rows, tokenizer, max_length=cfg.max_length)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_dpo(b, tokenizer.pad_token_id),
    )

    opt = AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    step = 0
    pbar = tqdm(total=cfg.max_steps, desc="coach-dpo")
    while step < cfg.max_steps:
        for batch in dl:
            batch = {k: v.to(policy.device) for k, v in batch.items()}
            loss = dpo_loss(policy, ref, batch, cfg.beta) / cfg.grad_accum
            loss.backward()
            if (step + 1) % cfg.grad_accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=float(loss.detach().cpu() * cfg.grad_accum))
            if step >= cfg.max_steps:
                break

    save_dir = out / "coach_dpo"
    policy.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return save_dir
