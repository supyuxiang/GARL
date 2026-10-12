"""Supervised fine-tuning for Coach (Eq. SFT / cold-start)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from garl.chat import apply_chat_template, coach_messages
from garl.training.load import load_causal_lm, load_tokenizer

class CoachSFTDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int = 4096) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        messages = coach_messages(row["question"], row["initial_solution"])
        prompt = apply_chat_template(self.tokenizer, messages, add_generation_prompt=True)
        full = prompt + row["guidance"]
        enc = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_len = len(
            self.tokenizer(prompt, truncation=True, max_length=self.max_length)["input_ids"]
        )
        labels = enc["input_ids"].clone()
        labels[:, :prompt_len] = -100
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


@dataclass
class CoachSFTConfig:
    model_name: str
    output_dir: str
    lr: float = 1e-5
    weight_decay: float = 1e-2
    num_epochs: int = 3
    batch_size: int = 1
    grad_accum: int = 8
    max_length: int = 4096
    warmup_ratio: float = 0.0


def train_coach_sft(data_path: str | Path, cfg: CoachSFTConfig) -> Path:
    data_path = Path(data_path)
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = load_tokenizer(cfg.model_name)
    model = load_causal_lm(cfg.model_name)
    model.train()

    ds = CoachSFTDataset(rows, tokenizer, max_length=cfg.max_length)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(cfg.num_epochs):
        pbar = tqdm(dl, desc=f"coach-sft epoch {epoch+1}/{cfg.num_epochs}")
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out.loss / cfg.grad_accum
            loss.backward()
            if (step + 1) % cfg.grad_accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
            pbar.set_postfix(loss=float(out.loss.detach().cpu()))

    save_dir = out / "coach_sft"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return save_dir
