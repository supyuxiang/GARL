"""Preference dataset for Coach DPO (verl SFT engine + custom loss)."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.tokenizer.chat_template import apply_chat_template


class CoachDPODataset(Dataset):
    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        del processor
        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]
        self.tokenizer = tokenizer if not isinstance(tokenizer, str) else hf_tokenizer(tokenizer)
        self.max_length = config.get("max_length", 4096)
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/garl_dpo"))

        frames = []
        for pf in parquet_files:
            local = copy_local_path_from_hdfs(pf, cache_dir=self.cache_dir)
            frames.append(pd.read_parquet(local))
        self.df = pd.concat(frames, ignore_index=True)
        if 0 < max_samples < len(self.df):
            self.df = self.df.iloc[:max_samples]

    def __len__(self) -> int:
        return len(self.df)

    def _encode(self, messages: list[dict], response: str) -> dict:
        prompt_text = apply_chat_template(
            self.tokenizer, messages, add_generation_prompt=True, tokenize=False
        )
        full = prompt_text + response
        enc = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_ids = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"]
        plen = prompt_ids.shape[1]
        labels = enc["input_ids"].clone()
        labels[:, :plen] = -100
        loss_mask = (labels != -100).long()
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "loss_mask": loss_mask.squeeze(0),
        }

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx].to_dict()
        messages = row["prompt_messages"]
        w = self._encode(messages, row["chosen"])
        l = self._encode(messages, row["rejected"])
        return {
            "chosen_input_ids": w["input_ids"],
            "chosen_attention_mask": w["attention_mask"],
            "chosen_loss_mask": w["loss_mask"],
            "rejected_input_ids": l["input_ids"],
            "rejected_attention_mask": l["attention_mask"],
            "rejected_loss_mask": l["loss_mask"],
        }


def collate_coach_dpo(batch: list[dict], pad_token_id: int) -> dict:
    def pad_stack(key: str, pad_val: int) -> torch.Tensor:
        tensors = [b[key] for b in batch]
        max_len = max(t.size(0) for t in tensors)
        out = torch.full((len(tensors), max_len), pad_val, dtype=tensors[0].dtype)
        for i, t in enumerate(tensors):
            out[i, : t.size(0)] = t
        return out

    return {
        "chosen_input_ids": pad_stack("chosen_input_ids", pad_token_id),
        "chosen_attention_mask": pad_stack("chosen_attention_mask", 0),
        "chosen_loss_mask": pad_stack("chosen_loss_mask", 0),
        "rejected_input_ids": pad_stack("rejected_input_ids", pad_token_id),
        "rejected_attention_mask": pad_stack("rejected_attention_mask", 0),
        "rejected_loss_mask": pad_stack("rejected_loss_mask", 0),
    }
