"""Hybrid-mode GRPO for Actor (question-only + guidance-conditioned)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from garl.chat import actor_guided_messages, actor_messages, apply_chat_template
from garl.generation import GenConfig, generate_text
from garl.rewards import actor_reward
from garl.training.load import load_causal_lm, load_tokenizer
from garl.training.logprob import sequence_logprob


@dataclass
class ActorGRPOConfig:
    model_name: str
    ref_model_name: str | None
    output_dir: str
    group_size: int = 8
    epsilon: float = 0.2
    beta_kl: float = 0.04
    lr: float = 1e-5
    weight_decay: float = 1e-2
    max_steps: int = 300
    alpha_guidance: float = 1.0  # weight for guidance-mode groups vs question-only
    gen: GenConfig | None = None


def _group_advantages(rewards: list[float]) -> list[float]:
    t = torch.tensor(rewards, dtype=torch.float32)
    mean = t.mean()
    std = t.std(unbiased=False).clamp(min=1e-6)
    return ((t - mean) / std).tolist()


def _encode_completion(tokenizer, messages: list[dict], completion: str, max_length: int = 4096):
    prompt = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    full = prompt + completion
    enc = tokenizer(full, truncation=True, max_length=max_length, return_tensors="pt")
    prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"])
    labels = enc["input_ids"].clone()
    labels[:, :prompt_len] = -100
    return enc["input_ids"], enc["attention_mask"], labels


def grpo_step(
    policy,
    ref,
    tokenizer,
    messages: list[dict],
    ground_truth: str,
    question: str,
    cfg: ActorGRPOConfig,
) -> torch.Tensor:
    gen = cfg.gen or GenConfig(temperature=1.0, top_p=1.0)
    prompt = apply_chat_template(tokenizer, messages, add_generation_prompt=True)

    completions: list[str] = []
    rewards: list[float] = []
    for _ in range(cfg.group_size):
        text = generate_text(policy, tokenizer, prompt, gen)
        completions.append(text)
        rewards.append(actor_reward(question, text, ground_truth))

    advantages = _group_advantages(rewards)
    losses = []
    for comp, adv in zip(completions, advantages):
        input_ids, attn, labels = _encode_completion(tokenizer, messages, comp)
        input_ids = input_ids.to(policy.device)
        attn = attn.to(policy.device)
        labels = labels.to(policy.device)
        logp = sequence_logprob(policy, input_ids, attn, labels, length_norm=False)
        with torch.no_grad():
            logp_old = sequence_logprob(ref, input_ids, attn, labels, length_norm=False)
            logp_ref = logp_old
        ratio = torch.exp(logp - logp_old)
        adv_t = torch.tensor(adv, device=policy.device)
        unclipped = ratio * adv_t
        clipped = torch.clamp(ratio, 1.0 - cfg.epsilon, 1.0 + cfg.epsilon) * adv_t
        pg = -torch.min(unclipped, clipped)
        kl = cfg.beta_kl * (logp_ref - logp)
        losses.append(pg + kl)
    return torch.stack(losses).mean()


def train_actor_grpo(data_path: str | Path, cfg: ActorGRPOConfig) -> Path:
    data_path = Path(data_path)
    rows = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokenizer = load_tokenizer(cfg.model_name)
    policy = load_causal_lm(cfg.model_name)
    ref_name = cfg.ref_model_name or cfg.model_name
    ref = load_causal_lm(ref_name)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    policy.train()

    opt = AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    pbar = tqdm(total=cfg.max_steps, desc="actor-grpo")
    step = 0
    while step < cfg.max_steps:
        row = rng.choice(rows)
        question = row.get("question") or row.get("problem")
        gt = row.get("answer") or row.get("ground_truth")
        guidance = row.get("guidance")

        loss_q = grpo_step(
            policy,
            ref,
            tokenizer,
            actor_messages(question),
            gt,
            question,
            cfg,
        )
        loss = loss_q
        if guidance:
            loss_g = grpo_step(
                policy,
                ref,
                tokenizer,
                actor_guided_messages(question, row.get("initial_solution", ""), guidance),
                gt,
                question,
                cfg,
            )
            loss = loss_q + cfg.alpha_guidance * loss_g

        opt.zero_grad(set_to_none=True)
        (loss / 1).backward()
        opt.step()
        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=float(loss.detach().cpu()))

    save_dir = out / "actor_grpo"
    policy.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return save_dir
