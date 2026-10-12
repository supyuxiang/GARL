"""Coach DPO trainer launched like verl SFT (torchrun -m garl.verl.coach_dpo_trainer).

Uses verl config layout (model/engine/optim) where possible; optimization loop is
vanilla DPO on FSDP-ready HF weights (single-process friendly for 7B Coach).
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from garl.verl.coach_dpo_dataset import CoachDPODataset, collate_coach_dpo
from verl.utils.device import auto_set_device
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group


def _seq_logprob(model, input_ids, attention_mask, loss_mask) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = loss_mask[:, 1:].bool()
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    token_lp = token_lp * mask
    return token_lp.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)


def dpo_step(policy, ref, batch, beta: float) -> torch.Tensor:
    pi_w = _seq_logprob(policy, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_loss_mask"])
    pi_l = _seq_logprob(policy, batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["rejected_loss_mask"])
    with torch.no_grad():
        ref_w = _seq_logprob(ref, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_loss_mask"])
        ref_l = _seq_logprob(ref, batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["rejected_loss_mask"])
    logits = beta * ((pi_w - pi_l) - (ref_w - ref_l))
    return -F.logsigmoid(logits).mean()


@hydra.main(config_path="../../verl/verl/trainer/config", config_name="garl_coach_dpo", version_base=None)
def main(config: DictConfig) -> None:
    auto_set_device(config)
    initialize_global_process_group()
    rank = dist.get_rank()
    world = dist.get_world_size()

    model_path = config.model.path
    ref_path = config.model.get("ref_path") or model_path
    train_files = config.data.train_files
    beta = float(config.trainer.get("dpo_beta", 0.1))
    lr = float(config.optim.lr)
    weight_decay = float(config.optim.get("weight_decay", 1e-2))
    max_steps = int(config.trainer.get("total_training_steps", 300))
    micro_bsz = int(config.data.micro_batch_size_per_gpu)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    policy = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True, device_map={"": rank}
    )
    ref = AutoModelForCausalLM.from_pretrained(
        ref_path, torch_dtype=dtype, trust_remote_code=True, device_map={"": rank}
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    policy.train()

    ds = CoachDPODataset(train_files, tokenizer, config.data)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    dl = DataLoader(
        ds,
        batch_size=micro_bsz,
        sampler=sampler,
        collate_fn=lambda b: collate_coach_dpo(b, tokenizer.pad_token_id),
    )

    opt = AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    save_dir = Path(config.trainer.default_local_dir)
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    pbar = tqdm(total=max_steps, disable=rank != 0, desc="coach-dpo")
    while step < max_steps:
        sampler.set_epoch(step)
        for batch in dl:
            batch = {k: v.to(policy.device) for k, v in batch.items()}
            loss = dpo_step(policy, ref, batch, beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=float(loss.detach().cpu()))
            if step >= max_steps:
                break

    if rank == 0:
        out = save_dir / "coach_dpo_hf"
        policy.save_pretrained(out)
        tokenizer.save_pretrained(out)
        (save_dir / "config.yaml").write_text(OmegaConf.to_yaml(config), encoding="utf-8")
        print(f"saved {out}")

    destroy_global_process_group()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    main()
