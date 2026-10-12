"""GARL: Guidance-Aware Reinforcement Learning for Coach–Actor math reasoning."""

__version__ = "0.1.0"

from garl.guidance_score import guidance_final_score, guidance_outcome_score
from garl.rewards import actor_reward, format_reward, outcome_reward
from garl.inference import DualAgentPipeline

__all__ = [
    "DualAgentPipeline",
    "actor_reward",
    "format_reward",
    "outcome_reward",
    "guidance_outcome_score",
    "guidance_final_score",
]
