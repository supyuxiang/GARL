"""Dual-agent inference: Actor -> Coach -> Actor (refine)."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from garl.chat import actor_guided_messages, actor_messages, apply_chat_template, coach_messages
from garl.generation import GenConfig, generate_text
from garl.math_grade import answer_from_solution, is_solution_correct


@dataclass
class DualAgentResult:
    question: str
    initial_solution: str
    guidance: str
    refined_solution: str
    initial_correct: bool | None = None
    refined_correct: bool | None = None


class DualAgentPipeline:
    def __init__(
        self,
        actor: PreTrainedModel,
        coach: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        gen_config: GenConfig | None = None,
    ) -> None:
        self.actor = actor
        self.coach = coach
        self.tokenizer = tokenizer
        self.gen_config = gen_config or GenConfig()

    def run(self, question: str, ground_truth: str | None = None) -> DualAgentResult:
        actor_prompt = apply_chat_template(
            self.tokenizer, actor_messages(question), add_generation_prompt=True
        )
        initial = generate_text(self.actor, self.tokenizer, actor_prompt, self.gen_config)

        coach_prompt = apply_chat_template(
            self.tokenizer,
            coach_messages(question, initial),
            add_generation_prompt=True,
        )
        guidance = generate_text(self.coach, self.tokenizer, coach_prompt, self.gen_config)

        guided_prompt = apply_chat_template(
            self.tokenizer,
            actor_guided_messages(question, initial, guidance),
            add_generation_prompt=True,
        )
        refined = generate_text(self.actor, self.tokenizer, guided_prompt, self.gen_config)

        result = DualAgentResult(
            question=question,
            initial_solution=initial,
            guidance=guidance,
            refined_solution=refined,
        )
        if ground_truth is not None:
            result.initial_correct = is_solution_correct(question, initial, ground_truth)
            result.refined_correct = is_solution_correct(question, refined, ground_truth)
        return result

    def predict_answer(self, question: str) -> str | None:
        out = self.run(question)
        return answer_from_solution(out.refined_solution) or answer_from_solution(out.initial_solution)
