import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..base import BaseRewardModel


class OasstrmPipe(BaseRewardModel):
    def __init__(self, **kwargs):
        reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rank_model = AutoModelForSequenceClassification.from_pretrained(
            reward_name
        )
        self.rank_model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(reward_name, device_map="auto")

    def get_reward_candidates(
        self, instruction: str, candidates: list[str], top_k: int = 3
    ) -> tuple[list[str], list[float]]:
        time_cost = []
        rewards = []
        for candidate in candidates:
            _t = time.time()
            inputs = self.tokenizer(instruction, candidate, return_tensors="pt").to(
                self.rank_model.device
            )
            score = self.rank_model(**inputs).logits[0].cpu().detach()
            rewards.append(score)
            time_cost.append(time.time() - _t)

        return [
            c for _, c in sorted(zip(rewards, candidates), reverse=True)[:top_k]
        ], time_cost
