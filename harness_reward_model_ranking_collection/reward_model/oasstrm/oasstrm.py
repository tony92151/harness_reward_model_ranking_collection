from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..base import BaseRewardModel


class OasstrmPipe(BaseRewardModel):
    def __init__(self):
        reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
        self.rank_model = AutoModelForSequenceClassification.from_pretrained(
            reward_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(reward_name)

    def get_reward_candidates(
        self, instruction: str, candidates: list[str], top_k: int = 3
    ) -> list[str]:
        rewards = []
        for candidate in candidates:
            inputs = self.tokenizer(instruction, candidate, return_tensors="pt")
            score = self.rank_model(**inputs).logits[0].cpu().detach()
            rewards.append(score)

        return [c for _, c in sorted(zip(rewards, candidates), reverse=True)[:top_k]]
