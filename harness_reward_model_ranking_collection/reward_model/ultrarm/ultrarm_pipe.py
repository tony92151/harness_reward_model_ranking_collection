from typing import List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (LlamaConfig, LlamaModel, LlamaTokenizer,
                          PreTrainedModel)

from ..base import BaseRewardModel

ULTRARM_MODEL_ID = "openbmb/UltraRM-13b"


class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward(  # args are the same as LlamaForCausalLM
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]
        rewards = self.regression_head(hidden_states).squeeze(-1)

        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1, 1)
        rewards = torch.gather(rewards, 1, ends)

        return rewards


ultrarm_template = """Human: {instruction}

Assistant: {completion}"""


class UltrarmPipe(BaseRewardModel):
    def __init__(self, **kwargs):
        self.tokenizer = LlamaTokenizer.from_pretrained(ULTRARM_MODEL_ID)
        self.model = LlamaRewardModel.from_pretrained(
            ULTRARM_MODEL_ID, device_map="auto"
        )

    def get_reward_candidates(
        self, instruction: str, candidates: list[str], top_k: int = 3, **kwargs
    ) -> list[str]:
        for candidate in candidates:
            self.check_candidate(candidate)

        rejected_candidates: Optional[list[dict]] = kwargs.get(
            "rejected_candidates", None
        )

        if rejected_candidates:
            assert len(candidates) == len(
                rejected_candidates
            ), "Candidates and rejected candidates must have the same length"
            for rejected_candidate in rejected_candidates:
                self.check_candidate(rejected_candidate)

        rewards = []
        for idx, _ in tqdm(enumerate(candidates), desc="Getting UltraRM rewards"):
            ultrarm_text = ultrarm_template.format(
                instruction=instruction,
                completion=candidates[idx],
            )
            inputs = self.tokenizer(ultrarm_text, return_tensors="pt")
            chosen_reward = self.model(**inputs).item()

            if rejected_candidates:
                rejected_ultrarm_text = ultrarm_template.format(
                    instruction=instruction,
                    completion=rejected_candidates[idx],
                )
                inputs = self.tokenizer(rejected_ultrarm_text, return_tensors="pt")
                rejected_reward = self.model(**inputs).item()
                rewards.append(chosen_reward - rejected_reward)
            else:
                rewards.append(chosen_reward)

        return [c for _, c in sorted(zip(rewards, candidates), reverse=True)[:top_k]]
