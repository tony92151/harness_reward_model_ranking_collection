from functools import lru_cache
import os
import torch
from vllm import LLM, SamplingParams

from ..base import BaseRewardModel
from .constants_prompt import (
    build_autoj_input,
)  # constants_prompt -> codes/constants_prompt.py


# https://github.com/GAIR-NLP/auto-j/tree/main=
def extract_pariwise_result(raw_output):
    raw_output = raw_output.strip()
    pos = raw_output.rfind("final decision is ")
    pred_label = -1
    if pos != -1:
        pred_rest = raw_output[pos + len("final decision is ") :].strip().lower()
        if pred_rest.startswith("response 1"):
            pred_label = 0
        elif pred_rest.startswith("response 2"):
            pred_label = 1
        elif pred_rest.startswith("tie"):
            pred_label = 2
    return pred_label


def extract_single_rating(score_output):
    pred_score = 0.0
    if "Rating: [[" in score_output:
        pos = score_output.rfind("Rating: [[")
        pos2 = score_output.find("]]", pos)
        assert pos != -1 and pos2 != -1
        pred_score = float(score_output[pos + len("Rating: [[") : pos2].strip())
    return pred_score


class AutoJPipe(BaseRewardModel):
    def __init__(self, **kwargs):
        num_gpus = torch.cuda.device_count()
        model_name_or_dir = (
            "GAIR/autoj-13b"  # or the local directory to store the downloaded model
        )
        self.llm = LLM(model=model_name_or_dir, tensor_parallel_size=num_gpus)

    @lru_cache(maxsize=128)
    def autoj_pairwise_compare(self, instruction: str, candidates: list[str]):
        assert len(candidates) == 2, "AutoJ only supports pairwise comparison"

        input_pairwise = build_autoj_input(
            prompt=instruction,
            resp1=candidates[0],
            resp2=candidates[1],
            protocol="pairwise_tie",
        )  # for pairwise response comparison
        sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)
        outputs = self.llm.generate(input_pairwise, sampling_params)
        judgment = outputs[0].text
        return extract_pariwise_result(judgment)
    
    def get_reward_candidates(
        self, instruction: str, candidates: list[str], top_k: int = 3, **kwargs
    ) -> list[str]:

        use_pairwise: bool = kwargs.get("use_pairwise", False)

        max_tokens = os.getenv("MAX_NEW_TOKEN", 1024)

        if not use_pairwise:
            rewards = []
            for candidate in candidates:
                input_single = build_autoj_input(
                    prompt=instruction, resp1=candidate, resp2=None, protocol="single"
                )  # for single response evaluation
                input_ = input_single
                sampling_params = SamplingParams(
                    temperature=0.0, top_p=1.0, max_tokens=max_tokens
                )
                outputs = self.llm.generate(input_, sampling_params)
                judgment = outputs[0].outputs[0].text
                score = extract_single_rating(judgment)
                rewards.append(score)
            return [
                c for _, c in sorted(zip(rewards, candidates), reverse=True)[:top_k]
            ]
        else:
            raise NotImplementedError("Pairwise comparison is not implemented yet")
