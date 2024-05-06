import copy
import os
import sys
import time
from collections import defaultdict
from importlib.util import find_spec
from typing import List, Literal, Optional, Tuple, Union

import llm_blender
import numpy as np
import torch
from llm_blender.blender.blender_utils import get_topk_candidates_from_ranks
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from harness_reward_model_ranking_collection.entry import (
    REWARD_MODEL_MAP, RewardModelRankingEntry)


def init_llm_blender() -> llm_blender.Blender:
    # gf = GenFuserConfig(model_name="meta-llama/Llama-2-7b-hf")

    blender = llm_blender.Blender()
    # Load Ranker
    blender.loadranker("llm-blender/PairRM")  # load ranker checkpoint
    # blender.loadranker("OpenAssistant/reward-model-deberta-v3-large-v2") # load ranker checkpoint
    # Load Fuser
    blender.loadfuser(
        "llm-blender/gen_fuser_3b"
    )  # load fuser checkpoint if you want to use pre-trained fuser; or you can use ranker only
    return blender


def get_llm_blender_pairwise_ranks(
    llm_blender: llm_blender.Blender,
    input: str,
    candidates: list[str],
    instruction: str | None,
) -> list[list[int]]:
    t = time.time()
    ranks = llm_blender.rank(
        [input],
        [candidates],
        instructions=instruction,
        return_scores=False,
        batch_size=1,
    )
    time_cost = time.time() - t

    topk_candidates = get_topk_candidates_from_ranks(ranks, [candidates], top_k=1)
    topk_candidates = topk_candidates[0].tolist()

    return topk_candidates, [time_cost, time_cost]


@register_model("llm_blender", "llmblender")
class RewardModelRanking(LM):
    def __init__(self, batch_size: Optional[Union[int, str]] = 1, **kwargs) -> None:
        super().__init__()
        self.batch_size = batch_size

        self.llmb_kwargs = kwargs if kwargs else {}

        self.models = self.llmb_kwargs.get("models", None)

        assert (
            self.models is not None
        ), "Models are not provided. Use | to separate the models."
        self.models = self.models.split("|")

        self.ranker_only = self.llmb_kwargs.get("ranker_only", False)

        self.cache_path = os.getenv("HARNESS_HF_CACHE", None)
        assert (
            self.cache_path is not None
        ), "Cache path is not provided. Please set HARNESS_HF_CACHE at the environment variable."

        self.llm_blender = init_llm_blender()

        self._cache = {}

    def _get_cache(
        self, target_mode: str, task: str, request_text: str
    ) -> Union[str, None]:
        if target_mode not in self._cache:
            self._cache[target_mode] = self._load_cache(target_mode, task)
        return self._cache[target_mode].get(request_text, None)

    def _load_cache(self, target_mode: str, task: str) -> None:
        target_mode = target_mode.replace("/", "-")
        cache_file = os.path.join(
            os.path.abspath(self.cache_path), f"{task}_{target_mode}.pt"
        )
        assert os.path.exists(cache_file), f"Cache file {cache_file} does not exist."

        return torch.load(cache_file)

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        pass

    def loglikelihood_rolling(self, requests) -> list[tuple[float, bool]]:
        pass

    def generate_until(self, requests: list[Instance]) -> list[str]:
        if not requests:
            return []

        cache = {}
        total_results = []
        for request in tqdm(requests):
            candidate_list: list[str] = []

            for target_model in self.models:
                cache_value = self._get_cache(
                    target_model, os.getenv("TASK"), request.args[0]
                )
                assert (
                    cache_value is not None
                ), f"Cache value for {target_model} is not found. Please check the cache file."
                candidate_list.append(cache_value["response"])

            instruction = request.args[0].split("\n\n")[-1]
            # print(f"{candidate_list=}")

            rank_result, time_cost = get_llm_blender_pairwise_ranks(
                self.llm_blender, instruction, candidate_list
            )

            total_results.append(rank_result[0])
            cache[request.args[0]] = {
                "candidate_dict": {
                    m: {"text": c, "latency": t}
                    for c, m, t in zip(self.models, candidate_list, time_cost)
                },
                "final_candidate_model": self.models[
                    candidate_list.index(rank_result[0])
                ],
                "final_rank_result": rank_result[0],
            }

        if os.getenv("TASK") and self.cache_path:
            os.makedirs(self.cache_path, exist_ok=True)

            combine_models_ = "_".join(
                [target_model.replace("/", "-") for target_model in self.models]
            )
            save_path = os.path.join(
                self.cache_path,
                f'{self.rm_name}_{os.getenv("TASK")}_{combine_models_}.pt',
            )
            print(f"Save to cache {save_path}")
            torch.save(cache, save_path)

        return total_results
