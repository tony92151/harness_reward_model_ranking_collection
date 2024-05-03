import copy
import os
import sys
from collections import defaultdict
from importlib.util import find_spec
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from harness_reward_model_ranking_collection.entry import (
    REWARD_MODEL_MAP, RewardModelRankingEntry)


@register_model("rmr", "reward_model_ranking")
class RewardModelRanking(LM):
    def __init__(self, batch_size: Optional[Union[int, str]] = 1, **kwargs) -> None:
        super().__init__()
        self.batch_size = batch_size

        self.rm_kwargs = kwargs if kwargs else {}

        if self.rm_kwargs.get("rm_name", None) is None:
            raise ValueError("Reward model name is not provided.")

        self.rm_name = self.rm_kwargs.get("rm_name", None)
        assert (
            self.rm_name in REWARD_MODEL_MAP.keys()
        ), f"Reward model {self.rm_name} is not supported, supported reward models are {REWARD_MODEL_MAP.keys()}."

        self.models = self.rm_kwargs.get("models", None)
        assert (
            self.models is not None
        ), "Models are not provided. Use | to separate the models."
        self.models = self.models.split("|")

        self.cache_path = os.getenv("HARNESS_HF_CACHE", None)
        assert self.cache_path is not None, "Cache path is not provided. Please set HARNESS_HF_CACHE at the environment variable."

        print(f"Reward model: {self.rm_name}")
        self.reward_model_pipe = RewardModelRankingEntry(
            reward_model_id=self.rm_name, **self.rm_kwargs
        )

        self._cache = {}

    # def get_batched_requests(self, requests: list[Instance], batch_size: int = 64):
    #     inp_list = []
    #     untils = []
    #     for request in [req.args for req in requests]:
    #         inp_list.append(request[0])
    #         untils.extend(request[1]["until"])

    #     batch_size = int(batch_size)
    #     num_batches = (len(inp_list) + batch_size - 1) // batch_size

    #     untils = list(set(untils))
    #     print(f"{untils=}")
    #     return [
    #         list(sub_arr) for sub_arr in np.array_split(inp_list, num_batches)
    #     ], untils

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
        
        print("############### warmup manually ###############")
        rank_result, time_cost = self.reward_model_pipe.rank(
                "Show me what you get.", ['Money', 'Love', 'Name'], top_k=3
        )

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
                candidate_list.append(cache_value['response'])

            instruction = request.args[0].split("\n\n")[-1]
            # print(f"{candidate_list=}")
            rank_result, time_cost = self.reward_model_pipe.rank(
                instruction, candidate_list, top_k=3
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
