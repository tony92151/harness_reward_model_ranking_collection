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


@register_model("llm_blender", "llmblender")
class RewardModelRanking(LM):
    def __init__(self, batch_size: Optional[Union[int, str]] = 1, **kwargs) -> None:
        super().__init__()
        self.batch_size = batch_size

        self.rm_kwargs = kwargs if kwargs else {}

        self.rm_name = "llm_blender"

        self.models = self.rm_kwargs.get("models", None)
        assert (
            self.models is not None
        ), "Models are not provided. Use | to separate the models."
        self.models = self.models.split("|")

        self.cache_path = os.getenv("HARNESS_HF_CACHE", None)
        assert (
            self.cache_path is not None
        ), "Cache path is not provided. Please set HARNESS_HF_CACHE at the environment variable."

        self.rm_kwargs["use_fuser"] = True

        print(f"Reward model: {self.rm_name}")
        self.reward_model_pipe = RewardModelRankingEntry(
            reward_model_id=self.rm_name, **self.rm_kwargs
        )

        self._cache = {}

        self.batch_size = 1

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
    def get_batched_requests(self, requests, batch_size: int = 64):
        inp_list = []
        untils = []
        for request in [req.args for req in requests]:
            inp_list.append(request[0])
            untils.extend(request[1]["until"])

        batch_size = int(batch_size)
        num_batches = (len(inp_list) + batch_size - 1) // batch_size

        untils = list(set(untils))
        print(f"{untils=}")
        return [
            list(sub_arr) for sub_arr in np.array_split(inp_list, num_batches)
        ], untils

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
        _, _ = self.reward_model_pipe.rank(
            "Show me what you get.", ["Money", "Love", "Name"], top_k=3
        )

        batch_request = self.get_batched_requests(requests, self.batch_size)

        cache = {}
        total_results = []
        for requests in tqdm(batch_request):
            batch_candidate_list: list[list[str]] = []
            batch_instruction: list[str] = []

            for request in requests:
                candidate_list: list[str] = []
                for target_model in self.models:
                    cache_value = self._get_cache(
                        target_model, os.getenv("TASK"), request
                    )
                    assert (
                        cache_value is not None
                    ), f"Cache value for {target_model} is not found. Please check the cache file."
                    candidate_list.append(cache_value["response"])

                batch_candidate_list.append(candidate_list)
                instruction = request.split("\n\n")[-1]
                batch_instruction.append(instruction)

            rank_results, time_costs = self.reward_model_pipe.batch_rank(
                batch_instruction, batch_instruction, top_k=3
            )
            final_results = [r[0] for r in rank_results]
            total_results.extend(final_results)

            batch_final_candidate_model = []
            for candidate_list, final_result in zip(
                batch_candidate_list, final_results
            ):
                if final_result in candidate_list:
                    batch_final_candidate_model.append(
                        self.models[candidate_list.index(final_result)]
                    )
                else:
                    batch_final_candidate_model.append("not found")

            for (
                request,
                candidate_list,
                final_result,
                final_candidate_model,
                time_cost,
            ) in zip(
                requests,
                batch_candidate_list,
                final_results,
                batch_final_candidate_model,
                time_costs,
            ):
                cache[request] = {
                    "candidate_dict": {
                        m: {"text": c, "latency": t}
                        for m, c, t in zip(self.models, candidate_list, time_cost)
                    },
                    "final_candidate_model": final_candidate_model,
                    "final_rank_result": final_result,
                }

        if os.getenv("TASK") and self.cache_path:
            os.makedirs(self.cache_path, exist_ok=True)

            combine_models_ = "_".join(
                [target_model.replace("/", "-") for target_model in self.models]
            )
            save_path = os.path.join(
                self.cache_path,
                f'{self.rm_name}_fuse_{os.getenv("TASK")}_{combine_models_}.pt',
            )
            print(f"Save to cache {save_path}")
            torch.save(cache, save_path)

        return total_results
