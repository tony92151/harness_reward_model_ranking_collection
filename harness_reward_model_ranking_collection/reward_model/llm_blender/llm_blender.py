import sys
import time

from ..base import BaseRewardModel

sys.path.append("/home/azureuser/LLM-Blender-harness")

import llm_blender
from llm_blender.blender.blender_utils import get_topk_candidates_from_ranks


def init_llm_blender(use_fuser: bool) -> llm_blender.Blender:
    # gf = GenFuserConfig(model_name="meta-llama/Llama-2-7b-hf")

    blender = llm_blender.Blender()
    # Load Ranker
    blender.loadranker("llm-blender/PairRM")  # load ranker checkpoint
    # blender.loadranker("OpenAssistant/reward-model-deberta-v3-large-v2") # load ranker checkpoint
    # Load Fuser
    if use_fuser:
        blender.loadfuser(
            "llm-blender/gen_fuser_3b"
        )  # load fuser checkpoint if you want to use pre-trained fuser; or you can use ranker only
    return blender


class LlmBlenderPipe(BaseRewardModel):
    def __init__(self, **kwargs):
        self.use_fuser = False
        if "use_fuser" in kwargs and kwargs["use_fuser"]:
            self.use_fuser = True
            print(f"{self.use_fuser=}")

        self.max_new_tokens = 128
        if "max_new_tokens" in kwargs:
            self.max_new_tokens = int(kwargs["max_new_tokens"])

        self.llm_blender = init_llm_blender(use_fuser=self.use_fuser)


        self.batch_size = kwargs.get("batch_size", 4)
        # self.stop_sequences = kwargs.get("stop_sequences", [])

    def get_reward_candidates(
        self, instruction: str, candidates: list[str], top_k: int = 3,  **kwargs
    ) -> tuple[list[str], list[float]]:
        stop_sequences = kwargs.get("stop_sequences", [])
        
        t = time.time()
        ranks = self.llm_blender.rank(
            [instruction],
            [candidates],
            # instructions=None,
            return_scores=False,
            batch_size=1,
            disable_tqdm=True,
        )

        topk_candidates = get_topk_candidates_from_ranks(
            ranks, [candidates], top_k=top_k
        )
        topk_candidates_list = topk_candidates[0].tolist()

        if self.use_fuser:
            generate_kwargs = {"max_new_tokens": self.max_new_tokens}
            fuse_generations = self.llm_blender.fuse(
                [instruction],
                topk_candidates,
                instructions=None,
                batch_size=1,
                stop_sequences=stop_sequences,
                **generate_kwargs
            )
            topk_candidates_list = [fuse_generations[0]]

        time_cost = time.time() - t

        return topk_candidates_list, [time_cost, time_cost]
    
    def get_batch_reward_candidates(self, instruction: list[str], candidates: list[list[str]], top_k: int = 3, **kwargs) -> tuple[list[list[str]], list[list[float]]]:
        stop_sequences = kwargs.get("stop_sequences", [])
        
        t = time.time()
        ranks = self.llm_blender.rank(
            instruction,
            candidates,
            # instructions=None,
            return_scores=False,
            batch_size=self.batch_size,
            disable_tqdm=True,
        )

        topk_candidates = get_topk_candidates_from_ranks(
            ranks, candidates, top_k=top_k
        )
        batch_topk_candidates_list = []
        for topk_candidate in topk_candidates:
            batch_topk_candidates_list.append(topk_candidate.tolist())

        if self.use_fuser:
            generate_kwargs = {"max_new_tokens": self.max_new_tokens}
            fuse_generations = self.llm_blender.fuse(
                instruction,
                topk_candidates,
                instructions=None,
                batch_size=self.batch_size,
                stop_sequences=stop_sequences,
                **generate_kwargs
            )
            batch_topk_candidates_list = [[fuse_generation]for fuse_generation in fuse_generations]

        time_cost = time.time() - t

        return batch_topk_candidates_list, [[time_cost, time_cost] for _ in batch_topk_candidates_list]
