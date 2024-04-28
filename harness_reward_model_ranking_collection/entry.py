from .reward_model import AutoJPipe, OasstrmPipe, UltrarmPipe

REWARD_MODEL_MAP = {
    "ultrarm": UltrarmPipe,
    "autoj": AutoJPipe,
    "oassterm": OasstrmPipe,
}


class RewardModelRankingEntry:
    def __init__(self, reward_model_id: str, **kwargs):
        if reward_model_id not in REWARD_MODEL_MAP.keys():
            raise ValueError(
                "Invalid reward_model_id. Supported reward models are: ",
                REWARD_MODEL_MAP.keys(),
            )
        self.reward_model_id = reward_model_id

        reward_model_pipe = REWARD_MODEL_MAP[reward_model_id]

        self.pipe = reward_model_pipe(**kwargs)

    def rank(
        self, instruction: str, candidates: list[str], top_k: int = 3, **kwargs
    ) -> tuple[list[str], list[float]]:
        return self.pipe.get_reward_candidates(instruction, candidates, top_k, **kwargs)
