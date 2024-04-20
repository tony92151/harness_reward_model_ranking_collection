class BaseRewardModel:
    def __init__(self):
        pass

    def get_reward_candidates(
        self, candidates: list[dict], top_k: int = 3
    ) -> list[dict]:
        raise NotImplementedError
