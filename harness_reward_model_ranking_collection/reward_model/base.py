class BaseRewardModel:
    def __init__(self):
        pass

    def get_reward_candidates(
        self, instruction: str, candidates: list[str], top_k: int = 3
    ) -> list[str]:
        raise NotImplementedError
