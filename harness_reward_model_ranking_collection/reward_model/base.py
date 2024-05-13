class BaseRewardModel:
    def __init__(self):
        pass

    def get_reward_candidates(
        self, instruction: str, candidates: list[str], top_k: int = 3
    ) -> tuple[list[list[str]], list[list[float]]]:
        raise NotImplementedError
    
    def get_batch_reward_candidates(
        self, instruction: list[str], candidates: list[list[str]], top_k: int = 3
    ) -> tuple[list[list[str]], list[list[float]]]:
        pass
