from typing import Optional

from .typing import MemoryType


class FeatureUnion:
    def __init__(
        self,
        transformer_list: list,
        n_jobs: Optional[int] = None,
        transformer_weights: Optional[dict] = None,
        verbose: bool = False,
    ):
        ...


class Pipeline:
    def __init__(self, steps: list, memory: MemoryType = None, verbose: bool = False):
        ...
