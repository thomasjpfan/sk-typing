from typing import Optional

from ._typing import MemoryType

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline


class FeatureUnionAnnotation:
    __estimator__ = FeatureUnion

    def __init__(
        self,
        transformer_list: list,
        n_jobs: Optional[int] = None,
        transformer_weights: Optional[dict] = None,
        verbose: bool = False,
    ):
        pass


class PipelineAnnotation:
    __estimator__ = Pipeline

    def __init__(self, steps: list, memory: MemoryType = None, verbose: bool = False):
        pass
