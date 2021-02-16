from typing import Optional

from .typing import EstimatorType
from .typing import RandomStateType


class OneVsOneClassifier:
    def __init__(self, estimator: EstimatorType, n_jobs: Optional[int] = None):
        ...


class OneVsRestClassifier:
    def __init__(self, estimator: EstimatorType, n_jobs: Optional[int] = None):
        ...


class OutputCodeClassifier:
    def __init__(
        self,
        estimator: EstimatorType,
        code_size: float = 1.5,
        random_state: RandomStateType = None,
        n_jobs: Optional[int] = None,
    ):
        ...
