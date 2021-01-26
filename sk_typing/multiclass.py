from typing import Optional

from ._typing import EstimatorType
from ._typing import RandomStateType

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier


class OneVsOneClassifierAnnotation:
    __estimator__ = OneVsOneClassifier

    def __init__(self, estimator: EstimatorType, n_jobs: Optional[int] = None):
        pass


class OneVsRestClassifierAnnotation:
    __estimator__ = OneVsRestClassifier

    def __init__(self, estimator: EstimatorType, n_jobs: Optional[int] = None):
        pass


class OutputCodeClassifierAnnotation:
    __estimator__ = OutputCodeClassifier

    def __init__(
        self,
        estimator: EstimatorType,
        code_size: float = 1.5,
        random_state: RandomStateType = None,
        n_jobs: Optional[int] = None,
    ):
        pass
