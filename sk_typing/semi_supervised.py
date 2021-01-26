from typing import Optional
from typing import Union
from typing import Callable

from ._typing import EstimatorType
from ._typing import Literal

from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import SelfTrainingClassifier


class LabelPropagationAnnotation:
    __estimator__ = LabelPropagation

    def __init__(
        self,
        kernel: Union[Literal["knn", "rbf"], Callable] = "rbf",
        gamma: float = 20,
        n_neighbors: int = 7,
        max_iter: int = 1000,
        tol: float = 0.001,
        n_jobs: Optional[int] = None,
    ):
        pass


class LabelSpreadingAnnotation:
    __estimator__ = LabelSpreading

    def __init__(
        self,
        kernel: Union[Literal["knn", "rbf"], Callable] = "rbf",
        gamma: float = 20,
        n_neighbors: int = 7,
        alpha: float = 0.2,
        max_iter: int = 30,
        tol: float = 0.001,
        n_jobs: Optional[int] = None,
    ):
        pass


class SelfTrainingClassifierAnnotation:
    __estimator__ = SelfTrainingClassifier

    def __init__(
        self,
        base_estimator: EstimatorType,
        threshold: float = 0.75,
        criterion: Literal["threshold", "k_best"] = "threshold",
        k_best: int = 10,
        max_iter: Optional[int] = 10,
        verbose: bool = False,
    ):
        pass
