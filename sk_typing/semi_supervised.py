from typing import Optional
from typing import Union
from typing import Callable

from .typing import EstimatorType
from .typing import Literal


class LabelPropagation:
    def __init__(
        self,
        kernel: Union[Literal["knn", "rbf"], Callable] = "rbf",
        gamma: float = 20,
        n_neighbors: int = 7,
        max_iter: int = 1000,
        tol: float = 0.001,
        n_jobs: Optional[int] = None,
    ):
        ...


class LabelSpreading:
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
        ...


class SelfTrainingClassifier:
    def __init__(
        self,
        base_estimator: EstimatorType,
        threshold: float = 0.75,
        criterion: Literal["threshold", "k_best"] = "threshold",
        k_best: int = 10,
        max_iter: Optional[int] = 10,
        verbose: bool = False,
    ):
        ...
