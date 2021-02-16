from typing import Optional
from typing import Union

from .typing import EstimatorType
from .typing import Literal
from .typing import RandomStateType
from .typing import ArrayLike
from .typing import CVType


class ClassifierChain:
    def __init__(
        self,
        base_estimator: EstimatorType,
        order: Union[Literal["random"], ArrayLike, None] = None,
        cv: CVType = None,
        random_state: RandomStateType = None,
    ):
        ...


class MultiOutputClassifier:
    def __init__(self, estimator: EstimatorType, n_jobs: Optional[int] = None):
        ...


class MultiOutputRegressor:
    def __init__(self, estimator: EstimatorType, n_jobs: Optional[int] = None):
        ...


class RegressorChain:
    def __init__(
        self,
        base_estimator: EstimatorType,
        order: Union[ArrayLike, Literal["random"], None] = None,
        cv: CVType = None,
        random_state: RandomStateType = None,
    ):
        ...
