import numpy as np
from typing import Optional
from typing import Union
from typing import Callable

from .typing import EstimatorType
from .typing import Literal
from .typing import RandomStateType
from .typing import CVType


class GridSearchCV:
    def __init__(
        self,
        estimator: EstimatorType,
        param_grid: Union[dict, list],
        scoring: Union[str, Callable, list, tuple, dict, None] = None,
        n_jobs: Optional[int] = None,
        refit: Union[bool, str, Callable] = True,
        cv: CVType = None,
        verbose: int = 0,
        pre_dispatch: str = "2*n_jobs",
        iid: Union[bool, Literal["deprecated"]] = "deprecated",
        error_score: Union[Literal["raise"], float] = np.nan,
        return_train_score: bool = False,
    ):
        ...


class RandomizedSearchCV:
    def __init__(
        self,
        estimator: EstimatorType,
        param_distributions: Union[dict, list],
        n_iter: int = 10,
        scoring: Union[str, Callable, list, tuple, dict, None] = None,
        n_jobs: Optional[int] = None,
        refit: Union[bool, str, Callable] = True,
        cv: CVType = None,
        verbose: int = 0,
        pre_dispatch: str = "2*n_jobs",
        iid: Union[bool, Literal["deprecated"]] = "deprecated",
        random_state: RandomStateType = None,
        error_score: Union[Literal["raise"], float] = np.nan,
        return_train_score: bool = False,
    ):
        ...
