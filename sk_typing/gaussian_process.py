from typing import Optional
from typing import Union
from typing import Callable

import numpy as np

from .typing import KernelType
from .typing import Literal
from .typing import RandomStateType


class GaussianProcessClassifier:
    def __init__(
        self,
        kernel: Optional[KernelType] = None,
        optimizer: Union[Literal["fmin_l_bfgs_b"], Callable] = "fmin_l_bfgs_b",
        n_restarts_optimizer: int = 0,
        max_iter_predict: int = 100,
        warm_start: bool = False,
        copy_X_train: bool = True,
        random_state: RandomStateType = None,
        multi_class: Literal["one_vs_rest", "one_vs_one"] = "one_vs_rest",
        n_jobs: Optional[int] = None,
    ):
        ...


class GaussianProcessRegressor:
    def __init__(
        self,
        kernel: Optional[KernelType] = None,
        alpha: Union[float, np.ndarray] = 1e-10,
        optimizer: Union[Literal["fmin_l_bfgs_b"], Callable] = "fmin_l_bfgs_b",
        n_restarts_optimizer: int = 0,
        normalize_y: bool = False,
        copy_X_train: bool = True,
        random_state: RandomStateType = None,
    ):
        ...
