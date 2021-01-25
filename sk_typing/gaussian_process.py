from typing import Optional
from typing import Union
from typing import Callable

from ._typing import KernelType
from ._typing import Literal
from ._typing import RandomStateType
from ._typing import NDArray

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier


class GaussianProcessClassifierAnnotation:
    __estimator__ = GaussianProcessClassifier

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
        pass


class GaussianProcessRegressorAnnotation:
    __estimator__ = GaussianProcessRegressor

    def __init__(
        self,
        kernel: Optional[KernelType] = None,
        alpha: Union[float, NDArray] = 1e-10,
        optimizer: Union[Literal["fmin_l_bfgs_b"], Callable] = "fmin_l_bfgs_b",
        n_restarts_optimizer: int = 0,
        normalize_y: bool = False,
        copy_X_train: bool = True,
        random_state: RandomStateType = None,
    ):
        pass
