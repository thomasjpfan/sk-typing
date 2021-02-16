from typing import Union

from .typing import Literal
from .typing import RandomStateType


class BernoulliRBM:
    def __init__(
        self,
        n_components: int = 256,
        learning_rate: float = 0.1,
        batch_size: int = 10,
        n_iter: int = 10,
        verbose: int = 0,
        random_state: RandomStateType = None,
    ):
        ...


class MLPClassifier:
    def __init__(
        self,
        hidden_layer_sizes: tuple = (100,),
        activation: Literal["identity", "logistic", "tanh", "relu"] = "relu",
        solver: Literal["lbfgs", "sgd", "adam"] = "adam",
        alpha: float = 0.0001,
        batch_size: Union[int, Literal["auto"]] = "auto",
        learning_rate: Literal["constant", "invscaling", "adaptive"] = "constant",
        learning_rate_init: float = 0.001,
        power_t: float = 0.5,
        max_iter: int = 200,
        shuffle: bool = True,
        random_state: RandomStateType = None,
        tol: float = 0.0001,
        verbose: bool = False,
        warm_start: bool = False,
        momentum: float = 0.9,
        nesterovs_momentum: bool = True,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-08,
        n_iter_no_change: int = 10,
        max_fun: int = 15000,
    ):
        ...


class MLPRegressor:
    def __init__(
        self,
        hidden_layer_sizes: tuple = (100,),
        activation: Literal["identity", "logistic", "tanh", "relu"] = "relu",
        solver: Literal["lbfgs", "sgd", "adam"] = "adam",
        alpha: float = 0.0001,
        batch_size: Union[int, Literal["auto"]] = "auto",
        learning_rate: Literal["constant", "invscaling", "adaptive"] = "constant",
        learning_rate_init: float = 0.001,
        power_t: float = 0.5,
        max_iter: int = 200,
        shuffle: bool = True,
        random_state: RandomStateType = None,
        tol: float = 0.0001,
        verbose: bool = False,
        warm_start: bool = False,
        momentum: float = 0.9,
        nesterovs_momentum: bool = True,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-08,
        n_iter_no_change: int = 10,
        max_fun: int = 15000,
    ):
        ...
