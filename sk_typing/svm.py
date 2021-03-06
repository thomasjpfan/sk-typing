from typing import Union

from .typing import Literal
from .typing import RandomStateType


class LinearSVC:
    def __init__(
        self,
        penalty: Literal["l1", "l2"] = "l2",
        loss: Literal["hinge", "squared_hinge"] = "squared_hinge",
        dual: bool = True,
        tol: float = 0.0001,
        C: float = 1.0,
        multi_class: Literal["ovr", "crammer_singer"] = "ovr",
        fit_intercept: bool = True,
        intercept_scaling: float = 1,
        class_weight: Union[dict, Literal["balanced"], None] = None,
        verbose: int = 0,
        random_state: RandomStateType = None,
        max_iter: int = 1000,
    ):
        ...


class LinearSVR:
    def __init__(
        self,
        epsilon: float = 0.0,
        tol: float = 0.0001,
        C: float = 1.0,
        loss: Literal[
            "epsilon_insensitive", "squared_epsilon_insensitive"
        ] = "epsilon_insensitive",
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        dual: bool = True,
        verbose: int = 0,
        random_state: RandomStateType = None,
        max_iter: int = 1000,
    ):
        ...


class NuSVC:
    def __init__(
        self,
        nu: float = 0.5,
        kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf",
        degree: int = 3,
        gamma: Literal["scale", "auto"] = "scale",
        coef0: float = 0.0,
        shrinking: bool = True,
        probability: bool = False,
        tol: float = 0.001,
        cache_size: float = 200,
        class_weight: Union[dict, Literal["balanced"], None] = None,
        verbose: bool = False,
        max_iter: int = -1,
        decision_function_shape: Literal["ovo", "ovr"] = "ovr",
        break_ties: bool = False,
        random_state: RandomStateType = None,
    ):
        ...


class NuSVR:
    def __init__(
        self,
        nu: float = 0.5,
        C: float = 1.0,
        kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf",
        degree: int = 3,
        gamma: Union[Literal["scale", "auto"], float] = "scale",
        coef0: float = 0.0,
        shrinking: bool = True,
        tol: float = 0.001,
        cache_size: float = 200,
        verbose: bool = False,
        max_iter: int = -1,
    ):
        ...


class OneClassSVM:
    def __init__(
        self,
        kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf",
        degree: int = 3,
        gamma: Union[Literal["scale", "auto"], float] = "scale",
        coef0: float = 0.0,
        tol: float = 0.001,
        nu: float = 0.5,
        shrinking: bool = True,
        cache_size: float = 200,
        verbose: bool = False,
        max_iter: int = -1,
    ):
        ...


class SVC:
    def __init__(
        self,
        C: float = 1.0,
        kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf",
        degree: int = 3,
        gamma: Union[Literal["scale", "auto"], float] = "scale",
        coef0: float = 0.0,
        shrinking: bool = True,
        probability: bool = False,
        tol: float = 0.001,
        cache_size: float = 200,
        class_weight: Union[dict, Literal["balanced"], None] = None,
        verbose: bool = False,
        max_iter: int = -1,
        decision_function_shape: Literal["ovo", "ovr"] = "ovr",
        break_ties: bool = False,
        random_state: RandomStateType = None,
    ):
        ...


class SVR:
    def __init__(
        self,
        kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf",
        degree: int = 3,
        gamma: Union[Literal["scale", "auto"], float] = "scale",
        coef0: float = 0.0,
        tol: float = 0.001,
        C: float = 1.0,
        epsilon: float = 0.1,
        shrinking: bool = True,
        cache_size: float = 200,
        verbose: bool = False,
        max_iter: int = -1,
    ):
        ...
