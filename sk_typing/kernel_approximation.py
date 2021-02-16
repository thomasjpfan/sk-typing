from typing import Optional
from typing import Union
from typing import Callable

from .typing import RandomStateType


class AdditiveChi2Sampler:
    def __init__(self, sample_steps: int = 2, sample_interval: Optional[float] = None):
        ...


class Nystroem:
    def __init__(
        self,
        kernel: Union[str, Callable] = "rbf",
        gamma: Optional[float] = None,
        coef0: Optional[float] = None,
        degree: Optional[float] = None,
        kernel_params: Optional[dict] = None,
        n_components: int = 100,
        random_state: RandomStateType = None,
        n_jobs: Optional[int] = None,
    ):
        ...


class PolynomialCountSketch:
    def __init__(
        self,
        gamma: float = 1.0,
        degree: int = 2,
        coef0: int = 0,
        n_components: int = 100,
        random_state: RandomStateType = None,
    ):
        ...


class RBFSampler:
    def __init__(
        self,
        gamma: float = 1.0,
        n_components: int = 100,
        random_state: RandomStateType = None,
    ):
        ...


class SkewedChi2Sampler:
    def __init__(
        self,
        skewedness: float = 1.0,
        n_components: int = 100,
        random_state: RandomStateType = None,
    ):
        ...
