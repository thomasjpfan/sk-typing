from typing import Union

from .typing import Literal
from .typing import RandomStateType


class GaussianRandomProjection:
    def __init__(
        self,
        n_components: Union[int, Literal["auto"]] = "auto",
        eps: float = 0.1,
        random_state: RandomStateType = None,
    ):
        ...


class SparseRandomProjection:
    def __init__(
        self,
        n_components: Union[int, Literal["auto"]] = "auto",
        density: Union[float, Literal["auto"]] = "auto",
        eps: float = 0.1,
        dense_output: bool = False,
        random_state: RandomStateType = None,
    ):
        ...
