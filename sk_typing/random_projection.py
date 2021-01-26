from typing import Union

from ._typing import Literal
from ._typing import RandomStateType

from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import GaussianRandomProjection


class GaussianRandomProjectionAnnotation:
    __estimator__ = GaussianRandomProjection

    def __init__(
        self,
        n_components: Union[int, Literal["auto"]] = "auto",
        eps: float = 0.1,
        random_state: RandomStateType = None,
    ):
        pass


class SparseRandomProjectionAnnotation:
    __estimator__ = SparseRandomProjection

    def __init__(
        self,
        n_components: Union[int, Literal["auto"]] = "auto",
        density: Union[float, Literal["auto"]] = "auto",
        eps: float = 0.1,
        dense_output: bool = False,
        random_state: RandomStateType = None,
    ):
        pass
