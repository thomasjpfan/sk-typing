from typing import Optional
from typing import Union

from .typing import RandomStateType
from .typing import ArrayLike
from .typing import CVType
from .typing import Literal


class EllipticEnvelope:
    def __init__(
        self,
        store_precision: bool = True,
        assume_centered: bool = False,
        support_fraction: Optional[float] = None,
        contamination: float = 0.1,
        random_state: RandomStateType = None,
    ):
        ...


class EmpiricalCovariance:
    def __init__(self, store_precision: bool = True, assume_centered: bool = False):
        ...


class GraphicalLasso:
    def __init__(
        self,
        alpha: float = 0.01,
        mode: Literal["cd", "lars"] = "cd",
        tol: float = 0.0001,
        enet_tol: float = 0.0001,
        max_iter: int = 100,
        verbose: bool = False,
        assume_centered: bool = False,
    ):
        ...


class GraphicalLassoCV:
    def __init__(
        self,
        alphas: Union[int, ArrayLike] = 4,
        n_refinements: int = 4,
        cv: CVType = None,
        tol: float = 0.0001,
        enet_tol: float = 0.0001,
        max_iter: int = 100,
        mode: Literal["cd", "lars"] = "cd",
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        assume_centered: bool = False,
    ):
        ...


class LedoitWolf:
    def __init__(
        self,
        store_precision: bool = True,
        assume_centered: bool = False,
        block_size: int = 1000,
    ):
        ...


class MinCovDet:
    def __init__(
        self,
        store_precision: bool = True,
        assume_centered: bool = False,
        support_fraction: Optional[float] = None,
        random_state: RandomStateType = None,
    ):
        ...


class OAS:
    def __init__(self, store_precision: bool = True, assume_centered: bool = False):
        ...


class ShrunkCovariance:
    def __init__(
        self,
        store_precision: bool = True,
        assume_centered: bool = False,
        shrinkage: float = 0.1,
    ):
        ...
