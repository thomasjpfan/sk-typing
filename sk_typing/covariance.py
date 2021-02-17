from typing import Optional
from typing import Union

import numpy as np

from .typing import RandomStateType
from .typing import ArrayLike
from .typing import CVType
from .typing import Literal


class EllipticEnvelope:
    location_: np.ndarray
    covariance_: np.ndarray
    precision_: np.ndarray
    support_: np.ndarray
    offset_: float

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
    location_: np.ndarray
    covariance_: np.ndarray
    precision_: np.ndarray

    def __init__(self, store_precision: bool = True, assume_centered: bool = False):
        ...


class GraphicalLasso:
    location_: np.ndarray
    covariance_: np.ndarray
    precision_: np.ndarray
    n_iter_: int

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
    location_: np.ndarray
    covariance_: np.ndarray
    precision_: np.ndarray
    alpha_: float
    cv_alphas_: list
    grid_scores_: np.ndarray
    n_iter_: int

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
    location_: np.ndarray
    covariance_: np.ndarray
    precision_: np.ndarray
    shrinkage_: float

    def __init__(
        self,
        store_precision: bool = True,
        assume_centered: bool = False,
        block_size: int = 1000,
    ):
        ...


class MinCovDet:
    raw_location_: np.ndarray
    raw_covariance_: np.ndarray
    raw_support_: np.ndarray
    location_: np.ndarray
    covariance_: np.ndarray
    precision_: np.ndarray
    support_: np.ndarray
    dist_: np.ndarray

    def __init__(
        self,
        store_precision: bool = True,
        assume_centered: bool = False,
        support_fraction: Optional[float] = None,
        random_state: RandomStateType = None,
    ):
        ...


class OAS:
    covariance_: np.ndarray
    precision_: np.ndarray
    shrinkage_: float

    def __init__(self, store_precision: bool = True, assume_centered: bool = False):
        ...


class ShrunkCovariance:
    location_: np.ndarray
    covariance_: np.ndarray
    precision_: np.ndarray

    def __init__(
        self,
        store_precision: bool = True,
        assume_centered: bool = False,
        shrinkage: float = 0.1,
    ):
        ...
