from typing import Optional
from typing import Literal
from typing import Union

from sklearn.covariance import OAS
from sklearn.covariance import GraphicalLasso
from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance import GraphicalLassoCV
from sklearn.covariance import MinCovDet
from sklearn.covariance import ShrunkCovariance
from sklearn.covariance import LedoitWolf
from sklearn.covariance import EllipticEnvelope

from ._typing import RandomState
from ._typing import ArrayLike
from ._typing import CVType


class EllipticEnvelopeAnnotation:
    __estimator__ = EllipticEnvelope

    def __init__(
        self,
        store_precision: bool = True,
        assume_centered: bool = False,
        support_fraction: Optional[float] = None,
        contamination: float = 0.1,
        random_state: RandomState = None,
    ):
        pass


class EmpiricalCovarianceAnnotation:
    __estimator__ = EmpiricalCovariance

    def __init__(self, store_precision: bool = True, assume_centered: bool = False):
        pass


class GraphicalLassoAnnotation:
    __estimator__ = GraphicalLasso

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
        pass


class GraphicalLassoCVAnnotation:
    __estimator__ = GraphicalLassoCV

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
        pass


class LedoitWolfAnnotation:
    __estimator__ = LedoitWolf

    def __init__(
        self,
        store_precision: bool = True,
        assume_centered: bool = False,
        block_size: int = 1000,
    ):
        pass


class MinCovDetAnnotation:
    __estimator__ = MinCovDet

    def __init__(
        self,
        store_precision: bool = True,
        assume_centered: bool = False,
        support_fraction: Optional[float] = None,
        random_state: RandomState = None,
    ):
        pass


class OASAnnotation:
    __estimator__ = OAS

    def __init__(self, store_precision: bool = True, assume_centered: bool = False):
        pass


class ShrunkCovarianceAnnotation:
    __estimator__ = ShrunkCovariance

    def __init__(
        self,
        store_precision: bool = True,
        assume_centered: bool = False,
        shrinkage: float = 0.1,
    ):
        pass
