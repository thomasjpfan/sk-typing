from sklearn.cross_decomposition import PLSCanonical
from sklearn.cross_decomposition import PLSRegression
from sklearn.cross_decomposition import PLSSVD
from sklearn.cross_decomposition import CCA

from ._typing import Literal


class CCAAnnotation:
    __estimator__ = CCA

    def __init__(
        self,
        n_components: int = 2,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-06,
        copy: bool = True,
    ):
        pass


class PLSCanonicalAnnotation:
    __estimator__ = PLSCanonical

    def __init__(
        self,
        n_components: int = 2,
        scale: bool = True,
        algorithm: Literal["nipals", "svd"] = "nipals",
        max_iter: int = 500,
        tol: float = 1e-06,
        copy: bool = True,
    ):
        pass


class PLSRegressionAnnotation:
    __estimator__ = PLSRegression

    def __init__(
        self,
        n_components: int = 2,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-06,
        copy: bool = True,
    ):
        pass


class PLSSVDAnnotation:
    __estimator__ = PLSSVD

    def __init__(self, n_components: int = 2, scale: bool = True, copy: bool = True):
        pass
