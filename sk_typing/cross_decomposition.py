import numpy as np
from .typing import Literal


class CCA:
    x_weights_: np.ndarray
    y_weights_: np.ndarray
    x_loadings_: np.ndarray
    y_loadings_: np.ndarray
    x_scores_: np.ndarray
    y_scores_: np.ndarray
    x_rotations_: np.ndarray
    y_rotations_: np.ndarray
    n_iter_: list

    def __init__(
        self,
        n_components: int = 2,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-06,
        copy: bool = True,
    ):
        ...


class PLSCanonical:
    x_weights_: np.ndarray
    y_weights_: np.ndarray
    x_loadings_: np.ndarray
    y_loadings_: np.ndarray
    x_scores_: np.ndarray
    y_scores_: np.ndarray
    x_rotations_: np.ndarray
    y_rotations_: np.ndarray
    n_iter_: list

    def __init__(
        self,
        n_components: int = 2,
        scale: bool = True,
        algorithm: Literal["nipals", "svd"] = "nipals",
        max_iter: int = 500,
        tol: float = 1e-06,
        copy: bool = True,
    ):
        ...


class PLSRegression:
    x_weights_: np.ndarray
    y_weights_: np.ndarray
    x_loadings_: np.ndarray
    y_loadings_: np.ndarray
    x_scores_: np.ndarray
    y_scores_: np.ndarray
    x_rotations_: np.ndarray
    y_rotations_: np.ndarray
    coef_: np.ndarray
    n_iter_: list

    def __init__(
        self,
        n_components: int = 2,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-06,
        copy: bool = True,
    ):
        ...


class PLSSVD:
    x_weights_: np.ndarray
    y_weights_: np.ndarray
    x_scores_: np.ndarray
    y_scores_: list

    def __init__(self, n_components: int = 2, scale: bool = True, copy: bool = True):
        ...
