from .typing import Literal


class CCA:
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
    def __init__(self, n_components: int = 2, scale: bool = True, copy: bool = True):
        ...
