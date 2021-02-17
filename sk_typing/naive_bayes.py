from typing import Optional

from .typing import ArrayLike


class BernoulliNB:
    def __init__(
        self,
        alpha: float = 1.0,
        binarize: Optional[float] = 0.0,
        fit_prior: bool = True,
        class_prior: Optional[ArrayLike] = None,
    ):
        ...


class CategoricalNB:
    def __init__(
        self,
        alpha: float = 1.0,
        fit_prior: bool = True,
        class_prior: Optional[ArrayLike] = None,
    ):
        ...


class ComplementNB:
    def __init__(
        self,
        alpha: float = 1.0,
        fit_prior: bool = True,
        class_prior: Optional[ArrayLike] = None,
        norm: bool = False,
    ):
        ...


class GaussianNB:
    def __init__(
        self, priors: Optional[ArrayLike] = None, var_smoothing: float = 1e-09
    ):
        ...


class MultinomialNB:
    def __init__(
        self,
        alpha: float = 1.0,
        fit_prior: bool = True,
        class_prior: Optional[ArrayLike] = None,
    ):
        ...
