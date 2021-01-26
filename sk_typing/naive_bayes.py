from typing import Optional
from typing import Union

from ._typing import ArrayLike

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import MultinomialNB


class BernoulliNBAnnotation:
    __estimator__ = BernoulliNB

    def __init__(
        self,
        alpha: float = 1.0,
        binarize: Optional[float] = 0.0,
        fit_prior: bool = True,
        class_prior: Optional[ArrayLike] = None,
    ):
        pass


class CategoricalNBAnnotation:
    __estimator__ = CategoricalNB

    def __init__(
        self,
        alpha: float = 1.0,
        fit_prior: bool = True,
        class_prior: Optional[ArrayLike] = None,
        min_categories: Union[int, ArrayLike, None] = None,
    ):
        pass


class ComplementNBAnnotation:
    __estimator__ = ComplementNB

    def __init__(
        self,
        alpha: float = 1.0,
        fit_prior: bool = True,
        class_prior: Optional[ArrayLike] = None,
        norm: bool = False,
    ):
        pass


class GaussianNBAnnotation:
    __estimator__ = GaussianNB

    def __init__(
        self, priors: Optional[ArrayLike] = None, var_smoothing: float = 1e-09
    ):
        pass


class MultinomialNBAnnotation:
    __estimator__ = MultinomialNB

    def __init__(
        self,
        alpha: float = 1.0,
        fit_prior: bool = True,
        class_prior: Optional[ArrayLike] = None,
    ):
        pass
