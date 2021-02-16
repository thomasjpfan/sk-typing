from typing import Optional
from typing import Union
from typing import Callable

from sklearn.feature_selection import f_classif

from .typing import EstimatorType
from .typing import Literal
from .typing import CVType


class GenericUnivariateSelect:
    def __init__(
        self,
        score_func: Callable = f_classif,
        mode: Literal["percentile", "k_best", "fpr", "fdr", "fwe"] = "percentile",
        param: Union[float, int] = 1e-05,
    ):
        ...


class RFE:
    def __init__(
        self,
        estimator: EstimatorType,
        n_features_to_select: Union[int, float, None] = None,
        step: Union[int, float] = 1,
        verbose: int = 0,
        importance_getter: Union[str, Callable] = "auto",
    ):
        ...


class RFECV:
    def __init__(
        self,
        estimator: EstimatorType,
        step: Union[int, float] = 1,
        min_features_to_select: int = 1,
        cv: CVType = None,
        scoring: Union[str, Callable, None] = None,
        verbose: int = 0,
        n_jobs: Union[int, None] = None,
        importance_getter: Union[str, Callable] = "auto",
    ):
        ...


class SelectFdr:
    def __init__(self, score_func: Callable = f_classif, alpha: float = 0.05):
        ...


class SelectFpr:
    def __init__(self, score_func: Callable = f_classif, alpha: float = 0.05):
        ...


class SelectFromModel:
    def __init__(
        self,
        estimator: EstimatorType,
        threshold: Union[str, float, None] = None,
        prefit: bool = False,
        norm_order: Union[int, float] = 1,
        max_features: Optional[None] = None,
        importance_getter: Union[str, Callable] = "auto",
    ):
        ...


class SelectFwe:
    def __init__(self, score_func: Callable = f_classif, alpha: float = 0.05):
        ...


class SelectKBest:
    def __init__(
        self, score_func: Callable = f_classif, k: Union[Literal["all"], int] = 10
    ):
        ...


class SelectPercentile:
    def __init__(self, score_func: Callable = f_classif, percentile: int = 10):
        ...


class SequentialFeatureSelector:
    def __init__(
        self,
        estimator: EstimatorType,
        n_features_to_select: Union[int, float, None] = None,
        direction: Literal["forward", "backward"] = "forward",
        scoring: Union[str, Callable, list, tuple, dict, None] = None,
        cv: CVType = 5,
        n_jobs: Optional[int] = None,
    ):
        ...


class VarianceThreshold:
    def __init__(self, threshold: float = 0.0):
        ...
