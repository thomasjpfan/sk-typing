from typing import Optional
from typing import Union
from typing import Callable

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import f_classif

from ._typing import EstimatorType
from ._typing import Literal
from ._typing import CVType


class GenericUnivariateSelectAnnotation:
    __estimator__ = GenericUnivariateSelect

    def __init__(
        self,
        score_func: Callable = f_classif,
        mode: Literal["percentile", "k_best", "fpr", "fdr", "fwe"] = "percentile",
        param: Union[float, int] = 1e-05,
    ):
        pass


class RFEAnnotation:
    __estimator__ = RFE

    def __init__(
        self,
        estimator: EstimatorType,
        n_features_to_select: Union[int, float, None] = None,
        step: Union[int, float] = 1,
        verbose: int = 0,
        importance_getter: Union[str, Callable] = "auto",
    ):
        pass


class RFECVAnnotation:
    __estimator__ = RFECV

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
        pass


class SelectFdrAnnotation:
    __estimator__ = SelectFdr

    def __init__(self, score_func: Callable = f_classif, alpha: float = 0.05):
        pass


class SelectFprAnnotation:
    __estimator__ = SelectFpr

    def __init__(self, score_func: Callable = f_classif, alpha: float = 0.05):
        pass


class SelectFromModelAnnotation:
    __estimator__ = SelectFromModel

    def __init__(
        self,
        estimator: EstimatorType,
        threshold: Union[str, float, None] = None,
        prefit: bool = False,
        norm_order: Union[int, float] = 1,
        max_features: Optional[None] = None,
        importance_getter: Union[str, Callable] = "auto",
    ):
        pass


class SelectFweAnnotation:
    __estimator__ = SelectFwe

    def __init__(self, score_func: Callable = f_classif, alpha: float = 0.05):
        pass


class SelectKBestAnnotation:
    __estimator__ = SelectKBest

    def __init__(
        self, score_func: Callable = f_classif, k: Union[Literal["all"], int] = 10
    ):
        pass


class SelectPercentileAnnotation:
    __estimator__ = SelectPercentile

    def __init__(self, score_func: Callable = f_classif, percentile: int = 10):
        pass


class SequentialFeatureSelectorAnnotation:
    __estimator__ = SequentialFeatureSelector

    def __init__(
        self,
        estimator: EstimatorType,
        n_features_to_select: Union[int, float, None] = None,
        direction: Literal["forward", "backward"] = "forward",
        scoring: Union[str, Callable, list, tuple, dict, None] = None,
        cv: CVType = 5,
        n_jobs: Optional[int] = None,
    ):
        pass


class VarianceThresholdAnnotation:
    __estimator__ = VarianceThreshold

    def __init__(self, threshold: float = 0.0):
        pass
