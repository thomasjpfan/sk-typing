from typing import Optional
from typing import Union
from typing import Callable

from ._typing import Literal
from ._typing import RandomStateType

from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import RadiusNeighborsTransformer
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import RadiusNeighborsRegressor


class KNeighborsClassifierAnnotation:
    __estimator__ = KNeighborsClassifier

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Union[Literal["uniform", "distance"], Callable] = "uniform",
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: Union[str, Callable] = "minkowski",
        metric_params: Optional[dict] = None,
        n_jobs: Optional[int] = None,
        **kwargs: dict
    ):
        pass


class KNeighborsRegressorAnnotation:
    __estimator__ = KNeighborsRegressor

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Union[Literal["uniform", "distance"], Callable] = "uniform",
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: Union[str, Callable] = "minkowski",
        metric_params: Optional[dict] = None,
        n_jobs: Optional[int] = None,
        **kwargs: dict
    ):
        pass


class KNeighborsTransformerAnnotation:
    __estimator__ = KNeighborsTransformer

    def __init__(
        self,
        mode: Literal["distance", "connectivity"] = "distance",
        n_neighbors: int = 5,
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        metric: Union[str, Callable] = "minkowski",
        p: int = 2,
        metric_params: Optional[dict] = None,
        n_jobs: int = 1,
    ):
        pass


class KernelDensityAnnotation:
    __estimator__ = KernelDensity

    def __init__(
        self,
        bandwidth: float = 1.0,
        algorithm: Literal["kd_tree", "ball_tree", "auto"] = "auto",
        kernel: Literal[
            "gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"
        ] = "gaussian",
        metric: str = "euclidean",
        atol: float = 0,
        rtol: float = 0,
        breadth_first: bool = True,
        leaf_size: int = 40,
        metric_params: Optional[dict] = None,
    ):
        pass


class LocalOutlierFactorAnnotation:
    __estimator__ = LocalOutlierFactor

    def __init__(
        self,
        n_neighbors: int = 20,
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        metric: Union[str, Callable] = "minkowski",
        p: int = 2,
        metric_params: Union[dict] = None,
        contamination: Union[Literal["auto"], float] = "auto",
        novelty: bool = False,
        n_jobs: Optional[int] = None,
    ):
        pass


class NearestCentroidAnnotation:
    __estimator__ = NearestCentroid

    def __init__(
        self,
        metric: Union[str, Callable] = "euclidean",
        shrink_threshold: Optional[float] = None,
    ):
        pass


class NearestNeighborsAnnotation:
    __estimator__ = NearestNeighbors

    def __init__(
        self,
        n_neighbors: int = 5,
        radius: float = 1.0,
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        metric: Union[str, Callable] = "minkowski",
        p: int = 2,
        metric_params: Optional[dict] = None,
        n_jobs: Optional[int] = None,
    ):
        pass


class NeighborhoodComponentsAnalysisAnnotation:
    __estimator__ = NeighborhoodComponentsAnalysis

    def __init__(
        self,
        n_components: Optional[int] = None,
        init: Literal["auto", "pca", "lda", "identity", "random"] = "auto",
        warm_start: bool = False,
        max_iter: int = 50,
        tol: float = 1e-05,
        callback: Optional[Callable] = None,
        verbose: int = 0,
        random_state: RandomStateType = None,
    ):
        pass


class RadiusNeighborsClassifierAnnotation:
    __estimator__ = RadiusNeighborsClassifier

    def __init__(
        self,
        radius: float = 1.0,
        weights: Union[Literal["uniform", "distance"], Callable] = "uniform",
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: Union[str, Callable] = "minkowski",
        outlier_label: Union[str, Literal["most_frequent"], None] = None,
        metric_params: Optional[dict] = None,
        n_jobs: Optional[int] = None,
        **kwargs: dict
    ):
        pass


class RadiusNeighborsRegressorAnnotation:
    __estimator__ = RadiusNeighborsRegressor

    def __init__(
        self,
        radius: float = 1.0,
        weights: Union[Literal["uniform", "distance"], Callable] = "uniform",
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: Union[str, Callable] = "minkowski",
        metric_params: Optional[dict] = None,
        n_jobs: Optional[int] = None,
        **kwargs: dict
    ):
        pass


class RadiusNeighborsTransformerAnnotation:
    __estimator__ = RadiusNeighborsTransformer

    def __init__(
        self,
        mode: Literal["distance", "connectivity"] = "distance",
        radius: float = 1.0,
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        metric: Union[str, Callable] = "minkowski",
        p: int = 2,
        metric_params: Optional[dict] = None,
        n_jobs: Optional[int] = 1,
    ):
        pass
