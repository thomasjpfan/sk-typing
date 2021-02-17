from typing import Union
from typing import Optional
from collections.abc import Callable
import numpy as np

from .typing import ArrayLike
from .typing import RandomStateType
from .typing import MemoryType
from .typing import EstimatorType
from .typing import Literal


class AffinityPropagation:
    def __init__(
        self,
        damping: float = 0.5,
        max_iter: int = 200,
        convergence_iter: int = 15,
        copy: bool = True,
        preference: Union[ArrayLike, float, None] = None,
        affinity: Literal["euclidean", "precomputed"] = "euclidean",
        verbose: bool = False,
    ):
        ...


class AgglomerativeClustering:
    def __init__(
        self,
        n_clusters: Optional[int] = 2,
        affinity: Union[str, Callable] = "euclidean",
        memory: MemoryType = None,
        connectivity: Union[ArrayLike, Callable, None] = None,
        compute_full_tree: Union[Literal["auto"], bool] = "auto",
        linkage: Literal["ward", "complete", "average", "single"] = "ward",
        distance_threshold: float = None,
    ):
        ...


class Birch:
    def __init__(
        self,
        threshold: float = 0.5,
        branching_factor: int = 50,
        n_clusters: Union[int, EstimatorType, None] = 3,
        compute_labels: bool = True,
        copy: bool = True,
    ):
        ...


class DBSCAN:
    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: Union[str, Callable] = "euclidean",
        metric_params: Optional[dict] = None,
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        p: Optional[float] = None,
        n_jobs: Optional[int] = None,
    ):
        ...


class FeatureAgglomeration:
    def __init__(
        self,
        n_clusters: int = 2,
        affinity: Union[str, Callable] = "euclidean",
        memory: MemoryType = None,
        connectivity: Union[ArrayLike, Callable] = None,
        compute_full_tree: Union[Literal["auto"], bool] = "auto",
        linkage: Literal["ward", "complete", "average", "single"] = "ward",
        pooling_func: Callable = np.mean,
        distance_threshold: Optional[float] = None,
        compute_distances: bool = False,
    ):
        ...


class KMeans:
    def __init__(
        self,
        n_clusters: int = 8,
        init: Union[Literal["k-means++", "random"], Callable, ArrayLike] = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 0.0001,
        precompute_distances: Union[Literal["auto", "deprecated"], bool] = "deprecated",
        verbose: int = 0,
        random_state: RandomStateType = None,
        copy_x: bool = True,
        algorithm: Literal["auto", "full", "elkan"] = "auto",
    ):
        ...


class MeanShift:
    def __init__(
        self,
        bandwidth: Optional[None] = None,
        seeds: Union[ArrayLike, None] = None,
        bin_seeding: bool = False,
        min_bin_freq: int = 1,
        cluster_all: bool = True,
        n_jobs: Optional[int] = None,
        max_iter: int = 300,
    ):
        ...


class MiniBatchKMeans:
    def __init__(
        self,
        n_clusters: int = 8,
        init: Union[Literal["k-means++", "random"], Callable, ArrayLike] = "k-means++",
        max_iter: int = 100,
        batch_size: int = 100,
        verbose: int = 0,
        compute_labels: bool = True,
        random_state: RandomStateType = None,
        tol: float = 0.0,
        max_no_improvement: int = 10,
        init_size: Optional[int] = None,
        n_init: int = 3,
        reassignment_ratio: float = 0.01,
    ):
        ...


class OPTICS:
    def __init__(
        self,
        min_samples: Union[int, float] = 5,
        max_eps: float = np.inf,
        metric: Union[str, Callable] = "minkowski",
        p: int = 2,
        metric_params: dict = None,
        cluster_method: Literal["xi", "dbscan"] = "xi",
        eps: Optional[float] = None,
        xi: float = 0.05,
        predecessor_correction: bool = True,
        min_cluster_size: Union[int, float] = None,
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        n_jobs: Optional[int] = None,
    ):
        ...


class SpectralBiclustering:
    def __init__(
        self,
        n_clusters: Union[int, tuple] = 3,
        method: Literal["bistochastic", "scale", "log"] = "bistochastic",
        n_components: int = 6,
        n_best: int = 3,
        svd_method: Literal["randomized", "arpack"] = "randomized",
        n_svd_vecs: Optional[int] = None,
        mini_batch: bool = False,
        init: Union[Literal["k-means++", "random"], np.ndarray] = "k-means++",
        n_init: int = 10,
        n_jobs: Union[int, None, Literal["deprecated"]] = "deprecated",
        random_state: RandomStateType = None,
    ):
        ...


class SpectralClustering:
    def __init__(
        self,
        n_clusters: int = 8,
        eigen_solver: Optional[Literal["arpack", "lobpcg", "amg"]] = None,
        n_components: Optional[int] = None,
        random_state: RandomStateType = None,
        n_init: int = 10,
        gamma: float = 1.0,
        affinity: Union[
            Literal[
                "nearest_neighbors",
                "rbf",
                "precomputed",
                "precomputed_nearest_neighbors",
            ],
            Callable,
        ] = "rbf",
        n_neighbors: int = 10,
        eigen_tol: float = 0.0,
        assign_labels: Literal["kmeans", "discretize"] = "kmeans",
        degree: float = 3,
        coef0: float = 1,
        kernel_params: dict = None,
        n_jobs: Optional[int] = None,
    ):
        ...


class SpectralCoclustering:
    def __init__(
        self,
        n_clusters: int = 3,
        svd_method: Literal["randomized", "arpack"] = "randomized",
        n_svd_vecs: Optional[int] = None,
        mini_batch: bool = False,
        init: Union[Literal["k-means++", "random"], np.ndarray] = "k-means++",
        n_init: int = 10,
        n_jobs: Union[int, None, Literal["deprecated"]] = "deprecated",
        random_state: RandomStateType = None,
    ):
        ...
