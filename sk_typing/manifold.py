from typing import Optional
from typing import Union
from typing import Callable

from .typing import Literal
from .typing import RandomStateType
from .typing import NDArray


class Isomap:
    def __init__(
        self,
        n_neighbors: int = 5,
        n_components: int = 2,
        eigen_solver: Literal["auto", "arpack", "dense"] = "auto",
        tol: float = 0,
        max_iter: Optional[int] = None,
        path_method: Literal["auto", "FW", "D"] = "auto",
        neighbors_algorithm: Literal["auto", "brute", "kd_tree", "ball_tree"] = "auto",
        n_jobs: Optional[int] = None,
        metric: Union[str, Callable] = "minkowski",
        p: int = 2,
        metric_params: Optional[dict] = None,
    ):
        ...


class LocallyLinearEmbedding:
    def __init__(
        self,
        n_neighbors: int = 5,
        n_components: int = 2,
        reg: float = 0.001,
        eigen_solver: Literal["auto", "arpack", "dense"] = "auto",
        tol: float = 1e-06,
        max_iter: int = 100,
        method: Literal["standard", "hessian", "modified", "ltsa"] = "standard",
        hessian_tol: float = 0.0001,
        modified_tol: float = 1e-12,
        neighbors_algorithm: Literal["auto", "brute", "kd_tree", "ball_tree"] = "auto",
        random_state: RandomStateType = None,
        n_jobs: Optional[int] = None,
    ):
        ...


class MDS:
    def __init__(
        self,
        n_components: int = 2,
        metric: bool = True,
        n_init: int = 4,
        max_iter: int = 300,
        verbose: int = 0,
        eps: float = 0.001,
        n_jobs: Optional[int] = None,
        random_state: RandomStateType = None,
        dissimilarity: Literal["euclidean", "precomputed"] = "euclidean",
    ):
        ...


class SpectralEmbedding:
    def __init__(
        self,
        n_components: int = 2,
        affinity: Literal[
            "nearest_neighbors", "rbf", "precomputed", "precomputed_nearest_neighbors"
        ] = "nearest_neighbors",
        gamma: Optional[float] = None,
        random_state: RandomStateType = None,
        eigen_solver: Literal["arpack", "logpcg", "amg"] = None,
        n_neighbors: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        ...


class TSNE:
    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        early_exaggeration: float = 12.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        n_iter_without_progress: int = 300,
        min_grad_norm: float = 1e-07,
        metric: Union[str, Callable] = "euclidean",
        init: Union[Literal["random", "pca"], NDArray] = "random",
        verbose: int = 0,
        random_state: RandomStateType = None,
        method: str = "barnes_hut",
        angle: float = 0.5,
        n_jobs: Optional[int] = None,
        square_distances: Union[bool, Literal["legacy"]] = "legacy",
    ):
        ...
