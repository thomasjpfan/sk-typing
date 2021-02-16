from typing import Optional
from typing import Union

from .typing import Literal
from .typing import RandomStateType
from .typing import ArrayLike


class BayesianGaussianMixture:
    def __init__(
        self,
        n_components: int = 1,
        covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
        tol: float = 0.001,
        reg_covar: float = 1e-06,
        max_iter: float = 100,
        n_init: int = 1,
        init_params: Literal["kmeans", "random"] = "kmeans",
        weight_concentration_prior_type: Literal[
            "dirichlet_process", "dirichlet_distribution"
        ] = "dirichlet_process",
        weight_concentration_prior: Optional[float] = None,
        mean_precision_prior: Optional[float] = None,
        mean_prior: Optional[ArrayLike] = None,
        degrees_of_freedom_prior: Optional[float] = None,
        covariance_prior: Union[float, ArrayLike, None] = None,
        random_state: RandomStateType = None,
        warm_start: bool = False,
        verbose: int = 0,
        verbose_interval: int = 10,
    ):
        ...


class GaussianMixture:
    def __init__(
        self,
        n_components: int = 1,
        covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
        tol: float = 0.001,
        reg_covar: float = 1e-06,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: Literal["kmeans", "random"] = "kmeans",
        weights_init: Optional[ArrayLike] = None,
        means_init: Optional[ArrayLike] = None,
        precisions_init: Optional[ArrayLike] = None,
        random_state: RandomStateType = None,
        warm_start: bool = False,
        verbose: int = 0,
        verbose_interval: int = 10,
    ):
        ...
