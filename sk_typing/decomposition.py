from typing import Optional
from typing import Union
from collections.abc import Callable

import numpy as np

from .typing import RandomStateType
from .typing import Literal


class DictionaryLearning:
    components_: np.ndarray
    error_: np.ndarray
    n_iter_: int

    def __init__(
        self,
        n_components: Optional[int] = None,
        alpha: float = 1,
        max_iter: int = 1000,
        tol: float = 1e-08,
        fit_algorithm: Literal["lars", "cd"] = "lars",
        transform_algorithm: Literal[
            "lasso_lars", "lasso_cd", "lars", "omp", "threshold"
        ] = "omp",
        transform_n_nonzero_coefs: Optional[int] = None,
        transform_alpha: Optional[float] = None,
        n_jobs: Optional[int] = None,
        code_init: Optional[np.ndarray] = None,
        dict_init: Optional[np.ndarray] = None,
        verbose: bool = False,
        split_sign: bool = False,
        random_state: RandomStateType = None,
        positive_code: bool = False,
        positive_dict: bool = False,
        transform_max_iter: int = 1000,
    ):
        ...


class FactorAnalysis:
    components_: np.ndarray
    loglike_: list
    noise_variance_: np.ndarray
    n_iter_: int
    mean_: np.ndarray

    def __init__(
        self,
        n_components: Optional[int] = None,
        tol: float = 0.01,
        copy: bool = True,
        max_iter: int = 1000,
        noise_variance_init: Optional[np.ndarray] = None,
        svd_method: Literal["lapack", "randomized"] = "randomized",
        iterated_power: int = 3,
        random_state: RandomStateType = 0,
    ):
        ...


class FastICA:
    components_: np.ndarray
    mixing_: np.ndarray
    mean_: np.ndarray
    n_iter_: int
    whitening_: np.ndarray

    def __init__(
        self,
        n_components: Optional[int] = None,
        algorithm: Literal["parallel", "deflation"] = "parallel",
        whiten: bool = True,
        fun: Union[Literal["logcosh", "exp", "cube"], Callable] = "logcosh",
        fun_args: Optional[dict] = None,
        max_iter: int = 200,
        tol: float = 0.0001,
        w_init: Optional[np.ndarray] = None,
        random_state: RandomStateType = None,
    ):
        ...


class IncrementalPCA:
    components_: np.ndarray
    explained_variance_: np.ndarray
    explained_variance_ratio_: np.ndarray
    singular_values_: np.ndarray
    mean_: np.ndarray
    var_: np.ndarray
    noise_variance_: float
    n_components_: int
    n_samples_seen_: int

    def __init__(
        self,
        n_components: Optional[int] = None,
        whiten: bool = False,
        copy: bool = True,
        batch_size: Optional[int] = None,
    ):
        ...


class KernelPCA:
    lambdas_: np.ndarray
    alphas_: np.ndarray
    dual_coef_: np.ndarray
    X_transformed_fit_: np.ndarray
    X_fit_: np.ndarray

    def __init__(
        self,
        n_components: Optional[None] = None,
        kernel: Literal[
            "linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"
        ] = "linear",
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1,
        kernel_params: Optional[dict] = None,
        alpha: float = 1.0,
        fit_inverse_transform: bool = False,
        eigen_solver: Literal["auto", "dense", "arpack"] = "auto",
        tol: float = 0,
        max_iter: Optional[None] = None,
        remove_zero_eig: bool = False,
        random_state: RandomStateType = None,
        copy_X: bool = True,
        n_jobs: Optional[int] = None,
    ):
        ...


class LatentDirichletAllocation:
    components_: np.ndarray
    n_batch_iter_: int
    n_iter_: int
    bound_: float
    doc_topic_prior_: float
    topic_word_prior_: float

    def __init__(
        self,
        n_components: int = 10,
        doc_topic_prior: Optional[float] = None,
        topic_word_prior: Optional[float] = None,
        learning_method: Literal["batch", "online"] = "batch",
        learning_decay: float = 0.7,
        learning_offset: float = 10.0,
        max_iter: int = 10,
        batch_size: int = 128,
        evaluate_every: int = -1,
        total_samples: int = 1_000_000,
        perp_tol: float = 0.1,
        mean_change_tol: float = 0.001,
        max_doc_update_iter: int = 100,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: RandomStateType = None,
    ):
        ...


class MiniBatchDictionaryLearning:
    components_: np.ndarray
    inner_stats_: tuple
    n_iter_: int
    iter_offset_: int
    random_state_: np.random.RandomState

    def __init__(
        self,
        n_components: Optional[None] = None,
        alpha: float = 1,
        n_iter: int = 1000,
        fit_algorithm: Literal["lars", "cd"] = "lars",
        n_jobs: Optional[int] = None,
        batch_size: int = 3,
        shuffle: bool = True,
        dict_init: Optional[np.ndarray] = None,
        transform_algorithm: Literal[
            "lasso_lars", "lasso_cd", "lars", "omp", "threshold"
        ] = "omp",
        transform_n_nonzero_coefs: Optional[int] = None,
        transform_alpha: Optional[float] = None,
        verbose: bool = False,
        split_sign: bool = False,
        random_state: RandomStateType = None,
        positive_code: bool = False,
        positive_dict: bool = False,
        transform_max_iter: int = 1000,
    ):
        ...


class MiniBatchSparsePCA:
    components_: np.ndarray
    n_iter_: int
    mean_: np.ndarray

    def __init__(
        self,
        n_components: Optional[int] = None,
        alpha: int = 1,
        ridge_alpha: float = 0.01,
        n_iter: int = 100,
        callback: Optional[Callable] = None,
        batch_size: int = 3,
        verbose: Union[int, bool] = False,
        shuffle: bool = True,
        n_jobs: Optional[int] = None,
        method: Literal["lars", "cd"] = "lars",
        random_state: RandomStateType = None,
        normalize_components: str = "deprecated",
    ):
        ...


class NMF:
    components_: np.ndarray
    n_components_: int
    reconstruction_err_: float
    n_iter_: int

    def __init__(
        self,
        n_components: Optional[int] = None,
        init: Optional[
            Literal["random", "nndsvd", "nndsvda", "nndsvdar", "custom", "warn"]
        ] = None,
        solver: Literal["cd", "mu"] = "cd",
        beta_loss: Union[
            float, Literal["frobenius", "kullback-leibler", "itakura-saito"]
        ] = "frobenius",
        tol: float = 0.0001,
        max_iter: int = 200,
        random_state: RandomStateType = None,
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
        verbose: int = 0,
        shuffle: bool = False,
    ):
        ...


class PCA:
    components_: np.ndarray
    explained_variance_: np.ndarray
    explained_variance_ratio_: np.ndarray
    singular_values_: np.ndarray
    mean_: np.ndarray
    n_components_: np.ndarray
    n_features_: int
    n_samples_: int
    noise_variance_: float

    def __init__(
        self,
        n_components: Union[int, float, None, Literal["mle"]] = None,
        copy: bool = True,
        whiten: bool = False,
        svd_solver: Literal["auto", "full", "arpack", "randomized"] = "auto",
        tol: float = 0.0,
        iterated_power: Union[int, Literal["auto"]] = "auto",
        random_state: RandomStateType = None,
    ):
        ...


class SparseCoder:
    components_: np.ndarray

    def __init__(
        self,
        dictionary: np.ndarray,
        transform_algorithm: Literal[
            "lasso_lars", "lasso_cd", "lars", "omp", "threshold"
        ] = "omp",
        transform_n_nonzero_coefs: Optional[int] = None,
        transform_alpha: Optional[float] = None,
        split_sign: bool = False,
        n_jobs: Optional[int] = None,
        positive_code: bool = False,
        transform_max_iter: int = 1000,
    ):
        ...


class SparsePCA:
    components_: np.ndarray
    error_: np.ndarray
    n_iter_: int
    mean_: np.ndarray

    def __init__(
        self,
        n_components: Optional[int] = None,
        alpha: float = 1,
        ridge_alpha: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-08,
        method: Literal["lars", "cd"] = "lars",
        n_jobs: Optional[int] = None,
        U_init: Optional[np.ndarray] = None,
        V_init: Optional[np.ndarray] = None,
        verbose: Union[int, bool] = False,
        random_state: RandomStateType = None,
        normalize_components: str = "deprecated",
    ):
        ...


class TruncatedSVD:
    components_: np.ndarray
    explained_variance_: np.ndarray
    explained_variance_ratio_: np.ndarray
    singular_values_: np.ndarray

    def __init__(
        self,
        n_components: int = 2,
        algorithm: Literal["arpack", "randomized"] = "randomized",
        n_iter: int = 5,
        random_state: RandomStateType = None,
        tol: float = 0.0,
    ):
        ...
