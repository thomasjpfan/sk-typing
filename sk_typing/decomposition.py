from typing import Optional
from typing import Literal
from typing import Union
from collections.abc import Callable

from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FastICA
from sklearn.decomposition import MiniBatchSparsePCA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import SparseCoder

from ._typing import NDArary
from ._typing import RandomState


class DictionaryLearningAnnotation:
    __estimator__ = DictionaryLearning

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
        code_init: Optional[NDArary] = None,
        dict_init: Optional[NDArary] = None,
        verbose: bool = False,
        split_sign: bool = False,
        random_state: RandomState = None,
        positive_code: bool = False,
        positive_dict: bool = False,
        transform_max_iter: int = 1000,
    ):
        pass


class FactorAnalysisAnnotation:
    __estimator__ = FactorAnalysis

    def __init__(
        self,
        n_components: Optional[int] = None,
        tol: float = 0.01,
        copy: bool = True,
        max_iter: int = 1000,
        noise_variance_init: Optional[NDArary] = None,
        svd_method: Literal["lapack", "randomized"] = "randomized",
        iterated_power: int = 3,
        rotation: Literal["varimax", "quartimax"] = None,
        random_state: RandomState = 0,
    ):
        pass


class FastICAAnnotation:
    __estimator__ = FastICA

    def __init__(
        self,
        n_components: Optional[int] = None,
        algorithm: Literal["parallel", "deflation"] = "parallel",
        whiten: bool = True,
        fun: Union[Literal["logcosh", "exp", "cube"], Callable] = "logcosh",
        fun_args: Optional[dict] = None,
        max_iter: int = 200,
        tol: float = 0.0001,
        w_init: Optional[NDArary] = None,
        random_state: RandomState = None,
    ):
        pass


class IncrementalPCAAnnotation:
    __estimator__ = IncrementalPCA

    def __init__(
        self,
        n_components: Optional[int] = None,
        whiten: bool = False,
        copy: bool = True,
        batch_size: Optional[int] = None,
    ):
        pass


class KernelPCAAnnotation:
    __estimator__ = KernelPCA

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
        random_state: RandomState = None,
        copy_X: bool = True,
        n_jobs: Optional[int] = None,
    ):
        pass


class LatentDirichletAllocationAnnotation:
    __estimator__ = LatentDirichletAllocation

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
        total_samples: int = 1000000.0,
        perp_tol: float = 0.1,
        mean_change_tol: float = 0.001,
        max_doc_update_iter: int = 100,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: RandomState = None,
    ):
        pass


class MiniBatchDictionaryLearningAnnotation:
    __estimator__ = MiniBatchDictionaryLearning

    def __init__(
        self,
        n_components: Optional[None] = None,
        alpha: float = 1,
        n_iter: int = 1000,
        fit_algorithm: Literal["lars", "cd"] = "lars",
        n_jobs: Optional[int] = None,
        batch_size: int = 3,
        shuffle: bool = True,
        dict_init: Optional[NDArary] = None,
        transform_algorithm: Literal[
            "lasso_lars", "lasso_cd", "lars", "omp", "threshold"
        ] = "omp",
        transform_n_nonzero_coefs: Optional[int] = None,
        transform_alpha: Optional[float] = None,
        verbose: bool = False,
        split_sign: bool = False,
        random_state: RandomState = None,
        positive_code: bool = False,
        positive_dict: bool = False,
        transform_max_iter: int = 1000,
    ):
        pass


class MiniBatchSparsePCAAnnotation:
    __estimator__ = MiniBatchSparsePCA

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
        random_state: RandomState = None,
    ):
        pass


class NMFAnnotation:
    __estimator__ = NMF

    def __init__(
        self,
        n_components: Optional[int] = None,
        init: Optional[
            Literal["random", "nndsvd", "nndsvda", "nndsvdar", "custom", "warn"]
        ] = "warn",
        solver: Literal["cd", "mu"] = "cd",
        beta_loss: Union[
            float, Literal["frobenius", "kullback-leibler", "itakura-saito"]
        ] = "frobenius",
        tol: float = 0.0001,
        max_iter: int = 200,
        random_state: RandomState = None,
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
        verbose: bool = 0,
        shuffle: bool = False,
        regularization: Literal["both", "components", "transformation", None] = "both",
    ):
        pass


class PCAAnnotation:
    __estimator__ = PCA

    def __init__(
        self,
        n_components: Union[int, float, None, Literal["mle"]] = None,
        copy: bool = True,
        whiten: bool = False,
        svd_solver: Literal["auto", "full", "arpack", "randomized"] = "auto",
        tol: float = 0.0,
        iterated_power: Union[int, Literal["auto"]] = "auto",
        random_state: RandomState = None,
    ):
        pass


class SparseCoderAnnotation:
    __estimator__ = SparseCoder

    def __init__(
        self,
        dictionary: NDArary,
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
        pass


class SparsePCAAnnotation:
    __estimator__ = SparsePCA

    def __init__(
        self,
        n_components: Optional[int] = None,
        alpha: float = 1,
        ridge_alpha: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-08,
        method: Literal["lars", "cd"] = "lars",
        n_jobs: Optional[int] = None,
        U_init: Optional[NDArary] = None,
        V_init: Optional[NDArary] = None,
        verbose: Union[int, bool] = False,
        random_state: RandomState = None,
    ):
        pass


class TruncatedSVDAnnotation:
    __estimator__ = TruncatedSVD

    def __init__(
        self,
        n_components: int = 2,
        algorithm: Literal["arpack", "randomized"] = "randomized",
        n_iter: int = 5,
        random_state: RandomState = None,
        tol: float = 0.0,
    ):
        pass
