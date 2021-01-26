import numpy as np
from typing import Optional
from typing import Union
from typing import Callable

from ._typing import EstimatorType
from ._typing import Literal
from ._typing import RandomStateType
from ._typing import ArrayLike
from ._typing import CVType
from ._typing import NDArray

from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import LarsCV
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LassoLars
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import GammaRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import Lasso
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Perceptron
from sklearn.linear_model import OrthogonalMatchingPursuitCV


class ARDRegressionAnnotation:
    __estimator__ = ARDRegression

    def __init__(
        self,
        n_iter: int = 300,
        tol: float = 0.001,
        alpha_1: float = 1e-06,
        alpha_2: float = 1e-06,
        lambda_1: float = 1e-06,
        lambda_2: float = 1e-06,
        compute_score: bool = False,
        threshold_lambda: float = 10000.0,
        fit_intercept: bool = True,
        normalize: bool = False,
        copy_X: bool = True,
        verbose: bool = False,
    ):
        pass


class BayesianRidgeAnnotation:
    __estimator__ = BayesianRidge

    def __init__(
        self,
        n_iter: int = 300,
        tol: float = 0.001,
        alpha_1: float = 1e-06,
        alpha_2: float = 1e-06,
        lambda_1: float = 1e-06,
        lambda_2: float = 1e-06,
        alpha_init: float = None,
        lambda_init: float = None,
        compute_score: bool = False,
        fit_intercept: bool = True,
        normalize: bool = False,
        copy_X: bool = True,
        verbose: bool = False,
    ):
        pass


class ElasticNetAnnotation:
    __estimator__ = ElasticNet

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        normalize: bool = False,
        precompute: Union[bool, ArrayLike] = False,
        max_iter: int = 1000,
        copy_X: bool = True,
        tol: float = 0.0001,
        warm_start: bool = False,
        positive: bool = False,
        random_state: RandomStateType = None,
        selection: Literal["cyclic", "random"] = "cyclic",
    ):
        pass


class ElasticNetCVAnnotation:
    __estimator__ = ElasticNetCV

    def __init__(
        self,
        l1_ratio: Union[float, list] = 0.5,
        eps: float = 0.001,
        n_alphas: int = 100,
        alphas: Optional[NDArray] = None,
        fit_intercept: bool = True,
        normalize: bool = False,
        precompute: Union[Literal["auto"], bool, ArrayLike] = "auto",
        max_iter: int = 1000,
        tol: float = 0.0001,
        cv: CVType = None,
        copy_X: bool = True,
        verbose: Union[bool, int] = 0,
        n_jobs: Optional[int] = None,
        positive: bool = False,
        random_state: RandomStateType = None,
        selection: Literal["cyclic", "random"] = "cyclic",
    ):
        pass


class GammaRegressorAnnotation:
    __estimator__ = GammaRegressor

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 0.0001,
        warm_start: bool = False,
        verbose: int = 0,
    ):
        pass


class HuberRegressorAnnotation:
    __estimator__ = HuberRegressor

    def __init__(
        self,
        epsilon: float = 1.35,
        max_iter: int = 100,
        alpha: float = 0.0001,
        warm_start: bool = False,
        fit_intercept: bool = True,
        tol: float = 1e-05,
    ):
        pass


class LarsAnnotation:
    __estimator__ = Lars

    def __init__(
        self,
        fit_intercept: bool = True,
        verbose: Union[bool, int] = False,
        normalize: bool = True,
        precompute: Union[bool, Literal["auto"], ArrayLike] = "auto",
        n_nonzero_coefs: int = 500,
        eps: float = 2.220446049250313e-16,
        copy_X: bool = True,
        fit_path: bool = True,
        jitter: Optional[int] = None,
        random_state: RandomStateType = None,
    ):
        pass


class LarsCVAnnotation:
    __estimator__ = LarsCV

    def __init__(
        self,
        fit_intercept: bool = True,
        verbose: bool = False,
        max_iter: int = 500,
        normalize: bool = True,
        precompute: Union[bool, Literal["auto"], ArrayLike] = "auto",
        cv: CVType = None,
        max_n_alphas: int = 1000,
        n_jobs: Optional[int] = None,
        eps: float = 2.220446049250313e-16,
        copy_X: bool = True,
    ):
        pass


class LassoAnnotation:
    __estimator__ = Lasso

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        normalize: bool = False,
        precompute: Union[Literal["auto"], bool, ArrayLike] = False,
        copy_X: bool = True,
        max_iter: int = 1000,
        tol: float = 0.0001,
        warm_start: bool = False,
        positive: bool = False,
        random_state: RandomStateType = None,
        selection: Literal["cyclic", "random"] = "cyclic",
    ):
        pass


class LassoCVAnnotation:
    __estimator__ = LassoCV

    def __init__(
        self,
        eps: float = 0.001,
        n_alphas: int = 100,
        alphas: Optional[NDArray] = None,
        fit_intercept: bool = True,
        normalize: bool = False,
        precompute: Union[Literal["auto"], bool, ArrayLike] = "auto",
        max_iter: int = 1000,
        tol: float = 0.0001,
        copy_X: bool = True,
        cv: CVType = None,
        verbose: Union[bool, int] = False,
        n_jobs: int = None,
        positive: bool = False,
        random_state: RandomStateType = None,
        selection: Literal["cyclic", "random"] = "cyclic",
    ):
        pass


class LassoLarsAnnotation:
    __estimator__ = LassoLars

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        verbose: Union[bool, int] = False,
        normalize: bool = True,
        precompute: Union[bool, Literal["auto"], ArrayLike] = "auto",
        max_iter: int = 500,
        eps: float = 2.220446049250313e-16,
        copy_X: bool = True,
        fit_path: bool = True,
        positive: bool = False,
        jitter: Optional[float] = None,
        random_state: RandomStateType = None,
    ):
        pass


class LassoLarsCVAnnotation:
    __estimator__ = LassoLarsCV

    def __init__(
        self,
        fit_intercept: bool = True,
        verbose: Union[bool, int] = False,
        max_iter: int = 500,
        normalize: bool = True,
        precompute: Union[bool, Literal["auto"]] = "auto",
        cv: CVType = None,
        max_n_alphas: int = 1000,
        n_jobs: Optional[int] = None,
        eps: float = 2.220446049250313e-16,
        copy_X: bool = True,
        positive: bool = False,
    ):
        pass


class LassoLarsICAnnotation:
    __estimator__ = LassoLarsIC

    def __init__(
        self,
        criterion: Literal["bic", "aic"] = "aic",
        fit_intercept: bool = True,
        verbose: Union[bool, int] = False,
        normalize: bool = True,
        precompute: Union[bool, Literal["auto"], ArrayLike] = "auto",
        max_iter: int = 500,
        eps: float = 2.220446049250313e-16,
        copy_X: bool = True,
        positive: bool = False,
    ):
        pass


class LinearRegressionAnnotation:
    __estimator__ = LinearRegression

    def __init__(
        self,
        fit_intercept: bool = True,
        normalize: bool = False,
        copy_X: bool = True,
        n_jobs: Optional[int] = None,
        positive: bool = False,
    ):
        pass


class LogisticRegressionAnnotation:
    __estimator__ = LogisticRegression

    def __init__(
        self,
        penalty: Literal["l1", "l2", "elasticnet", "none"] = "l2",
        dual: bool = False,
        tol: float = 0.0001,
        C: float = 1.0,
        fit_intercept: bool = True,
        intercept_scaling: bool = 1,
        class_weight: Union[dict, Literal["balanced"]] = None,
        random_state: RandomStateType = None,
        solver: Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"] = "lbfgs",
        max_iter: int = 100,
        multi_class: Literal["auto", "ovr", "multinomial"] = "auto",
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: Optional[int] = None,
        l1_ratio: Optional[float] = None,
    ):
        pass


class LogisticRegressionCVAnnotation:
    __estimator__ = LogisticRegressionCV

    def __init__(
        self,
        Cs: Union[int, list] = 10,
        fit_intercept: bool = True,
        cv: CVType = None,
        dual: bool = False,
        penalty: Literal["l1", "l2", "elasticnet"] = "l2",
        scoring: Union[str, Callable, None] = None,
        solver: Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"] = "lbfgs",
        tol: float = 0.0001,
        max_iter: int = 100,
        class_weight: Union[dict, Literal["balanced"], None] = None,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        refit: bool = True,
        intercept_scaling: float = 1.0,
        multi_class: Literal["auto", "ovr", "multinomial"] = "auto",
        random_state: RandomStateType = None,
        l1_ratios: Optional[list] = None,
    ):
        pass


class MultiTaskElasticNetAnnotation:
    __estimator__ = MultiTaskElasticNet

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        normalize: bool = False,
        copy_X: bool = True,
        max_iter: int = 1000,
        tol: float = 0.0001,
        warm_start: bool = False,
        random_state: RandomStateType = None,
        selection: Literal["cyclic", "random"] = "cyclic",
    ):
        pass


class MultiTaskElasticNetCVAnnotation:
    __estimator__ = MultiTaskElasticNetCV

    def __init__(
        self,
        l1_ratio: Union[float, list] = 0.5,
        eps: float = 0.001,
        n_alphas: int = 100,
        alphas: ArrayLike = None,
        fit_intercept: bool = True,
        normalize: bool = False,
        max_iter: int = 1000,
        tol: float = 0.0001,
        cv: CVType = None,
        copy_X: bool = True,
        verbose: Union[bool, int] = 0,
        n_jobs: Optional[int] = None,
        random_state: RandomStateType = None,
        selection: Literal["cyclic", "random"] = "cyclic",
    ):
        pass


class MultiTaskLassoAnnotation:
    __estimator__ = MultiTaskLasso

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        normalize: bool = False,
        copy_X: bool = True,
        max_iter: int = 1000,
        tol: float = 0.0001,
        warm_start: bool = False,
        random_state: RandomStateType = None,
        selection: Literal["cyclic", "random"] = "cyclic",
    ):
        pass


class MultiTaskLassoCVAnnotation:
    __estimator__ = MultiTaskLassoCV

    def __init__(
        self,
        eps: float = 0.001,
        n_alphas: int = 100,
        alphas: Optional[ArrayLike] = None,
        fit_intercept: bool = True,
        normalize: bool = False,
        max_iter: int = 1000,
        tol: float = 0.0001,
        copy_X: bool = True,
        cv: CVType = None,
        verbose: bool = False,
        n_jobs: Optional[int] = None,
        random_state: RandomStateType = None,
        selection: Literal["cyclic", "random"] = "cyclic",
    ):
        pass


class OrthogonalMatchingPursuitAnnotation:
    __estimator__ = OrthogonalMatchingPursuit

    def __init__(
        self,
        n_nonzero_coefs: Optional[int] = None,
        tol: Optional[float] = None,
        fit_intercept: bool = True,
        normalize: bool = True,
        precompute: Union[Literal["auto"], bool] = "auto",
    ):
        pass


class OrthogonalMatchingPursuitCVAnnotation:
    __estimator__ = OrthogonalMatchingPursuitCV

    def __init__(
        self,
        copy: bool = True,
        fit_intercept: bool = True,
        normalize: bool = True,
        max_iter: Optional[int] = None,
        cv: CVType = None,
        n_jobs: int = None,
        verbose: Union[bool, int] = False,
    ):
        pass


class PassiveAggressiveClassifierAnnotation:
    __estimator__ = PassiveAggressiveClassifier

    def __init__(
        self,
        C: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: Optional[float] = 0.001,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 5,
        shuffle: bool = True,
        verbose: int = 0,
        loss: str = "hinge",
        n_jobs: Optional[int] = None,
        random_state: RandomStateType = None,
        warm_start: bool = False,
        class_weight: Union[dict, Literal["balanced"], None] = None,
        average: Union[bool, int] = False,
    ):
        pass


class PassiveAggressiveRegressorAnnotation:
    __estimator__ = PassiveAggressiveRegressor

    def __init__(
        self,
        C: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 0.001,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 5,
        shuffle: bool = True,
        verbose: int = 0,
        loss: str = "epsilon_insensitive",
        epsilon: float = 0.1,
        random_state: int = None,
        warm_start: bool = False,
        average: Union[bool, int] = False,
    ):
        pass


class PerceptronAnnotation:
    __estimator__ = Perceptron

    def __init__(
        self,
        penalty: Optional[Literal["l2", "l1", "elasticnet"]] = None,
        alpha: float = 0.0001,
        l1_ratio: float = 0.15,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 0.001,
        shuffle: bool = True,
        verbose: int = 0,
        eta0: float = 1.0,
        n_jobs: Optional[int] = None,
        random_state: RandomStateType = 0,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 5,
        class_weight: Union[dict, Literal["balanced"], None] = None,
        warm_start: bool = False,
    ):
        pass


class PoissonRegressorAnnotation:
    __estimator__ = PoissonRegressor

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 0.0001,
        warm_start: bool = False,
        verbose: int = 0,
    ):
        pass


class RANSACRegressorAnnotation:
    __estimator__ = RANSACRegressor

    def __init__(
        self,
        base_estimator: Optional[EstimatorType] = None,
        min_samples: Union[int, float, None] = None,
        residual_threshold: Optional[float] = None,
        is_data_valid: Optional[Callable] = None,
        is_model_valid: Optional[Callable] = None,
        max_trials: int = 100,
        max_skips: float = np.inf,
        stop_n_inliers: float = np.inf,
        stop_score: float = np.inf,
        stop_probability: float = 0.99,
        loss: Union[
            Literal["absolute_loss", "squared_loss"], Callable
        ] = "absolute_loss",
        random_state: RandomStateType = None,
    ):
        pass


class RidgeAnnotation:
    __estimator__ = Ridge

    def __init__(
        self,
        alpha: Union[float, NDArray] = 1.0,
        fit_intercept: bool = True,
        normalize: bool = False,
        copy_X: bool = True,
        max_iter: Optional[int] = None,
        tol: float = 0.001,
        solver: Literal[
            "auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"
        ] = "auto",
        random_state: RandomStateType = None,
    ):
        pass


class RidgeCVAnnotation:
    __estimator__ = RidgeCV

    def __init__(
        self,
        alphas: tuple = (0.1, 1.0, 10.0),
        fit_intercept: bool = True,
        normalize: bool = False,
        scoring: Union[str, Callable, None] = None,
        cv: CVType = None,
        gcv_mode: Literal["auto", "svd", "eigen"] = None,
        store_cv_values: bool = False,
        alpha_per_target: bool = False,
    ):
        pass


class RidgeClassifierAnnotation:
    __estimator__ = RidgeClassifier

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        normalize: bool = False,
        copy_X: bool = True,
        max_iter: Optional[int] = None,
        tol: float = 0.001,
        class_weight: Union[dict, Literal["balanced"], None] = None,
        solver: Literal[
            "auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"
        ] = "auto",
        random_state: RandomStateType = None,
    ):
        pass


class RidgeClassifierCVAnnotation:
    __estimator__ = RidgeClassifierCV

    def __init__(
        self,
        alphas: tuple = (0.1, 1.0, 10.0),
        fit_intercept: bool = True,
        normalize: bool = False,
        scoring: Union[str, Callable] = None,
        cv: CVType = None,
        class_weight: Union[dict, Literal["balanced"]] = None,
        store_cv_values: bool = False,
    ):
        pass


class SGDClassifierAnnotation:
    __estimator__ = SGDClassifier

    def __init__(
        self,
        loss: str = "hinge",
        penalty: Literal["l2", "l1", "elasticnet"] = "l2",
        alpha: float = 0.0001,
        l1_ratio: float = 0.15,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 0.001,
        shuffle: bool = True,
        verbose: int = 0,
        epsilon: float = 0.1,
        n_jobs: Optional[int] = None,
        random_state: RandomStateType = None,
        learning_rate: Literal[
            "constant", "optimal", "invscaling", "adaptive"
        ] = "optimal",
        eta0: float = 0.0,
        power_t: float = 0.5,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 5,
        class_weight: Union[dict, Literal["balanced"], None] = None,
        warm_start: bool = False,
        average: Union[bool, int] = False,
    ):
        pass


class SGDRegressorAnnotation:
    __estimator__ = SGDRegressor

    def __init__(
        self,
        loss: str = "squared_loss",
        penalty: Literal["l2", "l1", "elasticnet"] = "l2",
        alpha: float = 0.0001,
        l1_ratio: float = 0.15,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 0.001,
        shuffle: bool = True,
        verbose: int = 0,
        epsilon: float = 0.1,
        random_state: RandomStateType = None,
        learning_rate: Literal[
            "constant", "optimal", "invscaling", "adaptive"
        ] = "invscaling",
        eta0: float = 0.01,
        power_t: float = 0.25,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 5,
        warm_start: bool = False,
        average: Union[bool, int] = False,
    ):
        pass


class TheilSenRegressorAnnotation:
    __estimator__ = TheilSenRegressor

    def __init__(
        self,
        fit_intercept: bool = True,
        copy_X: bool = True,
        max_subpopulation: int = 10000,
        n_subsamples: Optional[int] = None,
        max_iter: int = 300,
        tol: float = 0.001,
        random_state: RandomStateType = None,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
    ):
        pass


class TweedieRegressorAnnotation:
    __estimator__ = TweedieRegressor

    def __init__(
        self,
        power: float = 0.0,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        link: Literal["auto", "identity", "log"] = "auto",
        max_iter: int = 100,
        tol: float = 0.0001,
        warm_start: bool = False,
        verbose: int = 0,
    ):
        pass
