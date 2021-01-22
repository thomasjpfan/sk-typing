from typing import Optional
from typing import Union
from typing import Callable

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


from ._typing import BaseEstimatorType
from ._typing import Literal
from ._typing import RandomStateType
from ._typing import ArrayLike
from ._typing import CVType


class AdaBoostClassifierAnnotation:
    __estimator__ = AdaBoostClassifier

    def __init__(
        self,
        base_estimator: Optional[BaseEstimatorType] = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        algorithm: Literal["SAMME", "SAMME.R"] = "SAMME.R",
        random_state: RandomStateType = None,
    ):
        pass


class AdaBoostRegressorAnnotation:
    __estimator__ = AdaBoostRegressor

    def __init__(
        self,
        base_estimator: Optional[BaseEstimatorType] = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        loss: Literal["linear", "square", "exponential"] = "linear",
        random_state: RandomStateType = None,
    ):
        pass


class BaggingClassifierAnnotation:
    __estimator__ = BaggingClassifier

    def __init__(
        self,
        base_estimator: Optional[BaseEstimatorType] = None,
        n_estimators: int = 10,
        max_samples: Union[float, int] = 1.0,
        max_features: Union[float, int] = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        warm_start: bool = False,
        n_jobs: Optional[int] = None,
        random_state: RandomStateType = None,
        verbose: int = 0,
    ):
        pass


class BaggingRegressorAnnotation:
    __estimator__ = BaggingRegressor

    def __init__(
        self,
        base_estimator: Optional[BaseEstimatorType] = None,
        n_estimators: int = 10,
        max_samples: Union[float, int] = 1.0,
        max_features: Union[float, int] = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        warm_start: bool = False,
        n_jobs: Optional[int] = None,
        random_state: RandomStateType = None,
        verbose: int = 0,
    ):
        pass


class ExtraTreesClassifierAnnotation:
    __estimator__ = ExtraTreesClassifier

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: Literal["gini", "entropy"] = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[Literal["auto", "sqrt", "log2"], int, float] = "auto",
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        min_impurity_split: Optional[float] = None,
        bootstrap: bool = False,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: RandomStateType = None,
        verbose: int = 0,
        warm_start: bool = False,
        class_weight: Union[
            Literal["balanced", "balanced_subsample"], dict, list, None
        ] = None,
        ccp_alpha: float = 0.0,
        max_samples: Union[int, float, None] = None,
    ):
        pass


class ExtraTreesRegressorAnnotation:
    __estimator__ = ExtraTreesRegressor

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: Literal["mse", "mae"] = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[Literal["auto", "sqrt", "log2"], int, float] = "auto",
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        min_impurity_split: Optional[float] = None,
        bootstrap: bool = False,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: RandomStateType = None,
        verbose: int = 0,
        warm_start: bool = False,
        ccp_alpha: float = 0.0,
        max_samples: Union[int, float, None] = None,
    ):
        pass


class GradientBoostingClassifierAnnotation:
    __estimator__ = GradientBoostingClassifier

    def __init__(
        self,
        loss: Literal["deviance", "exponential"] = "deviance",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        criterion: Literal["friedman_mse", "mse", "mae"] = "friedman_mse",
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_depth: int = 3,
        min_impurity_decrease: float = 0.0,
        min_impurity_split: Optional[float] = None,
        init: Union[BaseEstimatorType, Literal["zero"], None] = None,
        random_state: RandomStateType = None,
        max_features: Union[Literal["auto", "sqrt", "log2"], int, float] = None,
        verbose: int = 0,
        max_leaf_nodes: Optional[None] = None,
        warm_start: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: Optional[int] = None,
        tol: float = 0.0001,
        ccp_alpha: float = 0.0,
    ):
        pass


class GradientBoostingRegressorAnnotation:
    __estimator__ = GradientBoostingRegressor

    def __init__(
        self,
        loss: Literal["ls", "lad", "huber", "quantile"] = "ls",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        criterion: Literal["friedman_mse", "mse", "mae"] = "friedman_mse",
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_depth: int = 3,
        min_impurity_decrease: float = 0.0,
        min_impurity_split: Optional[float] = None,
        init: Union[BaseEstimatorType, Literal["zero"], None] = None,
        random_state: RandomStateType = None,
        max_features: Union[Literal["auto", "sqrt", "log2"], int, float] = None,
        alpha: float = 0.9,
        verbose: int = 0,
        max_leaf_nodes: Optional[int] = None,
        warm_start: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: Optional[int] = None,
        tol: float = 0.0001,
        ccp_alpha: float = 0.0,
    ):
        pass


class HistGradientBoostingClassifierAnnotation:
    __estimator__ = HistGradientBoostingClassifier

    def __init__(
        self,
        loss: Literal[
            "auto", "binary_crossentropy", "categorical_crossentropy"
        ] = "auto",
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: Optional[int] = 31,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        max_bins: int = 255,
        categorical_features: Optional[ArrayLike] = None,
        monotonic_cst: Optional[ArrayLike] = None,
        warm_start: bool = False,
        early_stopping: Union[Literal["auto"], bool] = "auto",
        scoring: Union[str, Callable, None] = "loss",
        validation_fraction: Union[int, float, None] = 0.1,
        n_iter_no_change: int = 10,
        tol: Optional[float] = 1e-07,
        verbose: int = 0,
        random_state: RandomStateType = None,
    ):
        pass


class HistGradientBoostingRegressorAnnotation:
    __estimator__ = HistGradientBoostingRegressor

    def __init__(
        self,
        loss: Literal[
            "least_squares", "least_absolute_deviation", "poisson"
        ] = "least_squares",
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: Optional[int] = 31,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        max_bins: int = 255,
        categorical_features: Optional[ArrayLike] = None,
        monotonic_cst: Optional[ArrayLike] = None,
        warm_start: bool = False,
        early_stopping: Union[Literal["auto"], bool] = "auto",
        scoring: Union[str, Callable, None] = "loss",
        validation_fraction: Union[int, float, None] = 0.1,
        n_iter_no_change: int = 10,
        tol: Optional[float] = 1e-07,
        verbose: int = 0,
        random_state: RandomStateType = None,
    ):
        pass


class IsolationForestAnnotation:
    __estimator__ = IsolationForest

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[Literal["auto"], int, float] = "auto",
        contamination: Union[Literal["auto"], float] = "auto",
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = False,
        n_jobs: Optional[int] = None,
        random_state: RandomStateType = None,
        verbose: int = 0,
        warm_start: bool = False,
    ):
        pass


class RandomForestClassifierAnnotation:
    __estimator__ = RandomForestClassifier

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: Literal["gini", "entropy"] = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[Literal["auto", "sqrt", "log2"], int, float] = "auto",
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        min_impurity_split: Optional[float] = None,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: RandomStateType = None,
        verbose: int = 0,
        warm_start: bool = False,
        class_weight: Union[
            Literal["balanced", "balanced_subsample"], dict, list, None
        ] = None,
        ccp_alpha: float = 0.0,
        max_samples: Union[int, float, None] = None,
    ):
        pass


class RandomForestRegressorAnnotation:
    __estimator__ = RandomForestRegressor

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: Literal["mse", "mae"] = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[Literal["auto", "sqrt", "log2"], int, float] = "auto",
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        min_impurity_split: Optional[float] = None,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: RandomStateType = None,
        verbose: int = 0,
        warm_start: bool = False,
        ccp_alpha: float = 0.0,
        max_samples: Union[int, float, None] = None,
    ):
        pass


class RandomTreesEmbeddingAnnotation:
    __estimator__ = RandomTreesEmbedding

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_leaf_nodes: Optional[float] = None,
        min_impurity_decrease: float = 0.0,
        min_impurity_split: Optional[float] = None,
        sparse_output: bool = True,
        n_jobs: Optional[int] = None,
        random_state: RandomStateType = None,
        verbose: int = 0,
        warm_start: bool = False,
    ):
        pass


class StackingClassifierAnnotation:
    __estimator__ = StackingClassifier

    def __init__(
        self,
        estimators: list,
        final_estimator: Optional[BaseEstimatorType] = None,
        cv: CVType = None,
        stack_method: Literal[
            "auto", "predict_proba", "decision_function", "predict"
        ] = "auto",
        n_jobs: Optional[int] = None,
        passthrough: bool = False,
        verbose: int = 0,
    ):
        pass


class StackingRegressorAnnotation:
    __estimator__ = StackingRegressor

    def __init__(
        self,
        estimators: list,
        final_estimator: Optional[BaseEstimatorType] = None,
        cv: CVType = None,
        n_jobs: Optional[int] = None,
        passthrough: bool = False,
        verbose: int = 0,
    ):
        pass


class VotingClassifierAnnotation:
    __estimator__ = VotingClassifier

    def __init__(
        self,
        estimators: list,
        voting: Literal["hard", "soft"] = "hard",
        weights: Optional[ArrayLike] = None,
        n_jobs: Optional[int] = None,
        flatten_transform: bool = True,
        verbose: bool = False,
    ):
        pass


class VotingRegressorAnnotation:
    __estimator__ = VotingRegressor

    def __init__(
        self,
        estimators: list,
        weights: Optional[ArrayLike] = None,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
    ):
        pass
