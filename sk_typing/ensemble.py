from typing import Optional
from typing import Union
from typing import Callable

from .typing import EstimatorType
from .typing import Literal
from .typing import RandomStateType
from .typing import ArrayLike
from .typing import CVType


class AdaBoostClassifier:
    def __init__(
        self,
        base_estimator: Optional[EstimatorType] = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        algorithm: Literal["SAMME", "SAMME.R"] = "SAMME.R",
        random_state: RandomStateType = None,
    ):
        ...


class AdaBoostRegressor:
    def __init__(
        self,
        base_estimator: Optional[EstimatorType] = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        loss: Literal["linear", "square", "exponential"] = "linear",
        random_state: RandomStateType = None,
    ):
        ...


class BaggingClassifier:
    def __init__(
        self,
        base_estimator: Optional[EstimatorType] = None,
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
        ...


class BaggingRegressor:
    def __init__(
        self,
        base_estimator: Optional[EstimatorType] = None,
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
        ...


class ExtraTreesClassifier:
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
        ...


class ExtraTreesRegressor:
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
        ...


class GradientBoostingClassifier:
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
        init: Union[EstimatorType, Literal["zero"], None] = None,
        random_state: RandomStateType = None,
        max_features: Union[Literal["auto", "sqrt", "log2"], int, float] = None,
        verbose: int = 0,
        max_leaf_nodes: Optional[None] = None,
        warm_start: bool = False,
        presort: Union[bool, Literal["deprecated"]] = "deprecated",
        validation_fraction: float = 0.1,
        n_iter_no_change: Optional[int] = None,
        tol: float = 0.0001,
        ccp_alpha: float = 0.0,
    ):
        ...


class GradientBoostingRegressor:
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
        init: Union[EstimatorType, Literal["zero"], None] = None,
        random_state: RandomStateType = None,
        max_features: Union[Literal["auto", "sqrt", "log2"], int, float] = None,
        alpha: float = 0.9,
        verbose: int = 0,
        max_leaf_nodes: Optional[int] = None,
        warm_start: bool = False,
        presort: Union[bool, Literal["deprecated"]] = "deprecated",
        validation_fraction: float = 0.1,
        n_iter_no_change: Optional[int] = None,
        tol: float = 0.0001,
        ccp_alpha: float = 0.0,
    ):
        ...


class HistGradientBoostingClassifier:
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
        warm_start: bool = False,
        scoring: Union[str, Callable, None] = None,
        validation_fraction: Union[int, float, None] = 0.1,
        n_iter_no_change: Optional[int] = None,
        tol: Optional[float] = 1e-07,
        verbose: int = 0,
        random_state: RandomStateType = None,
    ):
        ...


class HistGradientBoostingRegressor:
    def __init__(
        self,
        loss: Literal["least_squares", "least_absolute_deviation"] = "least_squares",
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: Optional[int] = 31,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        max_bins: int = 255,
        warm_start: bool = False,
        scoring: Union[str, Callable, None] = None,
        validation_fraction: Union[int, float, None] = 0.1,
        n_iter_no_change: Optional[int] = None,
        tol: Optional[float] = 1e-07,
        verbose: int = 0,
        random_state: RandomStateType = None,
    ):
        ...


class IsolationForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[Literal["auto"], int, float] = "auto",
        contamination: Union[Literal["auto"], float] = "auto",
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = False,
        n_jobs: Optional[int] = None,
        behaviour: str = "deprecated",
        random_state: RandomStateType = None,
        verbose: int = 0,
        warm_start: bool = False,
    ):
        ...


class RandomForestClassifier:
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
        ...


class RandomForestRegressor:
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
        ...


class RandomTreesEmbedding:
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
        ...


class StackingClassifier:
    def __init__(
        self,
        estimators: list,
        final_estimator: Optional[EstimatorType] = None,
        cv: CVType = None,
        stack_method: Literal[
            "auto", "predict_proba", "decision_function", "predict"
        ] = "auto",
        n_jobs: Optional[int] = None,
        passthrough: bool = False,
        verbose: int = 0,
    ):
        ...


class StackingRegressor:
    def __init__(
        self,
        estimators: list,
        final_estimator: Optional[EstimatorType] = None,
        cv: CVType = None,
        n_jobs: Optional[int] = None,
        passthrough: bool = False,
        verbose: int = 0,
    ):
        ...


class VotingClassifier:
    def __init__(
        self,
        estimators: list,
        voting: Literal["hard", "soft"] = "hard",
        weights: Optional[ArrayLike] = None,
        n_jobs: Optional[int] = None,
        flatten_transform: bool = True,
    ):
        ...


class VotingRegressor:
    def __init__(
        self,
        estimators: list,
        weights: Optional[ArrayLike] = None,
        n_jobs: Optional[int] = None,
    ):
        ...
