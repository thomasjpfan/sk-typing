from typing import Optional
from typing import Union

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
        base_estimator=None,
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        pass


class BaggingRegressorAnnotation:
    __estimator__ = BaggingRegressor

    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        pass


class ExtraTreesClassifierAnnotation:
    __estimator__ = ExtraTreesClassifier

    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        pass


class ExtraTreesRegressorAnnotation:
    __estimator__ = ExtraTreesRegressor

    def __init__(
        self,
        n_estimators=100,
        criterion="mse",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        pass


class GradientBoostingClassifierAnnotation:
    __estimator__ = GradientBoostingClassifier

    def __init__(
        self,
        loss="deviance",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        init=None,
        random_state=None,
        max_features=None,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=0.0001,
        ccp_alpha=0.0,
    ):
        pass


class GradientBoostingRegressorAnnotation:
    __estimator__ = GradientBoostingRegressor

    def __init__(
        self,
        loss="ls",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        init=None,
        random_state=None,
        max_features=None,
        alpha=0.9,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=0.0001,
        ccp_alpha=0.0,
    ):
        pass


class HistGradientBoostingClassifierAnnotation:
    __estimator__ = HistGradientBoostingClassifier

    def __init__(
        self,
        loss="auto",
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        max_bins=255,
        categorical_features=None,
        monotonic_cst=None,
        warm_start=False,
        early_stopping="auto",
        scoring="loss",
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-07,
        verbose=0,
        random_state=None,
    ):
        pass


class HistGradientBoostingRegressorAnnotation:
    __estimator__ = HistGradientBoostingRegressor

    def __init__(
        self,
        loss="least_squares",
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        max_bins=255,
        categorical_features=None,
        monotonic_cst=None,
        warm_start=False,
        early_stopping="auto",
        scoring="loss",
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-07,
        verbose=0,
        random_state=None,
    ):
        pass


class IsolationForestAnnotation:
    __estimator__ = IsolationForest

    def __init__(
        self,
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
    ):
        pass


class RandomForestClassifierAnnotation:
    __estimator__ = RandomForestClassifier

    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        pass


class RandomForestRegressorAnnotation:
    __estimator__ = RandomForestRegressor

    def __init__(
        self,
        n_estimators=100,
        criterion="mse",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        pass


class RandomTreesEmbeddingAnnotation:
    __estimator__ = RandomTreesEmbedding

    def __init__(
        self,
        n_estimators=100,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        sparse_output=True,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
    ):
        pass


class StackingClassifierAnnotation:
    __estimator__ = StackingClassifier

    def __init__(
        self,
        estimators,
        final_estimator=None,
        cv=None,
        stack_method="auto",
        n_jobs=None,
        passthrough=False,
        verbose=0,
    ):
        pass


class StackingRegressorAnnotation:
    __estimator__ = StackingRegressor

    def __init__(
        self,
        estimators,
        final_estimator=None,
        cv=None,
        n_jobs=None,
        passthrough=False,
        verbose=0,
    ):
        pass


class VotingClassifierAnnotation:
    __estimator__ = VotingClassifier

    def __init__(
        self,
        estimators,
        voting="hard",
        weights=None,
        n_jobs=None,
        flatten_transform=True,
        verbose=False,
    ):
        pass


class VotingRegressorAnnotation:
    __estimator__ = VotingRegressor

    def __init__(self, estimators, weights=None, n_jobs=None, verbose=False):
        pass
