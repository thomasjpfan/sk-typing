from typing import Optional
from typing import Union

from ._typing import Literal
from ._typing import RandomStateType

from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeRegressor


class DecisionTreeClassifierAnnotation:
    __estimator__ = DecisionTreeClassifier

    def __init__(
        self,
        criterion: Literal["gini", "entropy"] = "gini",
        splitter: Literal["best", "random"] = "best",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[int, float, Literal["auto", "sqrt", "log2"]] = None,
        random_state: RandomStateType = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        min_impurity_split: Optional[float] = None,
        class_weight: Union[dict, list, Literal["balanced"], None] = None,
        ccp_alpha: float = 0.0,
    ):
        pass


class DecisionTreeRegressorAnnotation:
    __estimator__ = DecisionTreeRegressor

    def __init__(
        self,
        criterion: Literal["mse", "friedman_mse", "mae", "poisson"] = "mse",
        splitter: Literal["best", "random"] = "best",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[int, float, Literal["auto", "sqrt", "log2"], None] = None,
        random_state: RandomStateType = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        min_impurity_split: Optional[float] = None,
        ccp_alpha: float = 0.0,
    ):
        pass


class ExtraTreeClassifierAnnotation:
    __estimator__ = ExtraTreeClassifier

    def __init__(
        self,
        criterion: Literal["gini", "entropy"] = "gini",
        splitter: Literal["random", "best"] = "random",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[int, float, Literal["auto", "sqrt", "log2"], None] = "auto",
        random_state: RandomStateType = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        min_impurity_split: Optional[float] = None,
        class_weight: Union[dict, list, Literal["balanced"], None] = None,
        ccp_alpha: float = 0.0,
    ):
        pass


class ExtraTreeRegressorAnnotation:
    __estimator__ = ExtraTreeRegressor

    def __init__(
        self,
        criterion: Literal["mse", "friedman_mse", "mse"] = "mse",
        splitter: Literal["random", "best"] = "random",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[int, float, Literal["auto", "sqrt", "log2"], None] = "auto",
        random_state: RandomStateType = None,
        min_impurity_decrease: float = 0.0,
        min_impurity_split: Optional[float] = None,
        max_leaf_nodes: Optional[int] = None,
        ccp_alpha: float = 0.0,
    ):
        pass
