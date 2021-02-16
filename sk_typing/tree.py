from typing import Optional
from typing import Union

from .typing import Literal
from .typing import RandomStateType


class DecisionTreeClassifier:
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
        ...


class DecisionTreeRegressor:
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
        ...


class ExtraTreeClassifier:
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
        ...


class ExtraTreeRegressor:
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
        ...
